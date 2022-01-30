"""
--
@brief:     Movement Primitives in PyTorch
"""
import csv
import os
from collections import Callable

import torch
from torch.distributions import MultivariateNormal
import numpy as np
from nmp.data_process import BatchProcess
from nmp import util


# Classes of Phase Generator
class PhaseGenerator:
    def __init__(self):
        """
        Basis class constructor
        """

    def phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Basis class phase interface
        Args:
            times: times in Tensor

        Returns: phases in Tensor

        """
        raise NotImplementedError


class LinearPhaseGenerator(PhaseGenerator):
    def __init__(self,
                 phase_velocity=1.0):
        """
        Constructor for linear phase generator
        Args:
            phase_velocity: coefficient transfer time to phase
        """
        super(LinearPhaseGenerator, self).__init__()
        self.phase_velocity = phase_velocity

    def phase(self,
              times: torch.Tensor) -> torch.Tensor:
        """
        Compute phase
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = times * self.phase_velocity
        return phase


class ExpDecayPhaseGenerator(PhaseGenerator):
    def __init__(self,
                 tau,
                 alpha_phase=3.0):
        """
        Constructor for exponential decay phase generator
        Args:
            tau: time scale (normalization factor)
                 Use duration of the movement is a good choice
            alpha_phase: decaying factor: tau * dx/dt = -alpha_phase * x
        """
        super(ExpDecayPhaseGenerator, self).__init__()
        self.tau = tau
        self.alpha_phase = alpha_phase

    def phase(self, times):
        """
        Compute phase
        Args:
            times: times Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = torch.exp(-self.alpha_phase * times / self.tau)
        return phase


class BasisGenerator:

    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10):
        """
        Constructor for basis class
        Args:
            phase_generator: phase generator
            num_basis: number of basis functions
        """
        self.num_basis = num_basis
        self.phase_generator = phase_generator

    def basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Interface to generate value of single basis function at given time
        points
        Args:
            times: times in Tensor

        Returns:
            basis functions in Tensor

        """
        raise NotImplementedError

    def basis_multi_dofs(self,
                         times: torch.Tensor,
                         num_dof: int) -> torch.Tensor:
        """
        Interface to generate value of single basis function at given time
        points
        Args:
            times: times in Tensor
            num_dof: num of Degree of freedoms
        Returns:
            basis_multi_dofs: Multiple DoFs basis functions in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]
        #
        # Shape of basis_multi_dofs
        # [*add_dim, num_dof * num_times, num_dof * num_basis]

        # Extract additional dimensions
        add_dim = list(times.shape[:-1])

        # Get single basis, shape: [*add_dim, num_times, num_basis]
        basis_single_dof = self.basis(times)
        num_times = basis_single_dof.shape[-2]
        num_basis = basis_single_dof.shape[-1]

        # Multiple Dofs, shape:
        # [*add_dim, num_times, num_dof, num_dof * num_basis]
        basis_multi_dofs = torch.zeros(*add_dim,
                                       num_dof * num_times,
                                       num_dof * num_basis)
        # Assemble
        for i in range(num_dof):
            row_indices = slice(i * num_times,
                                (i + 1) * num_times)
            col_indices = slice(i * num_basis,
                                (i + 1) * num_basis)
            basis_multi_dofs[..., row_indices, col_indices] = basis_single_dof

        # Return
        return basis_multi_dofs


class NormalizedRBFBasisGenerator(BasisGenerator):

    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10,
                 duration: float = 1.,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0):
        """
        Constructor of class RBF

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            duration: "Time Duration!" to be covered, not phase!
            basis_bandwidth_factor: ...
            num_basis_outside: basis function out side the duration
        """
        super(NormalizedRBFBasisGenerator, self).__init__(phase_generator,
                                                          num_basis)

        self.basis_bandwidth_factor = basis_bandwidth_factor
        self.num_basis_outside = num_basis_outside

        # Distance between basis centers
        basis_dist = \
            duration / (self.num_basis - 2 * self.num_basis_outside - 1)

        # RBF centers in time scope
        centers_t = \
            torch.linspace(-self.num_basis_outside * basis_dist,
                           duration + self.num_basis_outside * basis_dist,
                           self.num_basis)

        # RBF centers in phase scope
        self.centers_p = self.phase_generator.phase(centers_t)

        tmp_bandwidth = \
            torch.cat((self.centers_p[1:] - self.centers_p[:-1],
                       self.centers_p[-1:] - self.centers_p[-2:-1]), dim=-1)

        # The Centers should not overlap too much (makes w almost random due
        # to aliasing effect).Empirically chosen
        self.bandWidth = self.basis_bandwidth_factor / (tmp_bandwidth ** 2)

    def basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Generate values of basis function at given time points
        Args:
            times: times in Tensor

        Returns:
            basis: basis functions in Tensor
        """
        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of basis:
        # [*add_dim, num_times, num_basis]

        # Extract dimension
        num_times = times.shape[-1]

        # Time to phase
        phase = self.phase_generator.phase(times)

        # Add one axis (basis centers) to phase and get shape:
        # [*add_dim, num_times, num_basis]
        phase = phase[..., None]
        phase = phase.expand([*phase.shape[:-1], self.num_basis])

        # Add one axis (times) to centers in phase scope and get shape:
        # [num_times, num_basis]
        centers = self.centers_p[None, :]
        centers = centers.expand([num_times, -1])

        # Basis
        tmp = torch.einsum('...ij,...j->...ij', (phase - centers) ** 2,
                           self.bandWidth)
        basis = torch.exp(-tmp / 2)

        # Normalization
        sum_basis = torch.sum(basis, dim=-1, keepdim=True)
        basis = basis / sum_basis

        # Return
        return basis


class DMPBasisGenerator(NormalizedRBFBasisGenerator):
    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10,
                 duration: float = 1.,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0):
        """
        Constructor of class DMPBasisGenerator

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            duration: "Time Duration!" to be covered, not phase!
            basis_bandwidth_factor: ...
            num_basis_outside: basis function out side the duration
        """
        super(DMPBasisGenerator, self).__init__(phase_generator,
                                                num_basis,
                                                duration,
                                                basis_bandwidth_factor,
                                                num_basis_outside)

    def basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Generate values of basis function at given time points
        Args:
            times: times in Tensor

        Returns:
            basis: basis functions in Tensor
        """
        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of basis:
        # [*add_dim, num_times, num_basis]
        phase = self.phase_generator.phase(times)
        rbf_basis = super(DMPBasisGenerator, self).basis(times)

        # Einsum shape: [*add_dim, num_times, num_basis]
        #               [*add_dim, num_basis]
        #            -> [*add_dim, num_times, num_basis]
        dmp_basis = torch.einsum('...i,...ij->...ij', phase, rbf_basis)
        return dmp_basis


class ProMP:
    """Mini ProMP in PyTorch"""

    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int):
        """
        Constructor of ProMP
        Args:
            basis_gn: basis function value generator

            num_dof: number of Degrees of Freedoms

        """
        # Extract additional batch dimension
        self.add_dim = None

        # Assignment
        self.basis_gn = basis_gn
        self.num_dof = num_dof
        self.num_w = basis_gn.num_basis * self.num_dof

        # Initialize mu, L and obs noise to None
        self.mu = None
        self.L = None
        self.obs_sigma = None

    @property
    def cov(self):
        """
        Compute covariance using L
        Returns:
            covariance matrix of ProMP weights
        """
        assert self.L is not None
        cov = torch.einsum('...ij,...kj->...ik',
                           self.L,
                           self.L)
        return cov

    def sample_trajectories(self, times, n_samples=1) -> torch.Tensor:
        """
        Sample trajectories from current ProMP

        Args:
            times: times in Tensor
            n_samples: num of trajectories to be sampled

        Returns:
            sampled trajectories
        """
        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of trajectories
        # [*add_dim, num_smp, num_times, num_dof]
        #
        # assert self.mu is not None and self.L is not None
        #
        # Get basis of all Dofs
        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = self.basis_gn.basis_multi_dofs(times, self.num_dof)
        #
        # Sample weights
        # Shape: [num_smp, *add_dim, num_dof * num_basis]
        weights = MultivariateNormal(loc=self.mu,
                                     scale_tril=self.L,
                                     validate_args=False).sample([n_samples])

        # Get trajectories
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [num_smp, *add_dim, num_dof * num_basis]
        #            -> [*add_dim, num_smp, num_dof * num_times]
        trajectories = torch.einsum('...jk,i...k->...ij',
                                    basis_multi_dofs,
                                    weights)
        # Reshape: [*add_dim, num_smp, num_dof * num_times]
        #       -> [*add_dim, num_smp, num_dof, num_times]
        irr_dims = trajectories.shape[:-1]
        trajectories = trajectories.reshape(*irr_dims, self.num_dof, -1)

        # Switch axis: (dof, time) -> (time, dof)
        # Note reshaping and switching should not be in one einsum operation
        trajectories = torch.einsum('...ij->...ji', trajectories)

        # return trajectories
        return trajectories

    def set_promp(self, add_dim: list = None,
                  mu=None, L=None, obs_sigma=None):
        """
        Setting promp mean and l, obs_sigma
        Args:
            add_dim: additional dimension of batch
            mu: mean of ProMP weights
            L: Cholesky Decomposition of ProMP weights covariance
            obs_sigma: observation noise, Sigma_y

        Returns: None

        """

        # Shape of mu:
        # [*add_dim, num_dof * num_basis]
        #
        # Shape of L:
        # [*add_dim, num_dof * num_basis, num_dof * num_basis]
        #
        # Shape of obs_sigma:
        # [*add_dim, num_dof]

        if add_dim is not None:
            self.add_dim = add_dim

        if mu is not None:
            assert self.add_dim is not None
            assert list(mu.shape) == [*self.add_dim, self.num_w]
            self.mu = mu

        if L is not None:
            assert self.add_dim is not None
            assert list(L.shape) == [*self.add_dim,
                                     self.num_w,
                                     self.num_w]
            self.L = L

        if obs_sigma is not None:
            assert self.add_dim is not None
            assert list(obs_sigma.shape) == [*self.add_dim,
                                             self.num_dof]
            self.obs_sigma = obs_sigma

    def get_traj_mean_and_cov(self, times: torch.Tensor):
        """
        Compute trajectory mean and covariance
        Args:
            times: time points

        Returns: traj_mean, traj_cov

        """
        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of traj_mean
        # [*add_dim, num_dof * num_times]
        #
        # Shape of traj_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        assert self.mu is not None

        # Get basis of all Dofs
        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dof = self.basis_gn.basis_multi_dofs(times, self.num_dof)

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_basis]
        #            -> [*add_dim, num_dof * num_times]
        traj_mean = torch.einsum('...ij,...j->...i', basis_multi_dof, self.mu)

        if self.L is None:
            return traj_mean, None
        else:
            # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
            #               [*add_dim, num_dof * num_basis, num_dof * num_basis]
            #               [*add_dim, num_dof * num_times, num_dof * num_basis]
            #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
            traj_cov = torch.einsum('...ik,...kl,...jl->...ij',
                                    basis_multi_dof,
                                    self.cov,
                                    basis_multi_dof)
            return traj_mean, traj_cov

    @staticmethod
    def draw_basis_functions(promp, times):
        """
        Helper function drawing basis function layout
        Args:
            promp: raw promp framework that you want to draw
            times: draw basis function in this time sequence

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        plt.figure()
        legend_list = list()
        for i in range(promp.basis_gn.num_basis):
            w = torch.zeros(size=[promp.num_w])
            w[..., i] += 1
            promp.set_promp(add_dim=[], mu=w)
            traj_mean_val = promp.get_traj_mean(times)[..., 0].squeeze()
            traj_mean_val = traj_mean_val.cpu().numpy()
            t = times.squeeze().cpu().numpy()
            plt.plot(t, traj_mean_val)
            legend_list.append("function_{}".format(i + 1))
        plt.title("Basis function layout")
        plt.legend(legend_list)
        plt.show()

    def weights_learner(self,
                        pos,
                        times,
                        reg: float = 1e-9):
        """

        Args:
            pos: trajectory from which weights should learned
            times: times of the position
            reg: factor for numerical stability

        Returns:
            weights
        """
        # Shape of pos:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of times:
        # [*add_dim, num_times]

        assert pos.shape[:-1] == times.shape

        # Transfer times to Tensor
        times = torch.Tensor(times)

        # Get multiple dof basis function values
        # Tensor [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dof = self.basis_gn.basis_multi_dofs(times, self.num_dof)

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_times, num_dof * num_basis],
        #            -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
        A = torch.einsum('...ki,...kj->...ij', basis_multi_dof,
                         basis_multi_dof)
        A += (torch.eye(self.num_w) * reg)

        # Reorder axis [*add_dim, num_times, num_dof]
        #           -> [*add_dim, num_dof, num_times]
        pos = torch.Tensor(pos)
        pos = torch.einsum('...ij->...ji', pos)

        # Reshape: [*add_dim, num_dof, num_times]
        #       -> [*add_dim, num_dof * num_times]
        irr_dims = pos.shape[:-2]
        pos = pos.reshape(*irr_dims, -1)

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_times],
        #            -> [*add_dim, num_dof * num_basis]
        B = torch.einsum('...ki,...k->...i', basis_multi_dof, pos)

        # Solve for weights, shape [*add_dim, num_dof * num_basis]
        w = torch.linalg.solve(A, B)

        # Return
        return w


class DMP:
    def __init__(self,
                 basis_gn,
                 num_dof: int,
                 **kwargs):
        """
        Constructor of NDP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            kwargs: keyword arguments
        """
        # Extract additional batch dimension
        self.add_dim = list()

        # This basis and phase generators are for pre-compute only!
        self.basis_gn = basis_gn
        self.phase_gn = basis_gn.phase_generator

        # Number of basis in IDMP = weights basis + goal basis
        self.num_basis = basis_gn.num_basis
        self.num_basis_g = self.num_basis + 1
        self.num_dof = num_dof

        # Weights now contain old weights plus goal
        self.num_w_g = self.num_basis_g * self.num_dof

        # Control parameters
        self.alpha = kwargs["alpha"]
        self.beta = self.alpha / 4

        # Time scaling parameter, by default, duration = tau
        self.tau = kwargs["tau"]
        self.dt = kwargs["dt"]
        self.num_times = int(self.tau / self.dt) + 1

        # Learnable parameters
        self.w = None
        self.g = None

        # Initialize boundary conditions, time, position, velocity
        self.bc_time = None
        self.bc_pos = None
        self.bc_vel = None

    def split_weights_goal(self, w_g):
        """

        Args:
            w_g:

        Returns:
            w:
            g:

        """
        w_g = w_g.reshape([*w_g.shape[:-1], self.num_dof, self.num_basis_g])
        w = w_g[..., :-1]
        g = w_g[..., -1]
        # return w.reshape[-1], g.reshape[-1]
        return w, g

    def set_dmp(self,
                add_dim: list = None,
                w_g=None,
                bc_dict=None):
        """
        Setting dmp learnable parameters and boundary conditions
        Args:
            add_dim: additional batch dimensions
            w_g: weights and goal
            bc_dict: boundary condition dictionary

        Returns: None

        """
        # Shape of w_g:
        # [*add_dim, num_dof * num_basis_g]
        #
        # Shape of bc_time:
        # [*add_dim]
        #
        # Shape of bc_pos:
        # [*add_dim, num_dof]
        #
        # Shape of bc_vel:
        # [*add_dim, num_dof]

        if add_dim is not None:
            self.add_dim = add_dim

        if w_g is not None:
            assert self.add_dim is not None
            assert list(w_g.shape) == [*self.add_dim, self.num_w_g]
            self.w, self.g = self.split_weights_goal(w_g)

        if bc_dict is not None:
            assert self.add_dim is not None
            bc_time = bc_dict["bc_time"]
            bc_pos = bc_dict["bc_pos"]
            bc_vel = bc_dict["bc_vel"]
            assert list(bc_time.shape) == [*self.add_dim]
            assert list(bc_pos.shape) == list(bc_vel.shape) == [*self.add_dim,
                                                                self.num_dof]
            self.bc_time = bc_time
            self.bc_pos = bc_pos
            self.bc_vel = bc_vel

    def compute_trajectory(self, times: torch.Tensor):
        """
        Compute DMP trajectory
        Args:
            times: time points

        Returns:
            pos: trajectory position
            vel: trajectory velocity
        """
        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of pos
        # [*add_dim, num_times, num_dof]
        #
        # Shape of vel
        # [*add_dim, num_times, num_dof]

        # Get basis, shape [*add_dim, num_times, num_basis]
        basis = self.basis_gn.basis(times)

        # Get forcing function
        # Einsum shape: [*add_dim, num_times, num_basis]
        #               [*add_dim, num_dof, num_basis]
        #            -> [*add_dim, num_times, num_dof]
        f = torch.einsum('...ik,...jk->...ij', basis, self.w)

        # Initialize trajectory position, velocity and acceleration
        pos = torch.zeros([*self.add_dim, times.shape[-1], self.num_dof])
        vel = torch.zeros([*self.add_dim, times.shape[-1], self.num_dof])
        # acc = torch.zeros([*self.add_dim, self.num_times, self.num_dof])

        # Boundary condition
        assert torch.all(torch.abs(self.bc_time - times[..., 0]) < 1e-8)
        pos[..., 0, :] = self.bc_pos
        vel[..., 0, :] = self.bc_vel

        # Apply Euler Integral
        for i in range(times.shape[-1] - 1):
            acc = (self.alpha * (self.beta * (self.g - pos[..., i, :])
                                 - self.tau * vel[..., i, :]) + f[..., i, :]) \
                  / self.tau ** 2
            vel[..., i + 1, :] = vel[..., i, :] + self.dt * acc
            pos[..., i + 1, :] = pos[..., i, :] + self.dt * vel[..., i + 1, :]

        # Return
        return pos, vel

    def get_traj_mean_and_cov(self, times: torch.Tensor):
        """
        External calling interface
        Args:
            times: time

        Returns:
            position,
            position_cov = None,
            velocity,
            velocity_cov = None
        """
        pos, vel = self.compute_trajectory(times)
        pos_cov = None
        vel_cov = None
        return pos, pos_cov, vel, vel_cov


class IDMP:
    """Integral version of DMP"""

    def __init__(self,
                 basis_gn,
                 num_dof: int,
                 **kwargs):
        """
        Constructor of IDMP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            kwargs: keyword arguments
        """
        # Extract additional batch dimension
        self.add_dim = list()

        # This basis and phase generators are for pre-compute only!
        self.basis_gn = basis_gn
        self.phase_gn = basis_gn.phase_generator

        # Number of basis in IDMP = weights basis + goal basis
        self.num_basis = basis_gn.num_basis
        self.num_basis_g = self.num_basis + 1
        self.num_dof = num_dof

        # Weights now contain old weights plus goal
        self.num_w = self.num_basis_g * self.num_dof

        # Control parameters
        self.alpha = kwargs["alpha"]
        self.beta = self.alpha / 4

        # Time scaling parameter
        self.tau = kwargs["tau"]

        # Initialize weights (goal included) mean and Cov Cholesky
        self.mu = None
        self.L = None

        # Initialize boundary conditions, time index, position, velocity
        self.bc_index = None
        self.bc_pos = None
        self.bc_vel = None

        # Pre-compute times, shape: [num_pc_times]
        self.num_pc_times = kwargs["num_pc_times"]
        self.pc_time_start = kwargs["pc_time_start"]
        self.pc_time_stop = kwargs["pc_time_stop"]
        self.pc_times = torch.linspace(self.pc_time_start, self.pc_time_stop,
                                       self.num_pc_times)

        # Only used for weights learner to compute velocity boundary condition
        self.dt = kwargs["dt"]

        # Pre-computed terms
        self.y_1_value = None
        self.y_2_value = None
        self.dy_1_value = None
        self.dy_2_value = None
        self.pos_basis = None
        self.vel_basis = None

        # Pre-compute
        self._pre_compute()

    @property
    def cov(self):
        """
        Compute covariance using L
        Returns:
            covariance matrix of DMP weights
        """
        assert self.L is not None
        cov = torch.einsum('...ij,...kj->...ik',
                           self.L,
                           self.L)
        return cov

    def set_dmp(self,
                add_dim: list = None,
                mu=None,
                L=None,
                bc_dict=None):
        """
        Setting dmp mean and l, obs_sigma
        Args:
            add_dim: additional batch dimensions
            mu: mean of DMP weights
            L: Cholesky Decomposition of DMP weights covariance
            bc_dict: boundary condition dictionary

        Returns: None

        """
        # Shape of mu:
        # [*add_dim, num_dof * num_basis_g]
        #
        # Shape of L:
        # [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        #
        # Shape of bc_index:
        # [*add_dim]
        #
        # Shape of bc_pos:
        # [*add_dim, num_dof]
        #
        # Shape of bc_vel:
        # [*add_dim, num_dof]

        if add_dim is not None:
            self.add_dim = add_dim

        if mu is not None:
            assert self.add_dim is not None
            assert list(mu.shape) == [*self.add_dim, self.num_w]
            self.mu = mu

        if L is not None:
            assert self.add_dim is not None
            assert list(L.shape) == [*self.add_dim,
                                     self.num_w,
                                     self.num_w]
            self.L = L

        if bc_dict is not None:
            assert self.add_dim is not None
            bc_index = bc_dict["bc_index"]
            bc_pos = bc_dict["bc_pos"]
            bc_vel = bc_dict["bc_vel"]
            assert list(bc_index.shape) == [*self.add_dim]
            assert list(bc_pos.shape) == list(bc_vel.shape) == [*self.add_dim,
                                                                self.num_dof]
            self.bc_index = bc_index.long()
            self.bc_pos = bc_pos
            self.bc_vel = bc_vel

    def _pre_compute(self):
        """
        Pre-compute the integral form basis

        Returns:
            None

        """
        # Shape of pc_times
        # [num_pc_times]

        # Shape of y_1_value, y_2_value, dy_1_value, dy_2_value:
        # [num_pc_times]
        #
        # Shape of q_1_value, q_2_value:
        # [num_pc_times]
        #
        # Shape of p_1_value, p_2_value:
        # [num_pc_times, num_basis]
        #
        # Shape of pos_basis, vel_basis:
        # [num_pc_times, num_basis_g]
        # Note: num_basis_g = num_basis + 1

        # y_1 and y_2
        self.y_1_value = torch.exp(-0.5 * self.alpha / self.tau * self.pc_times)
        self.y_2_value = self.pc_times * self.y_1_value

        self.dy_1_value = -0.5 * self.alpha / self.tau * self.y_1_value
        self.dy_2_value = -0.5 * self.alpha / self.tau * self.y_2_value \
                          + self.y_1_value

        # q_1 and q_2
        q_1_value = \
            (0.5 * self.alpha / self.tau * self.pc_times - 1) \
            * torch.exp(0.5 * self.alpha / self.tau * self.pc_times) + 1
        q_2_value = \
            0.5 * self.alpha / self.tau \
            * (torch.exp(0.5 * self.alpha / self.tau * self.pc_times) - 1)

        # p_1 and p_2
        # Get basis of one DOF, shape [num_pc_times, num_basis]
        basis_single_dof = self.basis_gn.basis(self.pc_times)
        assert list(basis_single_dof.shape) == [*self.pc_times.shape,
                                                self.num_basis]

        dp_1_value = \
            torch.einsum('...i,...ij->...ij',
                         self.pc_times / self.tau ** 2
                         * torch.exp(self.alpha * self.pc_times / self.tau / 2),
                         basis_single_dof)
        dp_2_value = \
            torch.einsum('...i,...ij->...ij',
                         1 / self.tau ** 2
                         * torch.exp(self.alpha * self.pc_times / self.tau / 2),
                         basis_single_dof)

        p_1_value = torch.zeros(size=dp_1_value.shape)
        p_2_value = torch.zeros(size=dp_2_value.shape)

        for i in range(self.pc_times.shape[0]):
            p_1_value[i] = \
                torch.trapz(dp_1_value[:i + 1], self.pc_times[:i + 1], dim=0)
            p_2_value[i] = \
                torch.trapz(dp_2_value[:i + 1], self.pc_times[:i + 1], dim=0)

        # Compute integral form basis values
        pos_basis_w = p_2_value * self.y_2_value[:, None] \
                      - p_1_value * self.y_1_value[:, None]
        pos_basis_g = q_2_value * self.y_2_value \
                      - q_1_value * self.y_1_value
        vel_basis_w = p_2_value * self.dy_2_value[:, None] \
                      - p_1_value * self.dy_1_value[:, None]
        vel_basis_g = q_2_value * self.dy_2_value \
                      - q_1_value * self.dy_1_value
        self.pos_basis = torch.cat([pos_basis_w, pos_basis_g[:, None]], dim=-1)
        self.vel_basis = torch.cat([vel_basis_w, vel_basis_g[:, None]], dim=-1)

    def compute_traj(self, time_indices=None, reg=1e-4):
        """
        Compute the position and velocity, mean and cov of the trajectory at
        desired time steps

        Args:
            time_indices: indices of the time where traj should be computed,
                None if use all pre-computed values
            reg: regularization term to avoid 0 uncertainty

        Returns:
            position and velocity

        """
        # Shape of time_indices:
        # [*add_dim, num_times] or None
        #
        # Shape of pos, vel:
        # [*add_dim, num_dof * num_times]
        #
        # Shape of pos_sigma, vel_sigma:
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Evaluate boundary condition values
        if time_indices is None and self.L is not None:
            bcolors = util.bcolors
            print(f"{bcolors.WARNING}Warning: given no time indices may lead "
                  f"to high cov computation error, because by the "
                  f"regularization term is proportional to the max term in "
                  f"the cov, which may be huge for the time-steps before the "
                  f"boundary condition time point. ")

        bc_values_dict = \
            self.compute_traj_coefficients(add_dim=self.add_dim,
                                           bc_index=self.bc_index,
                                           time_indices=time_indices)

        pos_coef_1 = bc_values_dict["pos_coef_1"]
        pos_coef_2 = bc_values_dict["pos_coef_2"]
        pos_coef_3 = bc_values_dict["pos_coef_3"]
        pos_coef_4 = bc_values_dict["pos_coef_4"]

        vel_coef_1 = bc_values_dict["vel_coef_1"]
        vel_coef_2 = bc_values_dict["vel_coef_2"]
        vel_coef_3 = bc_values_dict["vel_coef_3"]
        vel_coef_4 = bc_values_dict["vel_coef_4"]

        pos_basis_bc_multi_dofs = bc_values_dict["pos_basis_bc_multi_dofs"]
        vel_basis_bc_multi_dofs = bc_values_dict["vel_basis_bc_multi_dofs"]
        pos_basis_multi_dofs = bc_values_dict["pos_basis_multi_dofs"]
        vel_basis_multi_dofs = bc_values_dict["vel_basis_multi_dofs"]

        # Position and velocity part 1 and part 2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof]
        #            -> [*add_dim, num_dof, num_times]
        pos_1 = torch.einsum('...j,...i->...ij', pos_coef_1, self.bc_pos)
        vel_1 = torch.einsum('...j,...i->...ij', vel_coef_1, self.bc_pos)
        pos_2 = torch.einsum('...j,...i->...ij', pos_coef_2, self.bc_vel)
        vel_2 = torch.einsum('...j,...i->...ij', vel_coef_2, self.bc_vel)

        # Reshape: [*add_dim, num_dof, num_times]
        #       -> [*add_dim, num_dof * num_times]
        pos_1 = torch.reshape(pos_1, [*self.add_dim, -1])
        vel_1 = torch.reshape(vel_1, [*self.add_dim, -1])
        pos_2 = torch.reshape(pos_2, [*self.add_dim, -1])
        vel_2 = torch.reshape(vel_2, [*self.add_dim, -1])

        # Position and velocity part 3_1 and 3_2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof, num_times, num_basis_g * num_dof]
        pos_3_1 = torch.einsum('...j,...ik->...ijk', pos_coef_3,
                               pos_basis_bc_multi_dofs)
        pos_3_2 = torch.einsum('...j,...ik->...ijk', pos_coef_4,
                               vel_basis_bc_multi_dofs)
        vel_3_1 = torch.einsum('...j,...ik->...ijk', vel_coef_3,
                               pos_basis_bc_multi_dofs)
        vel_3_2 = torch.einsum('...j,...ik->...ijk', vel_coef_4,
                               vel_basis_bc_multi_dofs)

        # Reshape: [*add_dim, num_dof, num_times, num_basis_g * num_dof]
        #       -> [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        pos_3_1 = torch.reshape(pos_3_1, [*self.add_dim, -1, self.num_w])
        pos_3_2 = torch.reshape(pos_3_2, [*self.add_dim, -1, self.num_w])
        vel_3_1 = torch.reshape(vel_3_1, [*self.add_dim, -1, self.num_w])
        vel_3_2 = torch.reshape(vel_3_2, [*self.add_dim, -1, self.num_w])

        pos_3_3 = pos_3_1 + pos_3_2 + pos_basis_multi_dofs
        vel_3_3 = vel_3_1 + vel_3_2 + vel_basis_multi_dofs

        # Position and velocity part 3
        # Einsum shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof],
        #               [*add_dim, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof * num_times]
        pos_3 = torch.einsum('...ij,...j->...i', pos_3_3, self.mu)
        vel_3 = torch.einsum('...ij,...j->...i', vel_3_3, self.mu)

        pos = pos_1 + pos_2 + pos_3
        vel = vel_1 + vel_2 + vel_3

        pos_sigma, vel_sigma = None, None
        if self.L is None:
            return pos, vel, pos_sigma, vel_sigma

        # Uncertainty of position and velocity
        # Einsum shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof],
        #               [*add_dim, num_basis_g * num_dof, num_basis_g * num_dof]
        #               [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        pos_sigma = torch.einsum('...ik,...kl,...jl->...ij',
                                 pos_3_3, self.cov, pos_3_3)
        vel_sigma = torch.einsum('...ik,...kl,...jl->...ij',
                                 vel_3_3, self.cov, vel_3_3)

        # Add regularization term for numerical stability
        reg_term_pos = torch.max(torch.einsum('...ii->...i',
                                              pos_sigma)).item() * reg
        reg_term_vel = torch.max(torch.einsum('...ii->...i',
                                              vel_sigma)).item() * reg
        pos_sigma += torch.eye(pos_sigma.shape[-1]) * reg_term_pos
        vel_sigma += torch.eye(vel_sigma.shape[-1]) * reg_term_vel

        return pos, vel, pos_sigma, vel_sigma

    def get_traj_mean_and_cov(self, times_indices):
        """
        External calling interface for NN usage

        Args:
            times_indices: compute trajectories at these time steps

        Returns:

        """
        # Shape of times_indices
        # [*add_dim, num_times]
        pos, vel, pos_sigma, vel_sigma = self.compute_traj(times_indices)
        return pos, pos_sigma, vel, vel_sigma

    def weights_learner(self, pos, time_indices=None, reg=1e-9):
        """
        Learn DMP weights given trajectory position
        Use the initial position and velocity as boundary condition

        Args:
            pos: trajectory position in batch
            time_indices: indices of the time where traj should be computed,
                None if use all pre-computed values
            reg: regularization term

        Returns:
            weights: learned dmp weights
            bc_index: boundary condition index
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
        """
        # Shape of pos:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of bc_index:
        # [*add_dim]
        #
        # Shape of weights:
        # [*add_dim, num_w=num_basis_g * num_dof]
        #
        # Shape of bc_pos:
        # [*add_dim, num_dof]
        #
        # Shape of bc_vel:
        # [*add_dim, num_dof]

        # Assert pos shape
        assert list(pos.shape[-2:]) == [self.num_pc_times, self.num_dof]
        pos = torch.Tensor(pos)
        add_dim = pos.shape[:-2]
        bc_index = torch.zeros(*add_dim) if len(add_dim) != 0 else 0

        # Evaluate boundary condition values
        bc_values_dict = self.compute_traj_coefficients(add_dim=add_dim,
                                                        bc_index=bc_index,
                                                        time_indices=time_indices)
        pos_coef_1 = bc_values_dict["pos_coef_1"]
        pos_coef_2 = bc_values_dict["pos_coef_2"]
        pos_coef_3 = bc_values_dict["pos_coef_3"]
        pos_coef_4 = bc_values_dict["pos_coef_4"]
        pos_basis_bc_multi_dofs = bc_values_dict["pos_basis_bc_multi_dofs"]
        vel_basis_bc_multi_dofs = bc_values_dict["vel_basis_bc_multi_dofs"]
        pos_basis_multi_dofs = bc_values_dict["pos_basis_multi_dofs"]

        # Position and velocity part 1
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof]
        #            -> [*add_dim, num_dof, num_times]
        bc_pos = pos[..., 0, :]
        pos_1 = torch.einsum('...j,...i->...ij', pos_coef_1, bc_pos)

        # Position and velocity part 2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof]
        #            -> [*add_dim, num_dof, num_times]
        bc_vel = torch.diff(pos, dim=-2)[..., 0, :] / self.dt
        pos_2 = torch.einsum('...j,...i->...ij', pos_coef_2, bc_vel)

        # Position and velocity part 3_1 and 3_2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof, num_times, num_basis_g * num_dof]
        pos_3_1 = torch.einsum('...j,...ik->...ijk', pos_coef_3,
                               pos_basis_bc_multi_dofs)
        pos_3_2 = torch.einsum('...j,...ik->...ijk', pos_coef_4,
                               vel_basis_bc_multi_dofs)
        # Reshape: [*add_dim, num_dof, num_times, num_basis_g * num_dof]
        #       -> [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        pos_3_1 = torch.reshape(pos_3_1, [*self.add_dim, -1, self.num_w])
        pos_3_2 = torch.reshape(pos_3_2, [*self.add_dim, -1, self.num_w])

        pos_3_3 = pos_3_1 + pos_3_2 + pos_basis_multi_dofs

        # Solve this: Aw = B -> w = A^{-1} B
        # Einsum_shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #               [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #            -> [*add_dim, num_basis_g * num_dof, num_basis_g * num_dof]
        A = torch.einsum('...ki,...kj->...ij', pos_3_3, pos_3_3)
        A += torch.eye(self.num_w) * reg

        # Swap axis and reshape: [*add_dim, num_times, num_dof]
        #                     -> [*add_dim, num_dof, num_times]
        pos = torch.einsum("...ij->...ji", pos)

        # Position minus boundary condition terms,
        pos_wg = pos - pos_1 - pos_2
        # Reshape [*add_dim, num_dof, num_times]
        #      -> [*add_dim, num_dof * num_times]
        pos_wg = pos_wg.reshape([*self.add_dim, -1])

        # Einsum_shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #               [*add_dim, num_dof * num_times]
        #            -> [*add_dim, num_basis_g * num_dof]
        B = torch.einsum('...ki,...k->...i', pos_3_3, pos_wg)

        # Shape of weights: [*add_dim, num_w=num_basis_g * num_dof]
        weights = torch.linalg.solve(A, B)

        return weights, bc_index, bc_pos, bc_vel

    def compute_traj_coefficients(self, add_dim: list, bc_index,
                                  time_indices=None):
        """
        Evaluate boundary condition values of the pre-computed terms, as well as
        the formed up position and velocity coefficients
        Args:
            add_dim: list describing the additional batch dimensions
            bc_index: boundary condition index where the values should be
                evaluated
            time_indices: indices of the time where traj should be computed,
                None if using all pre-computed values

        Returns:
            bc_values_dict: a dictionary containing all evaluated values
        """
        # Shape of time_indices:
        # [*add_dim, num_times] or None

        # Shape of bc_index:
        # [*add_dim]

        # Given index, extract boundary condition values
        # Shape [num_pc_times] -> [*add_dim]
        y_1_bc = self.y_1_value[bc_index]
        y_2_bc = self.y_2_value[bc_index]
        dy_1_bc = self.dy_1_value[bc_index]
        dy_2_bc = self.dy_2_value[bc_index]

        # Shape [num_pc_times, num_basis_g] -> [*add_dim, num_basis_g]
        pos_basis_bc = self.pos_basis[bc_index, :]
        vel_basis_bc = self.vel_basis[bc_index, :]

        # Determinant of boundary condition,
        # Shape: [*add_dim]
        det = y_1_bc * dy_2_bc - y_2_bc * dy_1_bc

        if time_indices is not None:
            num_times = time_indices.shape[-1]
            time_indices = time_indices.long()

            # Get pre-computed values at desired time indices
            # Shape [num_pc_times] -> [*add_dim, num_times]
            y_1_value = self.y_1_value[time_indices]
            y_2_value = self.y_2_value[time_indices]
            dy_1_value = self.dy_1_value[time_indices]
            dy_2_value = self.dy_2_value[time_indices]

            # Shape [num_pc_times, num_basis_g]
            #    -> [*add_dim, num_times, num_basis_g]
            pos_basis = self.pos_basis[time_indices, :]
            vel_basis = self.vel_basis[time_indices, :]

            # Einstein summation convention string
            einsum_eq = "...,...i->...i"

        else:
            # Use all time indices
            num_times = self.num_pc_times
            y_1_value = self.y_1_value
            y_2_value = self.y_2_value
            dy_1_value = self.dy_1_value
            dy_2_value = self.dy_2_value
            pos_basis = self.pos_basis
            vel_basis = self.vel_basis

            # Einstein summation convention string
            einsum_eq = "...,i->...i"

        # Initialize the result dictionary
        bc_values_dict = dict()

        # Compute coefficients to form up traj position and velocity
        # If use all time indices:
        # Shape: [*add_dim], [num_times] -> [*add_dim, num_times]
        # Else:
        # Shape: [*add_dim], [*add_dim, num_times] -> [*add_dim, num_times]
        bc_values_dict["pos_coef_1"] = \
            torch.einsum(einsum_eq, dy_2_bc / det, y_1_value) \
            - torch.einsum(einsum_eq, dy_1_bc / det, y_2_value)
        bc_values_dict["pos_coef_2"] = \
            torch.einsum(einsum_eq, y_1_bc / det, y_2_value) \
            - torch.einsum(einsum_eq, y_2_bc / det, y_1_value)
        bc_values_dict["pos_coef_3"] = \
            torch.einsum(einsum_eq, dy_1_bc / det, y_2_value) \
            - torch.einsum(einsum_eq, dy_2_bc / det, y_1_value)
        bc_values_dict["pos_coef_4"] = \
            torch.einsum(einsum_eq, y_2_bc / det, y_1_value) \
            - torch.einsum(einsum_eq, y_1_bc / det, y_2_value)
        bc_values_dict["vel_coef_1"] = \
            torch.einsum(einsum_eq, dy_2_bc / det, dy_1_value) \
            - torch.einsum(einsum_eq, dy_1_bc / det, dy_2_value)
        bc_values_dict["vel_coef_2"] = \
            torch.einsum(einsum_eq, y_1_bc / det, dy_2_value) \
            - torch.einsum(einsum_eq, y_2_bc / det, dy_1_value)
        bc_values_dict["vel_coef_3"] = \
            torch.einsum(einsum_eq, dy_1_bc / det, dy_2_value) \
            - torch.einsum(einsum_eq, dy_2_bc / det, dy_1_value)
        bc_values_dict["vel_coef_4"] = \
            torch.einsum(einsum_eq, y_2_bc / det, dy_1_value) \
            - torch.einsum(einsum_eq, y_1_bc / det, dy_2_value)

        # Generate blocked basis boundary condition
        # Shape: [*add_dim, num_basis_g] ->
        # [*add_dim, num_dof, num_dof * num_basis_g]
        pos_basis_bc_multi_dofs = torch.zeros((*add_dim,
                                               self.num_dof,
                                               self.num_w))
        vel_basis_bc_multi_dofs = torch.zeros((*add_dim,
                                               self.num_dof,
                                               self.num_w))

        # Generated blocked basis
        # Shape: [*add_dim, num_times, num_basis_g] ->
        # [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        pos_basis_multi_dofs = torch.zeros(*add_dim,
                                           num_times * self.num_dof,
                                           self.num_w)
        vel_basis_multi_dofs = torch.zeros(*add_dim,
                                           num_times * self.num_dof,
                                           self.num_w)
        for i in range(self.num_dof):
            row_indices = slice(i * num_times,
                                (i + 1) * num_times)
            col_indices = slice(i * self.num_basis_g,
                                (i + 1) * self.num_basis_g)
            pos_basis_bc_multi_dofs[..., i, col_indices] = pos_basis_bc
            vel_basis_bc_multi_dofs[..., i, col_indices] = vel_basis_bc
            pos_basis_multi_dofs[..., row_indices, col_indices] = pos_basis
            vel_basis_multi_dofs[..., row_indices, col_indices] = vel_basis

        bc_values_dict["pos_basis_bc_multi_dofs"] = pos_basis_bc_multi_dofs
        bc_values_dict["vel_basis_bc_multi_dofs"] = vel_basis_bc_multi_dofs
        bc_values_dict["pos_basis_multi_dofs"] = pos_basis_multi_dofs
        bc_values_dict["vel_basis_multi_dofs"] = vel_basis_multi_dofs

        return bc_values_dict

    def save_pre_compute_result(self):
        pass


class MPProcess:
    @staticmethod
    def add_mp_to_dataframes(list_pd_df: list,
                             list_pd_df_static: list,
                             weights_learner: Callable,
                             **kwargs):
        """
        Add MP weights to dataset

        Args:
            list_pd_df: list of pandas DataFrames for time variant data
            list_pd_df_static: list of pandas DataFrames for time invariant data
            weights_learner: MP weights learner
            **kwargs: keyword arguments

        Returns:
            list_pd_df_static: dataset of time invariant data with MP weights

        """
        # Overwrite if weights already exist in dataset
        overwrite = kwargs["overwrite"]

        # Search for data given keys in dataset:
        time_key: str = kwargs["time_key"]
        data_keys: list = kwargs["data_keys"]

        # Store mp weight in this key:
        mp_key: str = kwargs["mp_key"]

        # Type of mp
        mp_type = kwargs["mp"]["type"]

        # Loop over all pandas dataframes
        for pd_df, pd_df_static in zip(list_pd_df, list_pd_df_static):

            # Store times and values
            pd_df_times = []
            pd_df_values = []

            # Loop over all degree of freedoms to be generated in MP
            for name in data_keys:
                pd_df_times.append(pd_df[time_key][pd_df[name].notna()].values)
                pd_df_values.append(pd_df[name][pd_df[name].notna()].values)

            # Check data validity
            assert all([np.array_equal(pd_df_times[0], times)
                        for times in pd_df_times[1:]]), \
                "All dof in MP should share same time points."

            # Form up trajectory
            traj = np.vstack(pd_df_values).transpose()

            # Check existence of MP in dataset
            assert (mp_key not in pd_df_static.columns) or overwrite, \
                "MP weights already exist, set overwrite to True " \
                "to overwrite them."

            # Check MP case and save to DataFrame
            if mp_type == "promp":
                w = weights_learner(pos=traj,
                                    times=pd_df_times[0]).cpu().numpy()
                pd_df_static[mp_key] = [w.flatten()]
            elif mp_type == "idmp":
                w, bc_idx, bc_pos, bc_vel = weights_learner(pos=traj)
                w = w.cpu().numpy()
                pd_df_static[mp_key] = [w.flatten()]
                pd_df_static["bc_index"] = bc_idx
                pd_df_static["bc_pos"] = [bc_pos.cpu().numpy()]
                pd_df_static["bc_vel"] = [bc_vel.cpu().numpy()]

            else:
                raise ValueError

        # Return
        return list_pd_df_static

    @staticmethod
    def add_mp_to_files(config_name: str):
        """
        Write MP weights to files
        Args:
            config_name: name of configuration storing dataset and MP setup

        Returns:
            None
        """
        # Get config
        config_path = util.get_config_path(config_name, config_type="mp")
        config = util.parse_config(config_path)
        mp_config = config["mp"]["args"]
        mp_type = config["mp"]["type"]

        # Read dataset
        list_pd_df, list_pd_df_static = util.read_dataset(config["dataset"])

        # Get time duration of all trajectories
        global_t_max = -1e30
        global_t_min = 1e30
        time_key = config["time_key"]
        for pd_df in list_pd_df:
            t_max = pd_df[time_key].values.max()
            t_min = pd_df[time_key].values.min()
            global_t_max = t_max if t_max > global_t_max else global_t_max
            global_t_min = t_min if t_min < global_t_min else global_t_min
        global_time_duration = global_t_max - global_t_min

        # Get basis function and mp
        if mp_type == "promp":
            phase_gn = \
                LinearPhaseGenerator(phase_velocity=1 / global_time_duration)
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                duration=global_time_duration,
                num_basis=mp_config["num_basis"],
                basis_bandwidth_factor=mp_config["basis_bandwidth_factor"],
                num_basis_outside=mp_config["num_basis_outside"])
            mp = ProMP(basis_gn=basis_gn,
                       num_dof=config["num_dof"])
        elif mp_type == "idmp":
            phase_gn = \
                ExpDecayPhaseGenerator(tau=mp_config["tau"],
                                       alpha_phase=mp_config["alpha_phase"])
            basis_gn = DMPBasisGenerator(phase_generator=phase_gn,
                                         num_basis=mp_config["num_basis"],
                                         duration=global_time_duration,
                                         basis_bandwidth_factor=
                                         mp_config["basis_bandwidth_factor"],
                                         num_basis_outside=
                                         mp_config["num_basis_outside"])
            num_pc_times = mp_config["num_pc_times"]
            pc_times = torch.linspace(global_t_min, global_t_max,
                                      num_pc_times)
            mp = IDMP(basis_gn=basis_gn,
                      num_dof=config["num_dof"],
                      **mp_config)

        else:
            raise ValueError("Unknown MP type")

        # Get MP weights learner
        weights_learner = mp.weights_learner

        # Add MP weights to dataset pandas dataframes
        list_pd_df_static_mp = \
            MPProcess.add_mp_to_dataframes(list_pd_df,
                                           list_pd_df_static,
                                           weights_learner,
                                           **config)

        # Save path
        if config["save_as_dataset"] is not None:
            save_path = util.get_dataset_dir(config["save_as_dataset"])
        else:
            save_path = util.get_dataset_dir(config["dataset"])

        # Remove existing directory
        util.remove_file_dir(save_path)

        # Generate directory in path
        os.makedirs(save_path)

        # Save to files
        for (index, (traj, traj_mp)) in \
                enumerate(zip(list_pd_df, list_pd_df_static_mp)):
            traj.to_csv(path_or_buf=save_path + "/" + str(index) + ".csv",
                        index=False,
                        quoting=csv.QUOTE_ALL)
            traj_mp.to_csv(path_or_buf=save_path + "/" + 'static_'
                                       + str(index) + ".csv",
                           index=False,
                           quoting=csv.QUOTE_ALL)


class TrajectoriesReconstructor:
    def __init__(self,
                 mp_config_name: str):
        """
        Args:
            mp_config_name: config of mp
        """
        config = util.parse_config(util.get_config_path(mp_config_name,
                                                        config_type="mp"),
                                   config_type="mp")
        self.data_keys = config["data_keys"]
        self.mp_config = config["mp"]["args"]
        self.mp_key = config["mp_key"]
        self.num_dof = config["num_dof"]
        self.mp_type = config["mp"]["type"]
        self.num_basis = self.mp_config["num_basis"]
        self.num_basis_outside = self.mp_config["num_basis_outside"]
        self.basis_bandwidth_factor = self.mp_config["basis_bandwidth_factor"]
        self.duration = self.mp_config["duration"]

        if self.mp_type == "dmp" or self.mp_type == "idmp":
            self.tau = self.mp_config["tau"]
            self.alpha_phase = self.mp_config["alpha_phase"]

        self.mp = self.initialize_mp(duration=self.duration)

    def get_config(self):
        return {"mp_key": self.mp_key,
                "data_keys": self.data_keys,
                "num_dof": self.num_dof,
                "mp_type": self.mp_type,
                "num_basis": self.num_basis,
                "num_basis_outside": self.num_basis_outside,
                "basis_bandwidth_factor": self.basis_bandwidth_factor,
                "duration": self.duration}

    def initialize_mp(self,
                      duration):
        """
        Initialize a mp
        Args:
            duration: duration

        Returns:
            A MP object
        """

        # Get basis function generator and mp
        if self.mp_type == "promp":
            # ProMP
            # Get phase generator
            phase_gn = LinearPhaseGenerator(phase_velocity=1 / duration)
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=self.num_basis,
                duration=duration,
                basis_bandwidth_factor=self.basis_bandwidth_factor,
                num_basis_outside=self.num_basis_outside)
            mp = ProMP(basis_gn=basis_gn, num_dof=self.num_dof)

        if self.mp_type == "dmp":
            # DMP
            # Get Phase Generator
            phase_gn = ExpDecayPhaseGenerator(tau=self.tau,
                                              alpha_phase=self.alpha_phase)
            basis_gn = DMPBasisGenerator(
                phase_generator=phase_gn,
                num_basis=self.num_basis,
                duration=duration,
                basis_bandwidth_factor=self.basis_bandwidth_factor,
                num_basis_outside=self.num_basis_outside)
            mp = DMP(basis_gn=basis_gn, num_dof=self.num_dof, **self.mp_config)

        elif self.mp_type == "idmp":
            # IDMP
            # Get Phase Generator
            phase_gn = ExpDecayPhaseGenerator(tau=self.tau,
                                              alpha_phase=self.alpha_phase)
            basis_gn = DMPBasisGenerator(
                phase_generator=phase_gn,
                num_basis=self.num_basis,
                duration=duration,
                basis_bandwidth_factor=self.basis_bandwidth_factor,
                num_basis_outside=self.num_basis_outside)
            mp = IDMP(basis_gn=basis_gn, num_dof=self.num_dof, **self.mp_config)

        else:
            raise ValueError("Invalid MP type")

        return mp

    def reconstruct(self,
                    duration,
                    w_mean,
                    w_diag=None,
                    w_off_diag=None,
                    w_L=None,
                    **kwargs):
        """
        reconstruct trajectories from mp

        Args:
            duration: duration of MP, todo remove this arg
            w_mean: predicted mean of weights
            w_diag: predicted diag of weights, can be None
            w_off_diag: predicted off-diag of weights, can be None,
            w_L: alternative weight Cholesky, can be None
            kwargs: key arguments
                times: reconstruct trajectories at these time points
                time_indices: reconstruct trajectories at these time indices
                condition: condition dict for current time and value
                std: True if apply std only else full cov
        Returns:
            predicted weights mean and cov
        """

        # Form up Cholesky Matrix
        if w_L is None:
            if w_diag is not None:
                w_L = util.build_lower_matrix(w_diag, w_off_diag)
            else:
                w_L = None
        else:
            pass

        # Predict std or cov? True if std, False if full Cov
        std = kwargs.get("std", True)

        # Get additional dimension
        add_dim = w_mean.shape[:-1]

        # Check if get velocity
        get_velocity = kwargs.get("get_velocity", False)

        # Set MP
        if self.mp_type == "promp":
            assert get_velocity is False
            times = kwargs["times"]
            self.mp.set_promp(add_dim=add_dim, mu=w_mean, L=w_L, obs_sigma=None)

        elif self.mp_type == "dmp":
            bc_dict = dict()
            bc_dict["bc_time"] = kwargs.get("bc_time")
            bc_dict["bc_pos"] = kwargs.get("bc_pos")
            bc_dict["bc_vel"] = kwargs.get("bc_vel")
            times = kwargs["times"]
            self.mp.set_dmp(add_dim=add_dim, w_g=w_mean, bc_dict=bc_dict)

        elif self.mp_type == "idmp":
            bc_dict = dict()
            bc_dict["bc_index"] = kwargs.get("bc_index")
            bc_dict["bc_pos"] = kwargs.get("bc_pos")
            bc_dict["bc_vel"] = kwargs.get("bc_vel")
            times = kwargs["time_indices"]
            self.mp.set_dmp(add_dim=add_dim, mu=w_mean, L=w_L, bc_dict=bc_dict)

        else:
            raise NotImplementedError

        if not get_velocity:
            traj_mean, traj_cov = self.mp.get_traj_mean_and_cov(times)[:2]
            if traj_cov is None or not std:
                return traj_mean, traj_cov
            else:
                traj_std = torch.sqrt(torch.einsum('...ii->...i', traj_cov))
                return traj_mean, traj_std

        else:
            traj_mean, traj_cov, vel_mean, vel_cov = \
                self.mp.get_traj_mean_and_cov(times)
            if traj_cov is None or not std:
                return traj_mean, traj_cov, vel_mean, vel_cov
            else:
                traj_std = torch.sqrt(torch.einsum('...ii->...i', traj_cov))
                vel_std = torch.sqrt(torch.einsum('...ii->...i', vel_cov))
                return traj_mean, traj_std, vel_mean, vel_std
