
import copy
import csv
import os
import time
import matplotlib.pyplot as plt

from nmp.ellipses_noise import EllipseNoiseTransform
from nmp.net import MPNet
from nmp.loss import *

# plt.xlim([12.5,27.5])
# plt.ylim([11.5,36.5])
# figsize=(3,5)

def mnist_add_noise():
    dataset_name = "s_mnist_15/40x40-smnist.npz"
    file = util.get_dataset_dir(dataset_name)
    images, outputs, original_traj = load_mnist_data_from_npz(file=file)
    # test_img = images[torch.randint(20000, size=[])]
    images = torch.from_numpy(images)
    num_images = len(images)
    transformed_imgs = torch.zeros([20000, 4, 1, 40, 40])
    transform = EllipseNoiseTransform()
    for idx, img in enumerate(images):
        img = img[None,]
        transformed_imgs[idx] = transform(img)
    transformed_imgs = transformed_imgs.numpy()

    save_dir = util.get_dataset_dir("s_mnist_15_noise")
    util.remove_file_dir(save_dir)
    os.makedirs(save_dir)
    np.savez(save_dir + '/s_mnist_15_noise.npz',
             images=transformed_imgs,
             traj_x=original_traj[:, :, 0],
             traj_y=original_traj[:, :, 1],
             init_x_y=outputs[:, :2],
             dmp_w_g=outputs[:, 2:])

    # fig, axes = plt.subplots(1, cfg.args.n_noisy + 1)
    # axes[0].imshow(test_img[0].cpu().numpy(), cmap='gray')
    # for i in range(cfg.args.n_noisy):
    #     axes[i + 1].imshow(transformed_img[i, 0], cmap='gray')

    # plt.imshow(transformed_img[-1, 0], cmap='gray')
    # plt.show()


def cnmp_mnist_post_processing(mp_net, exp_title, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    # Predict using test dataset
    result_dict = mp_net.use_test_dataset_to_predict(num_mc_smp=None)

    # Loss
    rec_ll = -result_dict["loss"]
    util.print_line_title("Reconstruct likelihood: {}".format(rec_ll))

    use_test_data_list = torch.linspace(0, 9, 10).long()

    # Get image
    images = \
        result_dict["test_batch"]["images"]["value"][use_test_data_list, -1, 0]
    images = images.cpu().numpy()
    img_size = images.shape[-1]

    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1
    # Get ground truth trajs
    traj_x_y = \
        result_dict["test_batch"]["trajs"]["value"][use_test_data_list,
        :].cpu().numpy()

    agg = -1

    mean_x = result_dict["mean"][use_test_data_list, agg, :, 0].cpu().numpy()
    mean_y = result_dict["mean"][use_test_data_list, agg, :, 1].cpu().numpy()

    h = 2
    w = 5

    fig, axes = plt.subplots(h, w, figsize=(10, 5),
                             tight_layout=True, dpi=200)
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            axes[i, j].imshow(images[idx], extent=[0, img_size, img_size, 0])
            axes[i, j].plot(mean_x[idx], mean_y[idx], 'g--')
            axes[i, j].plot(traj_x_y[idx, :, 0], traj_x_y[idx, :, 1], 'k')

    # mp_net.log_figure(fig, exp_title + ", test_dataset_mean")

    # Plot samples
    h = 5
    w = 10
    num_sum = h
    fig, axes = plt.subplots(h, w, figsize=(w * 2, h * 2),
                             tight_layout=True, dpi=200)

    # do sampling
    mean = result_dict["mean"][use_test_data_list, agg, :]
    L = result_dict["L"][use_test_data_list, agg, :]
    mvn = MultivariateNormal(loc=mean, scale_tril=L, validate_args=False)
    samples = mvn.sample([num_sum]).cpu().numpy()
    samples_x = samples[:, :, :, 0]
    samples_y = samples[:, :, :, 1]

    for i in range(h):
        for j in range(w):
            # axes[i,j].imshow(images[idx], extent=[0, img_size, img_size, 0])
            axes[i, j].plot(samples_x[i, j], samples_y[i, j], 'b')
            axes[i, j].set_xlim([0, img_size])
            axes[i, j].set_ylim([0, img_size])
            axes[i, j].invert_yaxis()
            axes[i, j].axis('off')

    mp_net.log_figure(fig, exp_title + ", test_dataset_samples")



def mnist_post_processing(mp_net, exp_title, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    # Predict using test dataset
    result_dict = mp_net.use_test_dataset_to_predict(num_mc_smp=None)

    # Loss
    rec_ll = -result_dict["loss"]
    util.print_line_title("Reconstruct likelihood: {}".format(rec_ll))

    use_test_data_list = torch.linspace(0, 9, 10).long()

    # Get image
    # images = \
    #     result_dict["test_batch"]["images"]["value"][use_test_data_list, 0, 0]
    images = \
        result_dict["test_batch"]["images"]["value"][use_test_data_list, -1, 0]

    # Get ground truth trajs
    traj_x_y = \
        result_dict["test_batch"]["trajs"]["value"][use_test_data_list, :]

    # Debug only
    # for i in range(len(images)):
    #     plt.imshow(images[i].cpu().numpy(), extent=[0, 28, 28, 0])
    #     plt.plot(traj_x_y[i, :, 0].cpu().numpy(), traj_x_y[i, :, 1].cpu().numpy())
    #     plt.show()

    # Get DMP ground truth trajs
    dmp_gd = \
        result_dict["test_batch"]["outputs"]["value"][use_test_data_list, 0]
    start = dmp_gd[:, :2]

    w_g = dmp_gd[:, 2:]
    tr_gd_input = {"bc_index": torch.zeros(use_test_data_list.shape).long(),
                   "bc_pos": start,
                   "bc_vel": torch.zeros(start.shape),
                   "time_indices": None}
    kwargs["normalizer"] = mp_net.normalizer
    dmp_traj_gt = idmp_to_traj(tr_gd_input, w_g, **kwargs)

    # Set agg state
    agg = -1
    # agg = 1

    # Get DMP predict trajs
    pred_mean = result_dict["mean"][use_test_data_list, agg, 0]
    pred_start = pred_mean[:, :2]
    pred_w_g = pred_mean[:, 2:]
    pred_L = result_dict["L"] if kwargs.get("sampling", False) is True else None
    if pred_L is not None:
        pred_L = pred_L[use_test_data_list, agg, 0, 2:, 2:]

    tr_pred_input = {"bc_index": torch.zeros(use_test_data_list.shape).long(),
                     "bc_pos": pred_start,
                     "bc_vel": torch.zeros(start.shape),
                     "time_indices": None}
    pred_dmp_traj_mean = idmp_to_traj(tr_pred_input, pred_w_g, **kwargs)

    fig = digits_plot(images, traj_x_y, dmp_traj_gt, pred_dmp_traj_mean)
    mp_net.log_figure(fig, exp_title + ", test_dataset_mean")

    if kwargs.get("sampling", False) is True:
        mvn = MultivariateNormal(loc=pred_w_g, scale_tril=pred_L,
                                 validate_args=False)
        #############
        num_smp = 10
        #############

        samples = mvn.sample([num_smp])
        tr_pred_input = \
            {"bc_index": torch.zeros(
                [num_smp, *use_test_data_list.shape]).long(),
             "bc_pos": util.add_expand_dim(pred_start, [0], [num_smp]),
             "bc_vel": torch.zeros(num_smp, *start.shape),
             "time_indices": None}

        pred_dmp_traj = idmp_to_traj(tr_pred_input, samples, **kwargs)

        plot_together = kwargs.get("plot_together", True)
        fig_samples = sample_digits_plot(images, traj_x_y, pred_dmp_traj,
                                         plot_together)
        mp_net.log_figure(fig_samples, exp_title + ", test_dataset_samples")


def mnist_multi_post_processing(mp_net, exp_title, **kwargs):
    # Draw different aggregate state result in one image, noise case

    kwargs = copy.deepcopy(kwargs)
    kwargs["normalizer"] = mp_net.normalizer
    # Predict using test dataset
    result_dict = \
        mp_net.use_test_dataset_to_predict(num_mc_smp=None)

    use_test_data_list = torch.linspace(0, 9, 10).long()
    # use_test_data_list = torch.linspace(0, 99,100).long()

    # Get image
    images = \
        result_dict["test_batch"]["images"]["value"][use_test_data_list, :, 0]

    # Get DMP predict trajs
    pred_mean = result_dict["mean"][use_test_data_list, :, 0]
    pred_start = pred_mean[:, :, :2]
    pred_w_g = pred_mean[:, :, 2:]

    pred_L = result_dict["L"] if kwargs.get("sampling", False) is True else None
    if pred_L is not None:
        pred_L = pred_L[use_test_data_list, :, 0, 2:, 2:]
    num_agg = pred_mean.shape[1]
    tr_pred_input = {"bc_index": torch.zeros([*use_test_data_list.shape,
                                              num_agg]).long(),
                     "bc_pos": pred_start,
                     "bc_vel": torch.zeros(pred_start.shape),
                     "time_indices": None}
    kwargs["std"] = True
    pred_dmp_traj_mean, pred_dmp_traj_std = \
        idmp_to_traj(tr_pred_input, pred_w_g, **kwargs)

    images = images.cpu().numpy()
    pred_dmp_traj_mean = pred_dmp_traj_mean.cpu().numpy()

    w = 8
    h = len(use_test_data_list)
    fig, axes = plt.subplots(h, w, figsize=(16, 20), tight_layout=True)

    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1

    for i in range(h):
        for j in range(w):
            # axes[i,j].axis('off')
            if j > 3:
                axes[i, j].set_ylim([0, img_size])
                axes[i, j].set_xlim([0, img_size])

    for i in range(h):
        axes[i, 0].imshow(images[i, -1], cmap='gray', extent=[0, img_size,
                                                              img_size, 0])

        for j in range(3):
            axes[i, j + 1].imshow(images[i, j],
                                  cmap='gray', extent=[0, img_size,
                                                       img_size, 0])

            axes[i, j + 5].plot(pred_dmp_traj_mean[i, j + 1, :, 0],
                                pred_dmp_traj_mean[i, j + 1, :, 1])
            axes[i, j + 5].invert_yaxis()

        axes[i, 4].plot(pred_dmp_traj_mean[i, 0, :, 0],
                        pred_dmp_traj_mean[i, 0, :, 1])
        axes[i, 4].invert_yaxis()

    mp_net.log_figure(fig, exp_title + ", test_dataset_aggregation")


def mnist_multi_sampling(mp_net, exp_title, **kwargs):
    # Sample aggregation images at each aggregation state
    kwargs = copy.deepcopy(kwargs)
    kwargs["normalizer"] = mp_net.normalizer

    # Predict using test dataset
    result_dict = \
        mp_net.use_test_dataset_to_predict(num_mc_smp=None)

    # use_test_data_list = torch.linspace(0, 9, 10).long()
    # use_test_data_list = torch.linspace(0, 49, 50).long()
    # use_test_data_list = torch.Tensor([18, 46]).long()
    use_test_data_list = torch.Tensor([46]).long()

    # Get image
    images = \
        result_dict["test_batch"]["images"]["value"][use_test_data_list, :, 0]
    images = images.cpu().numpy()

    # Get DMP predict trajs
    pred_mean = result_dict["mean"][use_test_data_list, :, 0]
    pred_start = pred_mean[:, :, :2]
    pred_w_g = pred_mean[:, :, 2:]

    pred_L = result_dict["L"] if kwargs.get("sampling", False) is True else None
    if pred_L is not None:
        pred_L = pred_L[use_test_data_list, :, 0, 2:, 2:]

    # Some plotting parameters
    w = 8
    h = len(use_test_data_list)

    # Sample DMPs
    mvn = MultivariateNormal(loc=pred_w_g, scale_tril=pred_L,
                             validate_args=False)
    num_smp = w - 4
    samples = mvn.sample([num_smp])

    num_agg = pred_mean.shape[1]
    tr_pred_input = {
        "bc_index": torch.zeros([num_smp, *use_test_data_list.shape,
                                 num_agg]).long(),
        "bc_pos": util.add_expand_dim(pred_start, [0], [num_smp]),
        "bc_vel": torch.zeros([num_smp, *pred_start.shape]),
        "time_indices": None}

    kwargs["std"] = True

    # From DMPs to trajs
    pred_dmp_traj_mean = idmp_to_traj(tr_pred_input, samples, **kwargs)
    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1

    for agg in range(num_agg):
        pred_dmp_traj = pred_dmp_traj_mean[..., agg, :, :]  # Last agg state

        samples_x = pred_dmp_traj[..., 0].cpu().numpy()
        samples_y = pred_dmp_traj[..., 1].cpu().numpy()

        fig, axes = plt.subplots(h, w, figsize=(2 * w, 2 * h),
                                 tight_layout=True)
        axes = axes.reshape(h, w)
        for i in range(h):
            for j in range(w):
                axes[i, j].axis('off')
                if j > 3:
                    axes[i, j].set_ylim([0, img_size])
                    axes[i, j].set_xlim([0, img_size])

        for i in range(h):
            axes[i, 0].imshow(images[i, -1], #cmap='gray',
                              extent=[0, img_size, img_size, 0])
            for j in range(agg_):
                axes[i, j + 1].imshow(images[i, j],
                                      #cmap='gray',
                                      extent=[0, img_size, img_size, 0])
            for j in range(4):
                axes[i, j + 4].plot(samples_x[j, i],
                                    samples_y[j, i])
                axes[i, j + 4].invert_yaxis()

        # mp_net.log_figure(fig, exp_title)
        # mp_net.log_figure(fig, exp_title + "_agg_{}".format(agg))


def mnist_replaning(mp_net, exp_title, **kwargs):
    # replaning based on different agg_state
    kwargs = copy.deepcopy(kwargs)
    normalizer = mp_net.normalizer
    duration = (normalizer["time"]["max"] - normalizer["time"]["min"]).item()

    config_name = kwargs["idmp_config"]
    tr = TrajectoriesReconstructor(config_name)

    # Predict using test dataset
    result_dict = \
        mp_net.use_test_dataset_to_predict(num_mc_smp=None)

    use_test_data_list = torch.linspace(0, 9, 10).long()
    # use_test_data_list = torch.linspace(0, 3, 4).long()
    # use_test_data_list = torch.linspace(500, 503, 4).long()

    # Get image
    images = \
        result_dict["test_batch"]["images"]["value"][use_test_data_list, :, 0]
    images = images.cpu().numpy()

    ground_truth = result_dict["test_batch"]["trajs"]["value"][
        use_test_data_list].cpu().numpy()
    gt_x = ground_truth[..., :, 0]
    gt_y = ground_truth[..., :, 1]

    reconstructor_input = result_dict["assigned_dict"]["reconstructor_input"]
    num_replan_steps = len(reconstructor_input)

    use_last_mean = False
    end_pos_mean, end_vel_mean = None, None

    pred_mean = result_dict["mean"][use_test_data_list]
    pred_L = result_dict["L"][use_test_data_list]
    num_dof = tr.get_config()["num_dof"]

    x_list = list()
    y_list = list()

    for step in range(num_replan_steps):
        # Get reconstruction prediction for current step
        curr_pred_mean = pred_mean[:, step + 1, ...]
        curr_pred_L = pred_L[:, step + 1, ...]

        # Get reconstruction input for current step
        curr_rec_input = reconstructor_input[step]
        curr_bc_index = curr_rec_input["bc_index"]
        curr_end_index = curr_rec_input["end_index"]

        # Dynamically add boundary pos and vel
        if step == 0:
            curr_bc_pos = curr_pred_mean[..., 0, :num_dof]
            curr_bc_vel = torch.zeros_like(curr_bc_pos)

        else:
            curr_bc_pos = end_pos_mean
            curr_bc_vel = end_vel_mean

        curr_pred_mean = curr_pred_mean[..., num_dof:]
        curr_pred_L = curr_pred_L[..., num_dof:, num_dof:]
        assert curr_pred_mean.ndim == 3

        #
        # # Sample
        # L = util.build_lower_matrix(torch.ones_like(curr_pred_mean), None)
        # mvn = MultivariateNormal(loc=torch.zeros_like(curr_pred_mean),
        #                          scale_tril=L, validate_args=False)
        # epsilon = mvn.sample()
        # sample = torch.einsum('...ik,...k->...i',
        #                       curr_pred_L, epsilon) + curr_pred_mean
        # sample


        # Get the predicted data dimension prepared, remove the dimension of time
        # weights is time-invariant
        curr_pred_mean = curr_pred_mean.squeeze(-2)
        curr_pred_L = curr_pred_L.squeeze(-3)
        # Get the last robot state in current replan-step
        end_rec_input = dict()
        curr_bc_index = \
            util.add_expand_dim(data=curr_bc_index, add_dim_indices=[0],
                                add_dim_sizes=[curr_pred_mean.shape[0]])
        curr_end_index = \
            util.add_expand_dim(data=curr_end_index, add_dim_indices=[0, -1],
                                add_dim_sizes=[curr_pred_mean.shape[0], 1])
        end_rec_input["bc_index"] = curr_bc_index
        end_rec_input["bc_pos"] = curr_bc_pos
        end_rec_input["bc_vel"] = curr_bc_vel
        end_rec_input["get_velocity"] = True
        end_rec_input["time_indices"] = curr_end_index

        # Use mean of the prediction to get end point
        end_pos_mean, _, end_vel_mean, __ = \
            tr.reconstruct(duration=duration,
                           w_mean=curr_pred_mean,
                           w_L=curr_pred_L,
                           **end_rec_input)

        # Predict entire trajectory
        curr_rec_input["bc_index"] = curr_bc_index
        curr_rec_input["bc_pos"] = curr_bc_pos
        curr_rec_input["bc_vel"] = curr_bc_vel
        curr_rec_input["std"] = False
        curr_rec_input["time_indices"] = None

        pos_mean, _ = tr.reconstruct(duration=duration,
                                     w_mean=curr_pred_mean,
                                     w_L=curr_pred_L,
                                     **curr_rec_input)
        x = pos_mean[:, :301].cpu().numpy()
        y = pos_mean[:, 301:].cpu().numpy()
        x_list.append(x)
        y_list.append(y)

    w = 5
    h = len(use_test_data_list)
    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1

    fig, axes = plt.subplots(h, w, figsize=(2 * w, 2 * h),
                             tight_layout=True)
    for i in range(h):
        for j in range(w):
            axes[i, j].axis('off')
            if j > 3:
                axes[i, j].set_ylim([0, img_size])
                axes[i, j].set_xlim([0, img_size])

    for i in range(h):
        axes[i, 0].imshow(images[i, -1],
                          # cmap='gray',
                          extent=[0, img_size, img_size, 0])
        # axes[i, 0].text(x=3, y=37, s="{}".format(i), fontsize="large",
        #                 fontfamily="sans-serif", color='w')
        for j in range(3):
            axes[i, j + 1].imshow(images[i, j],
                                  # cmap='gray',
                                  extent=[0, img_size, img_size, 0])
        # Ground-truth
        axes[i, 4].imshow(images[i, -1],
                              # cmap='gray',
                              extent=[0, img_size, img_size, 0], alpha=0.2)

        # axes[i, 4].plot(gt_x[i], gt_y[i], 'k')

        axes[i, 4].plot(x_list[0][i, :75+1], y_list[0][i, :75+1], 'b',
                        label='25%')
        axes[i, 4].plot(x_list[0][i, 75:], y_list[0][i, 75:], 'b--', alpha=0.3,
                        )

        axes[i, 4].plot(x_list[1][i, 75:150+1], y_list[1][i, 75:150+1], 'y',
                        label='25-50%')
        axes[i, 4].plot(x_list[1][i, 150:], y_list[1][i, 150:], 'y--',
                        alpha=0.3,)

        axes[i, 4].plot(x_list[2][i, 150:], y_list[2][i, 150:], 'r',
                        label='50-100%')
        axes[i, 4].legend()
        axes[i, 4].invert_yaxis()

    print("")
    # mp_net.log_figure(fig, exp_title)
    # mp_net.log_figure(fig, exp_title + "_agg_{}".format(agg))


def idmp_to_traj(reconstructor_input, w_g, **kwargs):
    # Get config_name
    config_name = kwargs["idmp_config"]
    tr = TrajectoriesReconstructor(config_name)

    reconstructor_input = copy.deepcopy(reconstructor_input)

    # Get time duration
    normalizer = kwargs["normalizer"]
    duration = (normalizer["time"]["max"] - normalizer["time"]["min"]).item()

    # Get velocity
    get_velocity = kwargs.get("get_velocity", False)
    reconstructor_input["get_velocity"] = get_velocity

    # std or full cov?
    reconstructor_input["std"] = True

    # Reconstruct
    mean, _ = tr.reconstruct(duration, w_mean=w_g, w_L=None,
                             **reconstructor_input)

    # Reshape mean
    num_dof = tr.get_config()["num_dof"]
    mean = mean.reshape([*mean.shape[:-1], num_dof, mean.shape[-1] // num_dof])
    mean = torch.einsum('...ij->...ji', mean)

    return mean


def digits_plot(images, traj_x_y, dmp_traj_gt, pred_dmp_traj):
    num_images = len(images)

    traj_x_y = traj_x_y.cpu().numpy()
    x = traj_x_y[..., 0]
    y = traj_x_y[..., 1]

    dmp_traj_gt = dmp_traj_gt.cpu().numpy()
    dmp_x = dmp_traj_gt[..., 0]
    dmp_y = dmp_traj_gt[..., 1]

    pred_dmp_traj = pred_dmp_traj.cpu().numpy()
    pred_dmp_x = pred_dmp_traj[..., 0]
    pred_dmp_y = pred_dmp_traj[..., 1]

    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1

    fig = plt.figure(figsize=(20, 15), dpi=200, tight_layout=True)  # todo size
    for i in range(num_images):
        plt.subplot(2, (num_images + 1) // 2, i + 1)

        plt.imshow(images[i].cpu().numpy(),
                   extent=[0, img_size, img_size, 0])
        plt.plot(x[i], y[i], 'k', label="ground_truth")
        plt.plot(dmp_x[i], dmp_y[i], 'r', label="ground_truth_dmp")
        plt.plot(pred_dmp_x[i], pred_dmp_y[i], 'g--', label="pred_dmp")

    plt.show()
    return fig


def sample_digits_plot(images, traj_x_y, pred_dmp_traj, together=True):
    num_images = len(images)

    traj_x_y = traj_x_y.cpu().numpy()
    x = traj_x_y[..., 0]
    y = traj_x_y[..., 1]

    # switch sample axis with num_traj axis
    pred_dmp_traj = torch.einsum('ij...->ji...', pred_dmp_traj)

    img_size = images.shape[-1]
    if img_size == 40:
        img_size += 1

    # reshape and switch
    samples_x = pred_dmp_traj[:, :, :, 0].cpu().numpy()
    samples_y = pred_dmp_traj[:, :, :, 1].cpu().numpy()

    if together:
        fig = plt.figure(figsize=(15, 20), dpi=200, tight_layout=True)
        for i in range(num_images):
            plt.subplot(2, (num_images + 1) // 2, i + 1)

            plt.imshow(images[i].cpu().numpy(),
                       extent=[0, img_size, img_size, 0], alpha=0)

            # plt.plot(x[i], y[i], 'k', label="ground_truth", zorder=200)
            plt.plot(samples_x[i].T, samples_y[i].T, 'b', label="pred_dmp_smp",
                     alpha=0.3)

        plt.show()
    else:
        num_smp = pred_dmp_traj.shape[1]
        fig, axes = plt.subplots(num_smp, num_images,
                                 figsize=(2 * num_images, 2 * num_smp),
                                 tight_layout=True)

        for i in range(num_images):
            if i == 8:
                tmp_path = util.clean_and_get_tmp_dir()
                fig, axes = plt.subplots(2, 3,
                                     figsize=(6, 4),
                                     tight_layout=True)
            for j in range(num_smp):
                axes[j, i].plot(samples_x[i, j], - samples_y[i, j], 'b')
                axes[j, i].axis('off')

    return fig


def idmp_digit_exp(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Mean")
    kwargs["idmp_config"] = "IDMP_digits_25_config"
    config_path = util.get_config_path("NMP_IDMP_MNIST_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_MEAN",
                                       mp_net.epoch)
    mnist_post_processing(mp_net, exp_title, **kwargs)

    # kwargs["y_lim"] = [-1, 5]
    # kwargs["x_lim"] = [0, 3.0]


def idmp_digit_exp_noise(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Mean")
    kwargs["idmp_config"] = "IDMP_digits_config"
    config_path = util.get_config_path("NMP_IDMP_MNIST_NOISE_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["noise"] = True
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_MEAN_NOISE",
                                       mp_net.epoch)
    mnist_multi_post_processing(mp_net, exp_title, **kwargs)

    # kwargs["y_lim"] = [-1, 5]
    # kwargs["x_lim"] = [0, 3.0]


def idmp_digit_std_exp(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Mean + STD")

    config_path = util.get_config_path("NMP_IDMP_MNIST_STD_25_config")
    kwargs["idmp_config"] = "IDMP_digits_25_config"

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    kwargs["plot_together"] = True

    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_STD",
                                       mp_net.epoch)
    mnist_post_processing(mp_net, exp_title, **kwargs)


def idmp_digit_cov_exp(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Mean + Cov")

    # kwargs["idmp_config"] = "IDMP_digits_config"
    # config_path = util.get_config_path("NMP_IDMP_MNIST_COV_config")

    # config_path = util.get_config_path("NMP_IDMP_MNIST_COV_20_config")
    # kwargs["idmp_config"] = "IDMP_digits_20_config"

    config_path = util.get_config_path("NMP_IDMP_MNIST_COV_25_config")
    kwargs["idmp_config"] = "IDMP_digits_25_config"

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    kwargs["plot_together"] = True # todo

    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_COV",
                                       mp_net.epoch)
    mnist_post_processing(mp_net, exp_title, **kwargs)


def idmp_digits_label_cov(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Label + Image")
    kwargs["idmp_config"] = "IDMP_digits_config"
    config_path = util.get_config_path("NMP_IDMP_MNIST_LABEL_COV_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_LABEL_COV",
                                       mp_net.epoch)
    # mnist_post_processing(mp_net, exp_title, **kwargs)


def idmp_digit_std_exp_noise(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Mean + STD NOISE")
    kwargs["idmp_config"] = "IDMP_digits_config"
    config_path = util.get_config_path("NMP_IDMP_MNIST_STD_NOISE_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    kwargs["noise"] = True

    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_STD_NOISE",
                                       mp_net.epoch)
    mnist_multi_sampling(mp_net, exp_title, **kwargs)


def idmp_digit_cov_exp_noise(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Mean + COV NOISE")
    kwargs["idmp_config"] = "IDMP_digits_config"
    config_path = util.get_config_path("NMP_IDMP_MNIST_COV_NOISE_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    kwargs["noise"] = True

    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_COV_NOISE",
                                       mp_net.epoch)
    mnist_multi_sampling(mp_net, exp_title, **kwargs)


def idmp_digits_std_noise_replan_exp(**kwargs):
    util.print_wrap_title("NMP IDMP of MNIST Digits Mean + STD NOISE REPLAN")
    kwargs["idmp_config"] = "IDMP_digits_config"
    config_path = util.get_config_path("NMP_IDMP_MNIST_STD_NOISE_REPLAN_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    kwargs["noise"] = True

    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_MNIST_STD_NOISE_REPLAN",
                                       mp_net.epoch)
    mnist_replaning(mp_net, exp_title, **kwargs)


def idmp_amnist_digit_cov_exp(**kwargs):
    util.print_wrap_title("NMP IDMP of A-MNIST Digits Mean + Cov")

    config_path = util.get_config_path("NMP_IDMP_AMNIST_2_COV_config")
    # config_path = util.get_config_path("NMP_IDMP_AMNIST_3_COV_config")
    # config_path = util.get_config_path("NMP_IDMP_AMNIST_5_COV_config")
    # config_path = util.get_config_path("NMP_IDMP_AMNIST_6_COV_config")
    # config_path = util.get_config_path("NMP_IDMP_AMNIST_8_COV_config")
    # config_path = util.get_config_path("NMP_IDMP_AMNIST_9_COV_config")

    kwargs["idmp_config"] = "IDMP_amnist_digits_config"

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    kwargs["plot_together"] = False
    if kwargs["max_epoch"] != kwargs["init_epoch"]:
        max_epoch = kwargs["max_epoch"] * 10
    else:
        max_epoch = kwargs["max_epoch"]
    mp_net = MPNet(config,
                   max_epoch,
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_AMNIST_COV",
                                       mp_net.epoch)
    mnist_post_processing(mp_net, exp_title, **kwargs)


def cnmp_digits_exp(**kwargs):
    util.print_wrap_title("CNMP of MNIST Digits Mean + STD")

    config_path = util.get_config_path("CNMP_MNIST_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    kwargs["sampling"] = True
    kwargs["plot_together"] = True

    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("CNMP_MNIST_STD",
                                       mp_net.epoch)
    cnmp_mnist_post_processing(mp_net, exp_title, **kwargs)


def test(exp: str, restart=True):
    exp_api = dict()

    exp_api["idmp_digits"] = {
        "func": idmp_digit_exp,
        "api":  None,
        "best_epoch": 2000}

    exp_api["idmp_digits_noise"] = {
        "func": idmp_digit_exp_noise,
        "api":  None,
        "best_epoch": 2000}

    exp_api["idmp_digits_std"] = {
        "func": idmp_digit_std_exp,
        "api": None,
        "best_epoch": 5000}

    exp_api["idmp_digits_cov"] = {
        "func": idmp_digit_cov_exp,
        "api": None,
        "best_epoch": 4500}

    exp_api["idmp_digits_label_cov"] = {
        "func": idmp_digits_label_cov,
        "api":  None,
        "best_epoch": 1500}

    exp_api["idmp_digits_std_noise"] = {
        "func": idmp_digit_std_exp_noise,
        "api":  None,
        "best_epoch": 5000}

    exp_api["idmp_digits_cov_noise"] = {
        "func": idmp_digit_cov_exp_noise,
        "api": None,
        "best_epoch": 5000}

    exp_api["idmp_digits_std_noise_replan"] = {
        "func": idmp_digits_std_noise_replan_exp,
        "api": None,
        "best_epoch": 600}

    exp_api["idmp_amnist_digits_cov"] = {
        "func": idmp_amnist_digit_cov_exp,
        "api":  None,
        "best_epoch": 10000}

    exp_api["cnmp_digits"] = {
        "func": cnmp_digits_exp,
        "api": None,
        "best_epoch": 3000}

    # Specify task
    exp_func = exp_api[exp]["func"]
    exp_kwargs = \
        {"max_epoch": 2000 if restart else exp_api[exp]["best_epoch"],
         "init_epoch": 0 if restart else exp_api[exp]["best_epoch"],
         "num_mc_smp": None,
         "model_api": None if restart else exp_api[exp]["api"],
         "manual_test": False}

    # Run task
    exp_func(**exp_kwargs)


def main(time_delay_hour=0):
    """

    Args:
        time_delay: schedule experiment in x hour later

    Returns:

    """
    util.print_line_title("Run exp in {} hours.".format(time_delay_hour))
    time.sleep(time_delay_hour * 3600)

    # Check if GPU is available
    util.check_torch_device()

    # test(exp="idmp_digits", restart=True)
    # test(exp="idmp_digits", restart=False)

    # test(exp="idmp_digits_noise", restart=True)
    # test(exp="idmp_digits_noise", restart=False)

    # test(exp="idmp_digits_std", restart=True)
    # test(exp="idmp_digits_std", restart=False)

    # test(exp="idmp_digits_cov", restart=True)
    # test(exp="idmp_digits_cov", restart=False)

    # test(exp="idmp_digits_label_cov", restart=True)
    # test(exp="idmp_digits_label_cov", restart=False)

    # test(exp="idmp_digits_std_noise", restart=True)
    test(exp="idmp_digits_std_noise", restart=False)

    # test(exp="idmp_digits_cov_noise", restart=True)
    # test(exp="idmp_digits_cov_noise", restart=False)

    # test(exp="idmp_digits_std_noise_replan", restart=True)
    # test(exp="idmp_digits_std_noise_replan", restart=False)

    # test(exp="idmp_amnist_digits_cov", restart=True)
    # test(exp="idmp_amnist_digits_cov", restart=False)

    # test(exp="cnmp_digits", restart=True)
    # test(exp="cnmp_digits", restart=False)

    pass


if __name__ == "__main__":
    main()
    # main(2)
    # main(4)
    # main(6)
    # main(8)
    # mnist_add_noise()
