"""
--
@brief:     Utilities
"""

# Import Python libs
import csv
import json
import os
import shutil
import time
import random
import numpy as np
import pandas as pd
import torch
import yaml
from natsort import os_sorted
from matplotlib import animation
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tabulate import tabulate
from datetime import datetime
from mnist import MNIST


# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def print_line(char="=", length=60, before=0, after=0):
    """
    Print a line with given letter in given length
    Args:
        char: char for print the line
        length: length of line
        before: number of new lines before print line
        after: number of new lines after print line

    Returns: None
    """

    print("\n" * before, end="")
    print(char * length)
    print("\n" * after, end="")
    # End of function print_line


def print_line_title(title="", middle=True, char="=", length=60, before=1,
                     after=1):
    """
    Print a line with title
    Args:
        title: title to print
        middle: if title should be in the middle, otherwise left
        char: char for print the line
        length: length of line
        before: number of new lines before print line
        after: number of new lines after print line

    Returns: None
    """
    assert len(title) < length, "Title is longer than line length"
    len_before_title = (length - len(title)) // 2 - 1
    len_after_title = length - len(title) - (length - len(title)) // 2 - 1
    print("\n" * before, end="")
    if middle is True:
        print(char * len_before_title, "", end="")
        print(title, end="")
        print("", char * len_after_title)
    else:
        print(title, end="")
        print(" ", char * (length - len(title) - 1))
    print("\n" * after, end="")
    # End of function print_line_title


def print_wrap_title(title="", char="*", length=60, wrap=1, before=1, after=1):
    """
    Print title with wrapped box
    Args:
        title: title to print
        char: char for print the line
        length: length of line
        wrap: number of wrapped layers
        before: number of new lines before print line
        after: number of new lines after print line

    Returns: None
    """

    assert len(title) < length - 4, "Title is longer than line length - 4"

    len_before_title = (length - len(title)) // 2 - 1
    len_after_title = length - len(title) - (length - len(title)) // 2 - 1

    print_line(char=char, length=length, before=before)
    for _ in range(wrap - 1):
        print(char, " " * (length - 2), char, sep="")
    print(char, " " * len_before_title, title, " " * len_after_title, char,
          sep="")

    for _ in range(wrap - 1):
        print(char, " " * (length - 2), char, sep="")
    print_line(char=char, length=length, after=after)
    # End of function print_wrap_title


def print_table(tabular_data: list,
                headers: list,
                table_format: str = "grid"):
    """
    Print nice table in using tabulate

    Example:
    print_table(tabular_data=[["value1", "value2"], ["value3", "value4"]],
               headers=["headers 1", "headers 2"],
               table_format="grid"))

    Args:
        tabular_data: data in table
        headers: column headers
        table_format: format

    Returns:

    """
    print(tabulate(tabular_data, headers, table_format))


def var_param(var: torch.Tensor) -> torch.Tensor:
    """
    Parametrize variance

    Args:
        var: variance

    Returns: parametrized variance

    """
    assert var.min() >= 0
    safe_log = 1e-8
    parametrized_cov = torch.log(var + safe_log)
    return parametrized_cov


def de_param(parametrized_cov: torch.Tensor) -> torch.Tensor:
    """
    De-parametrize covariance

    Args:
        parametrized_cov: parametrized covariance

    Returns: covariance

    """
    # lower_bound = 0
    # lower_bound = 1e-3
    lower_bound = 1e-2
    # lower_bound = 1e-1
    softplus = torch.nn.Softplus()
    cov = softplus(parametrized_cov) + lower_bound
    return cov


def check_torch_device() -> bool:
    """
    Check if GPU is available and set default torch datatype

    Returns:
        None
    """
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # torch.multiprocessing.set_start_method(method="spawn")
        return True
    else:
        return False


def generate_dummy_trajectory(sparse: bool = False) -> (pd.DataFrame,
                                                        pd.DataFrame):
    """
    Generate dummy data in the form of sin cos curves

    Args:
        sparse: if data has missing values "nan"

    Returns:
        dummy data in pandas DataFrames

    """
    t = np.linspace(0.0, 2 * np.pi, 10)
    phase = np.random.uniform(0, 2 * np.pi)
    x = np.sin(t + phase)
    y = np.cos(t + phase)
    # xy = np.stack([x, y], axis=-1).tolist()
    xy = list(np.stack([x, y], axis=-1))

    height = np.array([np.random.uniform(0, 1)])
    width = np.array([np.random.uniform(0, 1)])
    # hw = np.stack([height, width], axis=-1).tolist()
    hw = list(np.stack([height, width], axis=-1))
    if sparse:
        for i in range(1, 5):
            y[-i] = np.nan

    data = pd.DataFrame({"t": t,
                         "x": x,
                         "y": y,
                         "xy": xy})

    data_static = pd.DataFrame({"height": height,
                                "width": width,
                                "hw": hw})
    return data, data_static


def generate_dummy_dataset(num_traj: int,
                           save_path: str = None,
                           sparse: bool = False) -> (list, list):
    """
    Generate dummy dataset in the form of pandas list

    Args:
        num_traj: number of trajectories to be generated
        save_path: the directory to save all the data, in the format of .csv
        sparse: if data has missing values "nan"

    Returns:
        a list with pandas DataFrame

    """

    # Remove existing directory
    remove_file_dir(save_path)

    # Generate directory in path
    os.makedirs(save_path)

    trajs = list()
    trajs_static = list()

    for index in range(num_traj):
        # Generate data
        data, data_static = generate_dummy_trajectory(sparse)
        trajs.append(data)
        trajs_static.append(data_static)

    if save_path is not None:
        for (index, (traj, traj_static)) in enumerate(zip(trajs, trajs_static)):
            traj.to_csv(path_or_buf=save_path + "/" + str(index) + ".csv",
                        index=False,
                        quoting=csv.QUOTE_ALL)
            traj_static.to_csv(path_or_buf=save_path + "/" + 'static_' +
                                           str(index) + ".csv",
                               index=False,
                               quoting=csv.QUOTE_ALL)

    return trajs, trajs_static


def remove_file_dir(path):
    """
    Remove file or directory
    Args:
        path: path to directory or file

    Returns:
        True if successfully remove file or directory

    """
    if not os.path.exists(path):
        return False
    elif os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
        return True
    else:
        shutil.rmtree(path)
        return True


def get_dataset_dir(dataset_name: str):
    """
    Get the path to the directory storing the dataset
    Args:
        dataset_name: name of the dataset

    Returns:
        path to the directory storing the dataset
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "dataset",
                        dataset_name)


def clean_and_get_tmp_dir():
    """
    Get the path to the tmp folder
    Args:
        dataset_name: name of the dataset

    Returns:
        path to the directory storing the dataset
    """
    tmp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "tmp_video")
    remove_file_dir(tmp_path)
    os.mkdir(tmp_path)
    return tmp_path


def get_config_path(config_name: str, config_type: str = "local"):
    """
    Get the path to the config file
    Args:
        config_name: name of the config file
        config_type: configuration type

    Returns:
        path to the config file
    """
    # Check config type
    assert config_type in get_config_type(), \
        "Unknown config type."
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "config",
                        config_type,
                        config_name + ".yaml")


def get_log_dir(log_name: str):
    """
    Get the dir to the log
    Args:
        log_name: log's name

    Returns:
        directory to log file
    """

    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "log",
                        log_name,
                        get_formatted_date_time())


def get_nn_save_paths(log_dir: str, nn_name: str, epoch: int,
                      use_epoch: bool = True):
    """
    Get path storing nn structure parameters and nn weights
    Args:
        log_dir: directory to log
        nn_name: name of NN
        epoch: number of training epoch
        use_epoch: if epoch should be added

    Returns:
        path to nn structure parameters
        path to nn weights
    """
    s_path = os.path.join(log_dir,
                          nn_name + "_parameters.pkl")
    if use_epoch:
        w_path = os.path.join(log_dir,
                              nn_name + "_weights_{:d}".format(epoch))
    else:
        w_path = os.path.join(log_dir,
                              nn_name + "_weights")
    return s_path, w_path


def get_mnist_pretrained_path(mnist_name: str):
    log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "mnist_model")
    return get_nn_save_paths(log_dir, mnist_name, None, False)


def get_config_type():
    """
    Register current config type
    Returns:

    """
    return {"local", "mp", "cluster"}


def parse_config(config_path: str, config_type: str = "local"):
    """
    Parse config file into a dictionary
    Args:
        config_path: path to config file
        config_type: configuration type

    Returns:
        configuration in dictionary
    """
    assert config_type in get_config_type(), \
        "Unknown config type"

    all_config = list()
    with open(config_path, "r") as f:
        for config in yaml.load_all(f, yaml.FullLoader):
            all_config.append(config)
    if config_type == "cluster":
        return all_config
    else:
        return all_config[0]


def dump_config(config_dict: dict, config_name: str):
    """
    Dump configuration into yaml file
    Args:
        config_dict: config dictionary to be dumped
        config_name: config file name
    Returns:
        None
    """

    # Generate config path
    config_path = get_config_path(config_name)

    # Remove old config if exists
    remove_file_dir(config_path)

    # Write new config to file
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)


def get_file_names_in_directory(directory: str):
    """
    Get file names in given directory
    Args:
        directory: directory where you want to explore

    Returns:
        file names in list

    """
    file_names = None
    try:
        (_, _, file_names) = next(os.walk(directory))
    except StopIteration as e:
        print("Cannot read files from directory: ", directory)
        exit()
    return file_names


def move_files_from_to(from_dir: str,
                       to_dir: str,
                       copy=False):
    """
    Move or copy files from one directory to another
    Args:
        from_dir: from directory A
        to_dir: to directory B
        copy: True if copy instead of move

    Returns:
        None
    """
    file_names = get_file_names_in_directory(from_dir)
    for file in file_names:
        from_path = os.path.join(from_dir, file)
        to_path = os.path.join(to_dir, file)
        if copy:
            shutil.copy(from_path, to_path)
        else:
            shutil.move(from_path, to_path)


def read_dataset(dataset_name: str,
                 shuffle: bool = False,
                 seed=None) -> (list, list):
    """
    Read raw data from files

    Args:
        dataset_name: name of dataset to be read
        shuffle: shuffle the order of dataset files when reading
        seed: random seed

    Returns:
        list_pd_df: a list of pandas DataFrames with time-dependent data
        list_pd_df_static: ... time-dependent data, can be None

    """
    # Get dir to dataset
    dataset_dir = get_dataset_dir(dataset_name)

    # Get all data-file names
    file_names = get_file_names_in_directory(dataset_dir)

    # Check file names for both time-dependent and time-independent data exist
    num_files = len(file_names)
    file_names = os_sorted(file_names)

    # Check if both time-dependent and time-independent dataset exist
    if all(['static' in name for name in file_names]):
        # Only time-independent dataset
        list_pd_df = [pd.DataFrame() for data_file in file_names]
        # Construct a empty dataset for time independent data
        list_pd_df_static = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                         quoting=csv.QUOTE_ALL)
                             for data_file in file_names]
    elif all(['static' not in name for name in file_names]):
        # Only time-dependent dataset
        list_pd_df = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                  quoting=csv.QUOTE_ALL)
                      for data_file in file_names]
        # Construct a empty dataset for time independent data
        list_pd_df_static = [pd.DataFrame() for data_file in file_names]
    else:
        # Both exist
        assert \
            all(['static' not in name for name in file_names[:num_files // 2]])
        assert all(['static' in name for name in file_names[num_files // 2:]])

        # Read data from files and generate list of pandas DataFrame
        list_pd_df = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                  quoting=csv.QUOTE_ALL)
                      for data_file in file_names[:num_files // 2]]
        list_pd_df_static = [pd.read_csv(os.path.join(dataset_dir, data_file),
                                         quoting=csv.QUOTE_ALL)
                             for data_file in file_names[num_files // 2:]]

    if shuffle:
        list_zip = list(zip(list_pd_df, list_pd_df_static))
        random.seed(seed)
        random.shuffle(list_zip)
        list_pd_df, list_pd_df_static = zip(*list_zip)

    # Return
    return list_pd_df, list_pd_df_static


def build_lower_matrix(param_diag, param_off_diag):
    """
    Compose the lower triangular matrix L from diag and off-diag elements
    Args:
        param_diag: diagonal parameters
        param_off_diag: off-diagonal parameters

    Returns:
        Lower triangular matrix L

    """
    l_size = list(param_diag.shape)
    dim_pred = l_size[-1]
    l_size.append(dim_pred)
    L = torch.zeros(size=l_size)
    # Fill diagonal terms
    diag = range(dim_pred)
    L[..., diag, diag] = param_diag[..., :]
    if param_off_diag is not None:
        # Fill off-diagonal terms
        [row, col] = torch.tril_indices(dim_pred, dim_pred, -1)
        L[..., row, col] = param_off_diag[..., :]

    return L


def make_tuple(data):
    """
    Make data a tuple, i.e. (data)
    Args:
        data: some data

    Returns:
        (data) if it is not a tuple
    """
    if isinstance(data, tuple):
        return data
    else:
        return (data,)  # Do not use tuple()


def run_time_test(lock: bool):
    """
    A manual running time computing function. It will print the running time
    for the every seoncd call

    E.g.:
    run_time_test(lock=True)
    some_func1()
    some_func2()
    ...
    run_time_test(lock=False)

    Args:
        lock: flag indicating if time counter starts

    Returns:
        None

    """
    if not hasattr(run_time_test, "lock_state"):
        run_time_test.lock_state = False
        run_time_test.last_run_time = time.time()
        run_time_test.duration_list = list()
    assert run_time_test.lock_state != lock
    run_time_test.lock_state = lock
    if lock is False:
        duration = time.time() - run_time_test.last_run_time
        run_time_test.duration_list.append(duration)
        run_time_test.last_run_time = time.time()
        print("duration", duration)
    else:
        run_time_test.last_run_time = time.time()


def from_string_to_array(s: str):
    """
    Convert string in Pandas DataFrame cell to numpy array
    Args:
        s: string, e.g. "[1.0   2.3   4.5 \n 5.3   5.6]"

    Returns:
        1D numpy array
    """
    return np.asarray(s[1:-1].split(),
                      dtype=np.float64)


def get_formatted_date_time():
    """
    Get formatted date and time, e.g. May_05_2021_22_14_31
    Returns:
        dt_string: date time string
    """
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string


def debug_plot(x, y: list):
    """
    One line to plot some variable for debugging
    Args:
        x: data used for x-axis, can be None
        y: list of data used for y-axis

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    plt.figure()
    if x is not None:
        for yi in y:
            plt.plot(x, yi)
    else:
        for yi in y:
            plt.plot(yi)
    plt.show()


def from_figures_to_video(figure_list: list, video_name: str,
                          interval: int = 2000):
    """
    Generate and save a video given a list of figures
    Args:
        figure_list: list of matplotlib figure objects
        video_name: name of video
        interval: interval between two figures in [ms]

    Returns:
        path to the saved video
    """
    figure, ax = plt.subplots()
    figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    frames = []

    tmp_path = clean_and_get_tmp_dir()
    for i, fig in enumerate(figure_list):
        fig.savefig(tmp_path + "/{}.png".format(i), dpi=300,
                    bbox_inches="tight")

    for j in range(len(figure_list)):
        image = plt.imread((tmp_path + "/{}.png".format(j)))
        img = plt.imshow(image, animated=True)
        plt.axis('off')
        plt.gca().set_axis_off()

        frames.append([img])

    ani = animation.ArtistAnimation(figure, frames, interval=interval,
                                    blit=True,
                                    repeat=False)
    save_path = tmp_path + '/' + video_name + '.mp4'
    ani.save(save_path, dpi=300)

    return save_path


def add_expand_dim(data,
                   add_dim_indices: list,
                   add_dim_sizes: list):
    """
    Add additional dimensions to tensor and expand accordingly
    Args:
        data: tensor to be operated. Torch.Tensor or numpy.ndarray
        add_dim_indices: the indices of added dimensions in the result tensor
        add_dim_sizes: the expanding size of the additional dimensions

    Returns:
        result: result tensor
    """
    num_data_dim = data.ndim
    num_dim_to_add = len(add_dim_indices)

    add_dim_reverse_indices = [num_data_dim + num_dim_to_add + idx
                               for idx in add_dim_indices]

    str_add_dim = ""
    str_expand = ""
    add_dim_index = 0
    for dim in range(num_data_dim + num_dim_to_add):
        if dim in add_dim_indices or dim in add_dim_reverse_indices:
            str_add_dim += "None, "
            str_expand += str(add_dim_sizes[add_dim_index]) + ", "
            add_dim_index += 1
        else:
            str_add_dim += ":, "
            str_expand += "-1, "

    str_add_dime_eval = "data[" + str_add_dim + "]"
    return eval("eval(str_add_dime_eval).expand(" + str_expand + ")")


class TrajectoryLoader:
    """
    Static helper class for loading trejectoires from json files
    Credit: IMEDNET
    """

    @staticmethod
    def load(file):
        """
        Loads trajectory from file

        load(file) ->  trajectory containing all points in a form 'point = [x,y,t]'
        file -> file containing trajectory in json format
        """
        try:
            json_data = open(file).read()

            data = json.loads(json_data)
            path = data['Path']
            path = path.split('], [')
            path[0] = path[0][1:]
            path[-1] = path[-1][:-1]

            points = []

            for point in path:
                point = point.split(',')
                point = [float(x) for x in point]
                if len(point) != 3:
                    raise Exception('Error in file ' + file)
                points.append(point)
            points = np.array(points)
            if np.where(points[:, 2] == 0)[0].size > 1:
                print('File ' + file + 'is corrupted')
            return points
        except:
            print('Could not load file ' + file)

    @staticmethod
    def getAvailableTrajectoriesNumbers(folder):
        """
        Checks folder for all json files containing trajectories and returns its numbers gained from filenames
        Each number n coresponds to the n-th image from the MNIST dataset

        getAvailableTrajectoriesNumbers(folder) -> sorted list of numbers of available trajectories inf the folder
        folder -> the string path of the folder to check
        """
        datoteke = [f for f in os.listdir(folder)]
        available = []
        for datoteka in datoteke:
            if datoteka.startswith('image_') and datoteka.endswith('.json'):
                number = datoteka[len('image_'):]
                number = int(number[:-len('.json')])
                available.append(number)
        return sorted(available)

    @staticmethod
    def getTrajectoryFile(folder, n):
        """
        Returns the string path to the file with the n-th trajectory in the given folder

        getTrajectoryFile(folder, n) -> string path to the trajectory
        folder -> the string path to the folder containing the trajectory files
        n -> the sequential number of the desired trajectory
        """
        datoteke = [f for f in os.listdir(folder)]
        name = 'image_' + str(n) + '.json'
        if name in datoteke:
            return folder + "/" + name
        return False

    @staticmethod
    def loadNTrajectory(folder, n):
        """
        Loads n-th trajectory from the given folder

        loadNTrajectory(folder,n) ->
        folder -> the string path to the folder containing the trajectory files
        n -> the sequential number of the desired trajectory
        """
        return TrajectoryLoader.load(
            TrajectoryLoader.getTrajectoryFile(folder, n))


def load_mnist_data(mnist_folder):
    """
    Loads data from the folder containing mnist files
    Credit: originally from IMEDNET
    """
    mnistData = MNIST(mnist_folder)
    images, labels = mnistData.load_training()
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def interpolation(length, x_ori, data_ori):
    """
    Interpolates trajectories to desired length and data density

    Args:
        length: number of desired points
        x_ori: original data time
        data_ori: original data value

    Returns:

    """
    x_interp = np.linspace(x_ori[0], x_ori[-1], length)

    # Initialize result array as shape [num_time, num_dof]
    data_interp = np.zeros((length, data_ori.shape[1]))

    # Loop over dof
    for k in range(data_ori.shape[1]):
        #                            desired, original, data
        data_interp[:, k] = np.interp(x_interp, x_ori, data_ori[:, k])

    return data_interp


def joint_to_conditional(joint_mean, joint_L, sample_x):
    """
    Given joint distribution p(x,y), and a sample of x, do:
    Compute conditional distribution p(y|x)
    Args:
        joint_mean: mean of joint distribution
        joint_L: cholesky distribution of joint distribution
        sample_x: samples of x

    Returns:
        conditional mean and cov
    """

    # Shape of joint_mean:
    # [*add_dim, dim_x + dim_y]
    #
    # Shape of joint_L:
    # [*add_dim, dim_x + dim_y, dim_x + dim_y]
    #
    # Shape of sample_x:
    # [*add_dim, dim_x]
    #
    # Shape of conditional_mean:
    # [*add_dim, dim_y]
    #
    # Shape of conditional_cov:
    # [*add_dim, dim_y, dim_y]

    # Check dimension
    add_dim = list(joint_mean.shape[:-1])
    dim_x = sample_x.shape[-1]
    # dim_y = joint_mean.shape[-1] - dim_x

    # Decompose joint distribution parameters
    mu_x = joint_mean[..., :dim_x]
    mu_y = joint_mean[..., dim_x:]

    L_x = joint_L[..., :dim_x, :dim_x]
    L_y = joint_L[..., dim_x:, dim_x:]
    L_x_y = joint_L[..., dim_x:, :dim_x]

    # C = torch.einsum('...ik,...jk->...ij', L_x, L_x_y)
    # C_T = torch.einsum('...ij->...ji', C)

    # Unfortunately pytorch does not support batch manner cholesky_inverse()
    # L_x_inv = torch.linalg.solve(L_x, torch.ones(size=[*add_dim, dim_x]))
    L_x_inv = torch.linalg.inv(L_x)
    # L_x_inv_T = torch.einsum('...ij->...ji', L_x_inv)

    cond_mean = mu_y + torch.einsum('...ij,...jk,...k->i',
                                    L_x_y, L_x_inv, sample_x - mu_x)

    cond_L = L_y

    return cond_mean, cond_L


def euler2mat(euler):
    """ Convert Euler Angles to Rotation Matrix.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def interpolation(length, x_ori, data_ori):
    """
    Interpolates trajectories to desired length and data density

    Args:
        length: number of desired points
        x_ori: original data time
        data_ori: original data value

    Returns:

    """
    x_interp = np.linspace(x_ori[0], x_ori[-1], length)

    # Initialize result array as shape [num_time, num_dof]
    data_interp = np.zeros((length, data_ori.shape[1]))

    # Loop over dof
    for k in range(data_ori.shape[1]):
        #                            desired, original, data
        data_interp[:, k] = np.interp(x_interp, x_ori, data_ori[:, k])

    return data_interp


def pos2vel(pos, t):
    """
    change pos to vel

    Args:
        pos: array of position
        t: time scope for position
    Returns:

    """
    vel = np.zeros(pos.shape)
    delta_t = t / pos.shape[0]
    for i in range(pos.shape[0]):
        if i == 0:
            vel[i] = 0
        elif i == pos.shape[0] - 1:
            vel[i] = (pos[i] - pos[i-1]) / delta_t
        else:
            vel[i] = (pos[i+1] - pos[i-1]) / (2 * delta_t)
    return vel


def dict2frame(dict_data: dict):
    """
    transfer dict to pd.DataFrame

    Args:
        dict_data: dictionary to be transferred
    Returns:

    """
    if 't' in dict_data.keys():
        df = pd.DataFrame({'index': np.linspace(0, len(dict_data['t'])-1, len(dict_data['t']), dtype=int)})
    else:
        df = pd.DataFrame()
    for key, value in dict_data.items():
        if len(value.shape) < 2:
            df_value = pd.DataFrame({key: value})
            df = pd.concat([df, df_value], axis=1)
        else:
            for i in range(value.shape[1]):
                df_value = pd.DataFrame({key + '_' + str(i): value[:, i]})
                df = pd.concat([df, df_value], axis=1)
    return df


def compound_array(list_array: list):
    """
    compound array to a list, which can be saved in a DataFrame

    Args:
        list_array: array to be compounded
    Returns:

    """
    list_compound_data = list()
    for arr in list_array:
        for dof in range(arr.shape[1]):
            list_compound_data.append(arr[:, dof])
    compound_data = list(np.stack(list_compound_data, axis=-1))
    return compound_data
