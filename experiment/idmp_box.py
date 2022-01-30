import copy
import csv
import os
import time
import matplotlib.pyplot as plt

from nmp.net import MPNet
from nmp.loss import *


def idmp_box_std(**kwargs):
    util.print_wrap_title("NMP IDMP of BOX Dragging STD")
    # kwargs["idmp_config"] = "IDMP_Panda_Box_config"
    # config_path = util.get_config_path("NMP_IDMP_PANDA_BOX_STD_config")

    kwargs["idmp_config"] = "IDMP_Panda_Box_25_config"
    config_path = util.get_config_path("NMP_IDMP_PANDA_BOX_STD_25_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_PANDA_BOX_STD",
                                       mp_net.epoch)


def idmp_box_cov(**kwargs):
    util.print_wrap_title("NMP IDMP of BOX Dragging COV")
    kwargs["idmp_config"] = "IDMP_Panda_Box_config"
    config_path = util.get_config_path("NMP_IDMP_PANDA_BOX_COV_config")

    # kwargs["idmp_config"] = "IDMP_Panda_Box_25_config"
    # config_path = util.get_config_path("NMP_IDMP_PANDA_BOX_COV_25_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_PANDA_BOX_COV",
                                       mp_net.epoch)


def idmp_box_cov_act(**kwargs):
    util.print_wrap_title("NMP IDMP of BOX Dragging COV with actual robot "
                          "state")
    kwargs["ctx"] = "cart_object"
    kwargs["idmp_config"] = "IDMP_Panda_Box_config"
    config_path = util.get_config_path("NMP_IDMP_PANDA_BOX_COV_ACTUAL_config")

    config = util.parse_config(config_path)
    kwargs["dataset_name"] = config["dataset"]["name"]
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_PANDA_BOX_COV_ACTUAL",
                                       mp_net.epoch)


def test(exp: str, restart=True):
    exp_api = dict()
    exp_api["idmp_box_std"] = {
        "func": idmp_box_std,
        "api": "",
        "best_epoch": 10000}

    exp_api["idmp_box_cov"] = {
        "func": idmp_box_cov,
        "api": "",
        "best_epoch": 10000}


    exp_api["idmp_box_cov_act"] = {
        "func": idmp_box_cov_act,
        "api": "",
        "best_epoch": 10000}


    # Specify task
    exp_func = exp_api[exp]["func"]
    exp_kwargs = \
        {"max_epoch": 10000 if restart else exp_api[exp]["best_epoch"],
         "init_epoch": 0 if restart else exp_api[exp]["best_epoch"],
         "num_mc_smp": None,
         "model_api": None if restart else exp_api[exp]["api"],
         "manual_test": False}

    # Run task
    exp_func(**exp_kwargs)


def main():
    util.check_torch_device()

    # test(exp="idmp_box_std", restart=True)
    # test(exp="idmp_box_std", restart=False)

    test(exp="idmp_box_cov", restart=True)
    # test(exp="idmp_box_cov", restart=False)

    # test(exp="idmp_box_cov_act", restart=True)
    # test(exp="idmp_box_cov_act", restart=False)

if __name__ == "__main__":
    main()
