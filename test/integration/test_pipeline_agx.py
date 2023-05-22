"""
Pipeline test.
"""

import os
import yaml
import pathlib
import subprocess
import threading
import socket
import csv
import shutil
import pytest
import logging
import numpy as np
from typing import List

from common.utilities import net_instance, get_parameters
import main as server_main


def change_static_params():
    DEVICE_DICT = {
                #"agx4.nodes.edgelab.network" : 0, NOT WORKING ATM, fix it!! (flush and reinstall)
                # "agx6.nodes.edgelab.network": 0,
                "agx9.nodes.edgelab.network": 0,
                # "agx10.nodes.edgelab.network": 0,
                # "orin1.nodes.edgelab.network": 0,
                # "orin2.nodes.edgelab.network": 0
            }
    NUM_CLIENTS = 1
    PERCENTAGE_OF_DATA = 0.0001
    BATCH_SIZE = 8
    VAL_FACTOR = 0.1 # percentage of train data on edge node to use for validation
    SUBSET_FACTOR = 0.001 # 0.003 # subset of test frames to use



#TODO: mock edge_com.update_model
# TODO: fixture for "parameters"
@pytest.fixture()
def mocked_edge_node_training(mocker):
    model = net_instance("mocked_model")
    params = get_parameters(model)
    mocked_parameters = list(np.array(params, dtype=object)['arr_0'])
    # mocked_parameters = list(np.load("test/integration/parameters.npz", allow_pickle=True)['arr_0'])
    mocker.patch("edge_com.edge_com.update_model", return_value=mocked_parameters)


# def test_pipeline_agx(caplog, mocker, mocked_edge_node_training):
def test_pipeline_agx(caplog, mocker):
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)

    # delete old files if present
    for file in os.listdir(os.path.join(ROOT, "tmp")):
        if file.split(".")[-1] == "npz":
            os.remove(os.path.join(ROOT, "tmp", file))

    # change settings
    # change_static_params()
    
    logging.getLogger(__name__)
    
    # mocker.patch("main.ZODImporter", return_value=2)

    server_main.main()
    
    # change_back_static_params()
    
    # TODO: assert these npz files are stored in server
    # TODO: assert dimenions of these files

    # tmp_dir = os.listdir(os.path.join(ROOT, "tmp"))
    # assert "agg.npz" in os.listdir(tmp_dir)
    # assert "partitions.npz" in os.listdir(tmp_dir)
    
    assert True
    assert "Ray initialized" in caplog.text


def main():
    test_pipeline_agx()


if __name__ == "__main__":
    main()
