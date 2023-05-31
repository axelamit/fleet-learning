"""
Pipeline test of client side, with server side mocked.
"""

import os
import pathlib
import pytest
import logging
import numpy as np

from common.static_params import PartitionStrategy
from common.utilities import net_instance, get_parameters
from server_code.data_partitioner import partition_train_data
import edge_main

@pytest.fixture()
def static_params():
    return {
        "DEVICE_DICT": {"dummy_device_1": 0},
        "NUM_CLIENTS": 1000,
        "PERCENTAGE_OF_DATA": 0.001,
        "IMG_SIZE": 224, #256,
        "RUN_PRETRAINED": False,
        "BATCH_SIZE": 8,
        "VAL_FACTOR": 0.1,
        "SUBSET_FACTOR": 0.003,
        "NUM_GLOBAL_ROUNDS": 1,
        "NUM_LOCAL_EPOCHS": 1,
        "OUTPUT_SIZE": 66,
    }


@pytest.fixture()
def partitions(static_params):
    n_clients = static_params["NUM_CLIENTS"]
    return partition_train_data(PartitionStrategy.RANDOM, n_clients)


@pytest.fixture()
def agg():
    server_model = net_instance("server")
    return get_parameters(server_model)


def test_pipeline_client(
    caplog,
    mocker,
    static_params,
    partitions,
    agg,
):
    logging.getLogger(__name__)

    # delete old files if present
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
    tmp_dir = os.path.join(ROOT, "tmp")

    for file in os.listdir(tmp_dir):
        if file == "res0.npz":
            os.remove(os.path.join(ROOT, "tmp", file))

    testargs = [None, "0"]
    mocker.patch.object(edge_main.sys, 'argv', testargs)

    # mock GPU usage
    mocker.patch("edge_main.use_gpu")

    # mock SSH connection
    mocker.patch("edge_main.SSHClient")

    # mock file system operations
    mocker.patch("edge_main.os.remove")

    # mock reading agg.npz file
    mocker.patch("edge_main.list", return_value=agg)

    # mock reading partitions.npz file
    mocker.patch("edge_main.np.load", side_effect=[{"arr_0": "dummy_agg"}, partitions])

    edge_main.main()

    assert "res0.npz" in os.listdir(tmp_dir)
    mocker.stopall()
    np.load(os.path.join(tmp_dir, "res0.npz"), allow_pickle=True)["arr_0"]

    # os.remove(os.path.join(tmp_dir, "res0.npz"))
    assert "done" in caplog.text


def main():
    test_pipeline_client()


if __name__ == "__main__":
    main()
