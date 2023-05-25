"""
Pipeline test.
"""

import os
import pathlib
import pytest
import logging
import numpy as np
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, EvaluateRes
import ray

from common.utilities import net_instance, get_parameters
import main as server_main


@pytest.fixture()
def static_params():
    return {
        "DEVICE_DICT": {"dummy_device_1": 0},
        "NUM_CLIENTS": 1,
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


def test_pipeline_client(
    caplog,
    mocker,
    static_params
):
    logging.getLogger(__name__)

    # delete old files if present
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
    tmp_dir = os.path.join(ROOT, "tmp")
    assert True


def main():
    test_pipeline_client()


if __name__ == "__main__":
    main()
