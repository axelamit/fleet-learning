"""
Pipeline test off complete FL session with real AGX nodes and no mocking.

Please note that all changes need to be pushed to git in order to be applicable for
clients.
"""

import os
import pathlib
import logging
import numpy as np

from common.static_params import global_configs
import main as server_main


def test_pipeline(
    caplog,
    mocker,
):
    logging.getLogger(__name__)

    # delete old files if present
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
    tmp_dir = os.path.join(ROOT, "tmp")
    for file in os.listdir(tmp_dir):
        if file.split(".")[-1] == "npz" and file.split(".")[0] != "res0":
            os.remove(os.path.join(ROOT, "tmp", file))

    # run main script
    server_main.main()

    # assert results are retrieved by server
    assert "agg.npz" in os.listdir(tmp_dir)
    assert "partitions.npz" in os.listdir(tmp_dir)

    partition = np.load("partitions.npz")[0]
    parameters = np.load("agg.npz", allow_pickle=True)['arr_0']

    print(partition.shape)
    print(parameters.shape)

    # assert info log contains
    assert "Ray initialized" in caplog.text
    assert f"fit_round 1: strategy sampled {global_configs.NUM_CLIENTS} clients (out of {global_configs.NUM_CLIENTS})" in caplog.text
    assert "fit_round 1 received 1 results and 0 failures" in caplog.text
    assert f"evaluate_round 1: strategy sampled {global_configs.NUM_CLIENTS} clients (out of {global_configs.NUM_CLIENTS})" in caplog.text
    assert "evaluate_round 1 received 1 results and 0 failures" in caplog.text
    assert "FL finished" in caplog.text


def main():
    test_pipeline()


if __name__ == "__main__":
    main()
