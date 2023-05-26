"""
Unit test of ZOD dataset loader and partitioner.
"""

from logging import INFO
import torch
import pytest
from dataclasses import dataclass
from edge_code.data_loader import load_datasets
from server_code.data_partitioner import partition_train_data
from common.static_params import PartitionStrategy
from common.logger import fleet_log

@pytest.fixture()
def global_configs():
    @dataclass
    class GlobalConfigs:
        def __init__(self):
            self.SERVER_MAIN_PATH = "/root/Fleet/oscar/fleet-learning/"
            self.TARGET_DISTANCES = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165]
            self.NUM_CLIENTS = 1000
            self.PERCENTAGE_OF_DATA = 0.02
            self.NUM_OUTPUT = 51
            self.IMG_SIZE = 256
            self.RUN_PRETRAINED = False
            self.BATCH_SIZE = 8
            self.VAL_FACTOR = 0.1  # percentage of train data to use for validation
            self.SUBSET_FACTOR = 0.003  # subset of test frames to use
            self.OUTPUT_SIZE = 66
            self.STORED_GROUND_TRUTH_PATH = "/mnt/ZOD/ground_truth.json"
            self.DATASET_ROOT = "/mnt/ZOD"

    return GlobalConfigs()


def test_dataloader(mocker, global_configs):

    # mock global params
    mocker.patch("common.static_params.GlobalConfigs", return_value=global_configs)

    # partition data among clients
    partitions = partition_train_data(
        PartitionStrategy.RANDOM, global_configs.NUM_CLIENTS
    )
    fleet_log(INFO, partitions['0'][:5])
    fleet_log(INFO, partitions['1'][:5])

    # create dataloaders for first client
    train_loader, val_loader, test_loader = load_datasets(partitions['0'])
    fleet_log(INFO, train_loader)
    fleet_log(INFO, type(train_loader))

    # checks
    assert len(partitions.keys()) == global_configs.NUM_CLIENTS

    train_data, train_targets = next(iter(train_loader))
    val_data, val_targets = next(iter(val_loader))
    test_data, test_targets = next(iter(test_loader))

    data_shape = torch.Size(
        [
            global_configs.BATCH_SIZE,
            3,
            global_configs.IMG_SIZE,
            global_configs.IMG_SIZE,
        ]
    )

    targets_shape = torch.Size(
        [global_configs.BATCH_SIZE, global_configs.NUM_OUTPUT]
    )

    assert train_data.shape == data_shape
    assert train_targets.shape == targets_shape

    assert val_data.shape == data_shape
    assert val_targets.shape == targets_shape

    assert test_data.shape == data_shape
    assert test_targets.shape == targets_shape


def main():
    test_dataloader()


if __name__ == '__main__':
    main()
