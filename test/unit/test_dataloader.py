"""
Unit test of ZOD dataset loader and partitioner.
"""

from logging import INFO
import torch
import pytest

from edge_code.data_loader import load_datasets
from server_code.data_partitioner import partition_train_data
from common.static_params import PartitionStrategy
from common.logger import fleet_log
from common.static_params import global_configs


@pytest.fixture()
def n_clients():
    return 100


def test_dataloader(n_clients):

    # partition data among clients
    partitions = partition_train_data(
        PartitionStrategy.RANDOM, n_clients
    )
    fleet_log(INFO, partitions['0'][:5])
    fleet_log(INFO, partitions['1'][:5])

    # create dataloaders for first client
    train_loader, val_loader, test_loader = load_datasets(partitions['0'])
    fleet_log(INFO, train_loader)
    fleet_log(INFO, type(train_loader))

    # checks
    assert len(partitions.keys()) == n_clients

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
