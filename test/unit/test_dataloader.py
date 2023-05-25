from edge_code.data_loader import load_datasets
from server_code.data_partitioner import partition_train_data
from common.static_params import PartitionStrategy
from common.logger import log
from logging import INFO
import torch
import pytest

@pytest.fixture()
def static_params():
    return {
        "IMG_SIZE": 224,  # 256
        "BATCH_SIZE": 8,
        "OUTPUT_SIZE": 66,
        "NUM_OUTPUT": 51,
        "NUM_CLIENTS": 1000,
    }


def test_dataloader(mocker, static_params):
    # change static params
    for key, value in static_params.items():
        mocker.patch(f"main.global_configs.{key}", return_value=value)
        mocker.patch(f"edge_main.global_configs.{key}", return_value=value)
        mocker.patch(f"common.datasets.global_configs.{key}", return_value=value)
        mocker.patch(f"common.groundtruth_utils.global_configs.{key}", return_value=value)
        mocker.patch(f"common.models.global_configs.{key}", return_value=value)
        mocker.patch(f"edge_code.data_loader.global_configs.{key}", return_value=value)
        mocker.patch(f"server_code.data_partitioner.global_configs.{key}", return_value=value)

    # partition data among clients
    n_clients = static_params["NUM_CLIENTS"]
    partitions = partition_train_data(PartitionStrategy.RANDOM, n_clients)
    log(INFO, partitions['0'][:5])
    log(INFO, partitions['1'][:5])

    # create dataloaders for first client
    train_loader, val_loader, test_loader = load_datasets(partitions['0'])
    log(INFO, train_loader)
    log(INFO, type(train_loader))

    # checks
    assert len(partitions['0']) == 91
    assert len(partitions.keys()) == n_clients

    train_data, train_targets = next(iter(train_loader))
    val_data, val_targets = next(iter(val_loader))
    test_data, test_targets = next(iter(test_loader))

    data_shape = torch.Size(
        [
            static_params["BATCH_SIZE"],
            3,
            static_params["IMG_SIZE"],
            static_params["IMG_SIZE"],
        ]
    )

    targets_shape = torch.Size(
        [static_params["BATCH_SIZE"], static_params["NUM_OUTPUT"]]
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
