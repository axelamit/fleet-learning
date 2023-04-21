import unittest
from edge_code.data_loader import *
from server_code.data_partitioner import *
from common.static_params import *

class TestDataLoad(unittest.TestCase):

    def test_dataload(self):
        partitions = partition_train_data(PartitionStrategy.RANDOM, 10000)
        log(partitions['0'])
        log(partitions['1'])
        train, val, test = load_datasets(partitions['0'])
        log(train)
        log(type(train))

if __name__ == '__main__':
    unittest.main()