import numpy as np
from edge_code.data_loader import load_datasets
from common.utilities import train, net_instance, get_parameters, set_parameters
from common.static_params import global_configs
from common.logger import fleet_log
from logging import INFO


def train_simulated(parameters, cid, config):
    # Get the command-line arguments

    fleet_log(INFO,'Simulated FL training starting')
    
    model = net_instance(f"{cid}")
    set_parameters(model, parameters)

    fleet_log(INFO,f"load the data partition for client cid {cid}")
    
    server_round = config["server_round"]
    partition = np.load("tmp/partitions.npz")[cid]
    n_data = len(partition)//global_configs.NUM_GLOBAL_ROUNDS
    partition = partition[n_data*(server_round-1):n_data*server_round]
    
    fleet_log(INFO,'data partition loaded')
    
    fleet_log(INFO,'load dataset to ram')
    
    trainloader, valloader, testloader = load_datasets(partition)
    
    fleet_log(INFO,'train the model')
    
    losses, accs, val_losses, val_accs = train(model, trainloader, valloader, global_configs.NUM_LOCAL_EPOCHS, plot = True,client_cid = cid)
    
    fleet_log(INFO,f"loss: {losses}, accs: {accs}, val_losses: {val_losses}, val_accs: {val_accs}")
    
    return(get_parameters(model))