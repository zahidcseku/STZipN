import torch
import sys
import argparse
import numpy as np
from parse_config import ConfigParser
import collections
from datetime import datetime
import json
from utils import read_json, write_json

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.service.ax_client import AxClient
from train import train_model

torch.manual_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_evaluate(parameterization, config):
    """
    Setup model parameters from parameterization and config file.
    Model parameters depend on the model type:
        - traditional: fdim: feature dimension
                       hidden_size: no of units of the first hidden layer
                       hidden_dims: list of units in the output hidden layers
                       poissonmodel: determines no of outputs

        - SP: site_fdim: dimension of site features
              ndays: no of days in the dataset
              hidden_dims: list of units in the output hidden layers
              poissonmodel: determines no of outputs

        - temp-LSTM: fdim: feature dimension
                     hidden_size: no of units of the first hidden layer
                     hidden_dims: list of units in the output hidden layers
                     poissonmodel: determines no of outputs
                     nsites: number sites in the dataset
                     device: gpu or cpu

        - temp-GRU: fdim: feature dimension
                    hidden_size: no of units of the first hidden layer
                    hidden_dims: list of units in the output hidden layers
                    poissonmodel: determines no of outputs
                    nsites: number sites in the dataset
                    device: gpu or cpu

        - GCN: fdim: feature dimension
               hidden_size: no of units of the first hidden layer
               site_fdim: dimension of site features
               ndays: no of days in the dataset
               hidden_dims: list of units in the output hidden layers
               poissonmodel: determines no of outputs
               device: gpu or cpu
    """
    hidden_dims = []
    for i in range(parameterization.get("hidden_n_layers")):
        hidden_dims.append(parameterization.get("nunits"))

    params = [dict(target="arch;args;hidden_dims", value=hidden_dims),
              dict(target="optimizer;args;lr", value=parameterization.get("lr")),
              dict(target="trainer;batchsize", value=parameterization.get("backward"))
              ]
    if model_name == "GCN":
        params.append(dict(target="arch;args;kernelwidth",
                           value=parameterization.get("kernelwidth"))
                      )

    if not config["trainer"]["validation"]:
        params.append(dict(target="trainer;save_period", value=5))
        print(f"validation params : {parameterization}")

    config.update_values(params)
    logs = train_model(config)

    if config["trainer"]["validation"]:
        return logs["val_absolute_error"], logs["epoch"]


if __name__ == '__main__':
    """
    Setup config file
    """
    ax_client = AxClient()
    ntrials = 50
    model_name = "GCN"
    fdim = 2

    lst_units = [16, 32, 64, 128, 200, 256]
    mdl = ""
    if model_name == "GCN":
        mdl = "inp-GRU-gcn-GRU-adaptive"
        lh = 6
    elif model_name == "SP":
        mdl = "sp"
        lh = 5
        lst_units = [16, 32, 64]
    elif model_name == "temp-LSTM":
        mdl = "inp-temp-LSTM"
        lh = 5
    elif model_name == "temp-GRU":
        mdl = "inp-temp-GRU"
        lh = 6
    elif model_name == "trivial":
        mdl = "inp-"
        lh = 5
        lst_units = [16, 32]
    else:
        print("Warning!!!!Unknown model")
        exit(0)

    experimentname = f"F-{model_name}-fdim-{fdim}-{mdl}-out"

    #print(f'model name - {model_name} - {lst_units}')
    #exit(0)

    parameters = [
        {
            "name": "kernelwidth",
            "type": "choice",
            "values": [800, 900, 950, 1000, 1050, 1100, 1200, 1300],
            "value_type": "int"
        },
        {
            "name": "hidden_n_layers",
            "type": "range",
            "bounds": [1, lh],
            "value_type": "int"
        },
        {
            "name": "nunits",
            "type": "choice",
            "values": lst_units,
            "value_type": "int"
        },
        {
            "name": "backward",
            "type": "choice",
            "values": [15, 30, 40, 50, 60, 70],
            "value_type": "int"
        },
        {
            "name": "lr",
            "type": "range",
            "bounds": [0.0001, 0.01], #GRU 0.00004944
            "log_scale": True,
            "value_type": "float"
        }
    ]

    "Common params for dataloader and model"
    args = argparse.ArgumentParser(description='GBR COTS estimation')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')

    """
    Initialize the config file
    """
    with open(f"config-{model_name}.json", 'r+') as file:
        file_data = json.load(file)
        file_data["name"] = experimentname
        file_data["common_params"]["model_name"] = model_name
        file_data["common_params"]["fdim"] = fdim
        file_data["trainer"]["validation"] = True
        file_data["trainer"]["early_stop"] = 40
        file_data["trainer"]["epochs"] = 300
        file.seek(0)
        file.truncate()
        json.dump(file_data, file, indent=4)

    config = ConfigParser.from_args(args)

    ax_client.create_experiment(
                                name=experimentname,
                                parameters=parameters,
                                objective_name='val_error',
                                minimize=True
                                )

    for _ in range(ntrials):
        parameters, trial_index = ax_client.get_next_trial()
        val_err, eps = train_evaluate(parameters, config)
        print(val_err, eps)
        ax_client.complete_trial(trial_index=trial_index, raw_data=val_err)
        ax_client.update_trial_data(trial_index, {"eps": eps})

    best_parameters, metrics = ax_client.get_best_parameters()

    hidden_dims = [best_parameters.get("nunits")] * best_parameters.get("hidden_n_layers")
    # print(ax_client.get_best_trial()[2][0]["eps"])
    # searchdf = ax_client.generation_strategy.trials_as_df

    # save search logs
    ts = datetime.now().strftime("%d%m%Y-%H%M")
    ax_client.save_to_json_file(filepath=f"paramsearch_logs/{experimentname}-{ts}.json")

    # train model with best params and save for future
    with open(f"config-{model_name}.json", 'r+') as file:
        file_data = json.load(file)
        file_data["trainer"]["validation"] = False
        file_data["trainer"]["epochs"] = int(ax_client.get_best_trial()[2][0]["eps"])
        file_data["optimizer"]["args"]["lr"] = best_parameters.get("lr")
        file_data["arch"]["args"]["hidden_dims"] = hidden_dims
        file_data["arch"]["args"]["kernelwidth"] = best_parameters.get("kernelwidth")
        file_data["trainer"]["batchsize"] = best_parameters.get("backward")
        file.seek(0)
        file.truncate()
        json.dump(file_data, file, indent=4)

    """reload config"""
    config = ConfigParser.from_args(args)

    print(f"Running {model_name} training with best paramters")
    print(f'Batchsize: {config["trainer"]["batchsize"]}')
    print(f'fdim: {config["common_params"]["fdim"]}')
    print(f'validation: {config["trainer"]["validation"]}')
    print(f'hidden: {config["arch"]["args"]["hidden_dims"]}')
    print(f'epochs: {config["trainer"]["epochs"]}')

    train_evaluate(best_parameters, config)

    print("training completed successfully...")