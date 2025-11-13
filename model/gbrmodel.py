import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataprocessing import GBRDataset

import configs.data_config as dc
import configs.model_config as mc
import configs.run_config as rc

from model.temporal_encoder import TemporalEncoder
from model.spatial_encoder import SpatialEncoder
from model.input_encoder import InputEncoder
from model.gcn_encoder import GCNEncoder
from model.output_encoder import OutputEncoder


class GBRModel(nn.Module):
    def __init__(self, **kwargs):
        #super(GBRModel, self).__init__()
        super().__init__()

        self.model_name = kwargs["model_name"]      # {trivial, temp-LSTM, temp-GRU, SP, GCN}
        self.hidden_dims = kwargs["hidden_dims"]    # a list [32, 16, 32, 16]
        self.device = kwargs["device"]              # {CPU, GPU}
        self.poissonmodel = kwargs["poissonmodel"]  # {true, false} determines the loss_func and output_dim
        self.hidden_size = self.hidden_dims[0]      # the hidden dims of the MLPs in al submodules are fixed to the
                                                    # first hidden_dim
        self.nsites = kwargs["nsites"]


        """trivial model only with the input encoding"""
        if self.model_name in ["trivial", "temp-GRU", "temp-LSTM", "GCN"]:
            # input feature dim currently {1, 2}
            self.inp_encoder = InputEncoder(hidden_size=self.hidden_size, fdim=kwargs["fdim"])
        if self.model_name == "temp-LSTM":
            self.temp_encoder = TemporalEncoder(nsites=self.nsites,
                                                hidden_size=self.hidden_size,
                                                temporalunit="lstm",
                                                device=self.device
                                                )
        if self.model_name == "temp-GRU" or self.model_name == "GCN":
            self.temp_encoder = TemporalEncoder(nsites=self.nsites,
                                                hidden_size=self.hidden_size,
                                                temporalunit="gru",
                                                device=self.device
                                                )
        if self.model_name == "SP" or self.model_name == "GCN":
            self.site_fdim = kwargs["site_fdim"]
            self.ndays = kwargs["ndays"]

            if self.model_name == "SP":
                self.layers = []

            self.sp_encoder = SpatialEncoder(site_fdim=self.site_fdim,
                                             ndays=self.ndays,
                                             hidden_size=self.hidden_size,
                                             device=self.device
                                             )
        if self.model_name == "GCN":
            self.kernelwidth = kwargs["kernelwidth"]
            self.threshold = kwargs["threshold"]
            self.D = kwargs["D"]
            #print(kwargs.keys())

            self.gc_encoder = GCNEncoder(
                                         kernelwidth=self.kernelwidth,
                                         threshold=self.threshold,
                                         hidden_size=self.hidden_size,
                                         device=self.device,
                                         D=self.D
                                        )
        """add the output layer"""
        self.output_encoder = OutputEncoder(hidden_dims=self.hidden_dims, poissonmodel=self.poissonmodel)

    def forward(self, cots, Z=None):
        if self.model_name in ["trivial", "temp-GRU", "temp-LSTM", "GCN"]:
            h = self.inp_encoder(cots)
        if self.model_name in ["temp-LSTM", "temp-GRU", "GCN"]:
            h = self.temp_encoder(h)
        if self.model_name == "SP" or self.model_name == "GCN":
            static_h = self.sp_encoder(Z)

            if self.model_name == "GCN":
                h, A = self.gc_encoder(h, static_h)
            else:
                h = static_h.repeat(1, cots.shape[0], 1) # cots.shape[0]

        output_emb = self.output_encoder(h)

        if self.model_name == "GCN":
            return output_emb, A

        return output_emb


    def get_modelname(self):
        return self.model_name

    def init_weights(self, m):
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-.01, .01)
            m.bias.data.fill_(0)

            #print("init")
            #torch.nn.init.xavier_uniform_(m.weight)
            #m.bias.data.fill_(0.01)



if __name__ == "__main__":
    fdim = 2
    poissonmodel = True
    kernelwidth = 1400
    threshold = 1e-5

    """loading the dataset"""
    data_obj = GBRDataset(fileloc="data/rawdata/gbr_cots_culldata.xlsx",
                          poissonmodel=True,
                          fdim=fdim,
                          simulate_data="",
                          normalize=False,
                          savefeatures=False
                          )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """get the temporal feature embeddings"""
    X = data_obj[...]['x']
    Y = data_obj[...]['y']

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    """-----------------loading the dataset---------------"""
    X = torch.tensor(X.reshape(-1, X.shape[1], X.shape[2]),
                     dtype=torch.float32,
                     requires_grad=True,
                     device=device
                     )

    model_names = "trivial temp-LSTM temp-GRU SP GCN".split()
    modelparams = dict(hidden_dims=[4, 4],
                       fdim=fdim,
                       device=device,
                       poissonmodel=poissonmodel,
                       nsites=X.shape[1]
                       )

    model_names = ["GCN"]
    for modelname in model_names:
        print(f"------{modelname}-------")
        modelparams["model_name"] = modelname

        if modelname in ["SP", "GCN"]:
            modelparams["site_fdim"] = 2
            modelparams["ndays"] = X.shape[0]

        if modelname == "GCN":
            #print("hello")
            modelparams["kernelwidth"] = kernelwidth
            modelparams["threshold"] = threshold
            modelparams["D"] = data_obj.get_distmat()

        print(modelparams)
        model = GBRModel(**modelparams)

        print(model)
        #exit()

        if modelname == "GCN":
            output, A = model(X, data_obj.Z)
        else:
            output = model(X, data_obj.Z)
        print(f"Output shape of the {modelname} model: {output.shape}")
        print(f"Output shape of the {modelname} model: {A.shape}")
        print(f"{output[...,:]}")
        """
            Shape of input_emb: nsites x ndays x hidden_size
            Shape of temp_emb: nsites x ndays x hidden_size
            Shape of spacial_emb: nsites x ndays x hidden_size
            Shape of gcn_emb: nsites x ndays x hidden_size
            Shape of output_emb: 
            - ndays x nsites x 2
        """

        print(f"---------------------\n")