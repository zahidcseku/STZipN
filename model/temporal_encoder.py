import torch
import torch.nn as nn
import numpy as np


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import configs.data_config as dc
import configs.model_config as mc
import configs.run_config as rc

from data.dataprocessing import GBRDataset
from model.input_encoder import InputEncoder


class TemporalEncoder(nn.Module):
    """
    # on the temporal dimension, the model must be auto-regressive (no bi-directional!)
    Parameter: nsites is required to clear the context history
    """

    def __init__(self, **kwargs):
        super(TemporalEncoder, self).__init__()

        self.temporalunit = kwargs["temporalunit"]
        self.hidden_size = kwargs["hidden_size"]
        self.device = kwargs["device"]
        self.nsites = kwargs["nsites"]

        if self.temporalunit == "lstm":
            self.temporal_layer = nn.LSTM(self.hidden_size,
                                          self.hidden_size,
                                          batch_first=True,
                                          num_layers=1
                                          )
        elif self.temporalunit == "gru":
            self.temporal_layer = nn.GRU(self.hidden_size,
                                         self.hidden_size,
                                         batch_first=True,
                                         num_layers=1
                                         )
        else:
            print("Unknown type for temporal unit!!!")
            exit(0)

        self.reset_context()

    def reset_context(self):
        '''
        reset the context/memory vectors
        '''
        self.context0 = torch.zeros(1, self.nsites, self.hidden_size, device=self.device)

        if self.temporalunit == "lstm":
            self.cell0 = torch.zeros(1, self.nsites, self.hidden_size, device=self.device)

    def forward(self, h):
        """
            h: input embedding of shape: (num_days, num_sites, hidden_size)
            temporal_emb: (num_days, num_sites, hidden_size)
        """
        # encoder_output:
        # num_sites x num_days x hidden_size
        if self.temporalunit == "lstm":
            temporal_emb, (context0, cell0) = self.temporal_layer(h, (self.context0, self.cell0))
            # context:
            # 1 x num_sites x hidden_size - memorize the previous batches

            self.context0 = context0.detach()
            self.cell0 = cell0.detach()
        else:
            temporal_emb, context0 = self.temporal_layer(h, self.context0)

            self.context0 = context0.detach()

        return temporal_emb


if __name__ == "__main__":

    # run parameters
    # fdim specifies the dimension of the features required by the input layer
    fdim = rc.fdim
    # hidden_size is required by the InputEncoder class. If defines the number of units in the layers.
    hidden_size = mc.hidden_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # load the feature matrix
    features = np.load(f"{dc.data_output_dir}/{rc.feature_file}")
    X = features['X']
    Y = features['Y']

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    input_encoder = InputEncoder(hidden_size=hidden_size, fdim=fdim)
    print(input_encoder)

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    input_emb = input_encoder(X)
    nsites = X.shape[1]
    temporal_encoder = TemporalEncoder(temporalunit=mc.temporalunit,
                                       hidden_size=32,
                                       nsites=nsites,
                                       device=device
                                       )
    print(temporal_encoder)

    temporal_emb = temporal_encoder(input_emb)
    print(f"Temporal embedding shape: {temporal_emb.shape}")

    for name, params in temporal_encoder.named_parameters():
        print(f"{name} - dim {params.shape}")
