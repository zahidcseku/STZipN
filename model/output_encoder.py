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


class OutputEncoder( nn.Module ):
    def __init__(self, **kwargs):
        super(OutputEncoder, self).__init__()

        hidden_dims = kwargs["hidden_dims"]
        poissonmodel = kwargs["poissonmodel"]

        _dims = hidden_dims[:]
        if poissonmodel:
            _dims.append(2)
        else:
            _dims.append(1)

        output_layers = []
        for h1, h2 in zip(_dims, _dims[1:]):
            output_layers.append(nn.ELU())
            output_layers.append(nn.Linear(h1, h2))

        self.output_layers = nn.Sequential(*output_layers)

    def forward(self, h):
        # output layers (MLP)
        h = self.output_layers(h)

        # comment on transpoose and squeeze
        return torch.transpose(h.squeeze(-1), 0, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fdim = rc.fdim
    # hidden_size is required by the InputEncoder class. If defines the number of units in the layers.
    hidden_size = mc.hidden_dim

    # load the feature matrix
    features = np.load(f"{dc.data_output_dir}/{rc.feature_file}")
    X = features['X']
    Y = features['Y']

    zfile = np.load(f"{dc.data_output_dir}/{rc.normalized_locs}")
    Z = zfile["static_features"]

    input_encoder = InputEncoder(hidden_size=hidden_size, fdim=fdim)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    input_emb = input_encoder(X)

    nsites = X.shape[1]
    temporal_encoder = TemporalEncoder(temporalunit="lstm",
                                       hidden_size=hidden_size,
                                       nsites=nsites,
                                       device=device
                                       )
    temporal_emb = temporal_encoder(input_emb)
    print(f"Temporal embedding shape: {temporal_emb.shape}")

    """GCN embeddings"""
    distFile = np.load(f"{dc.data_output_dir}/{rc.distance_matrix}")
    D = distFile["dist"]
    gcn_encoder = GCNEncoder(D=D, kernelwidth=1400, threshold=1e-5, hidden_size=hidden_size, device=device)

    """get the spacial feature embeddings"""
    zfile = np.load(f"{dc.data_output_dir}/{rc.normalized_locs}")
    Z = zfile["static_features"]

    spatial_encoder = SpatialEncoder(site_fdim=Z.shape[-1], hidden_size=hidden_size, device=device)
    spfeat_emb = spatial_encoder(Z)

    gcn_emb = gcn_encoder(temporal_emb, spfeat_emb)

    """output encoding"""
    output_encoder = OutputEncoder(hidden_dims=[hidden_size, 32, 32], poissonmodel=True)

    print(output_encoder)

    output = output_encoder(gcn_emb)

    print(f"Output shape: {output.shape}")

    for name, params in output_encoder.named_parameters():
        print(f"{name} - dim {params.shape}")

    print(output)