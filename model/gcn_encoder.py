import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import configs.data_config as dc
import configs.model_config as mc
import configs.run_config as rc
from data.dataprocessing import GBRDataset
from model.temporal_encoder import TemporalEncoder
from model.spatial_encoder import SpatialEncoder
from model.input_encoder import InputEncoder


class GCNEncoder(nn.Module):
    def __init__(self, **kwargs):
        """
            - A:2D array of shape(nsites, nsites): the adjacency matrix.
        """
        super(GCNEncoder, self).__init__()

        self.D = kwargs["D"]
        self.hidden_size = kwargs["hidden_size"]
        self.kernelwidth = kwargs["kernelwidth"]
        self.threshold = kwargs["threshold"]
        self.device = kwargs["device"]

        """Computation of tilda A(normalized graph adjacency matrix): graph convolution operations"""
        A = np.exp(-self.D ** 2 / self.kernelwidth ** 2)
        A = A * (A > self.threshold)

        degree = A.sum(0)
        self.D = 1 / np.sqrt(degree)
        normalizedA = self.D * A * self.D[:, None]
        self.denseA = torch.tensor(normalizedA, dtype=torch.float32, device=self.device)

        """
            - GCN Layer
        """
        self.gcn_layer1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, inp_emb, staticfeat_emb):
        """
        inp_emb shape: nsites x num days to predict x hidden_size
        staticfeat_emb shape: nsites x 1 x hidden_size
        """

        inp_emb = inp_emb.reshape(inp_emb.shape[0], -1)

        # graph convolution
        h = torch.mm(self.denseA, inp_emb)
        h = h.reshape(self.denseA.shape[0], -1, self.hidden_size)

        gcn_emb = self.gcn_layer1(h)

        # reshape staticfeat_emb to match with gcn_emb
        gcn_emb = gcn_emb + staticfeat_emb.reshape(gcn_emb.shape[0], 1, -1)

        return gcn_emb, self.denseA


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

    print('GCN input shapes..')
    print(f'Z shape - {spfeat_emb.shape}')
    print(f'temp shape - {temporal_emb.shape}')

    gcn_emb = gcn_encoder(temporal_emb, spfeat_emb)

    print(f"GCN embedding shape: {gcn_emb.shape}")
