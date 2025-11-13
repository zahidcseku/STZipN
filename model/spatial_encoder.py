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

class SpatialEncoder( nn.Module ):
    """
    This module generates the static features encoding from the normalisedd site locations. For each site, we will
    crearte a h dimension encoding. For each site we have pair i.e., normalized locations and our network will generate
    a h dimension reprenestaion of the static feature.
    Input:
        - Z:2D array of shape(nsites, 2): holds the normalized site locations (longitude, latitude).
    Output:
        - static_h:2D array of shape(nsites, h): holds the static feature encoding of the site locations.  We transform
          in output to a 3D tensor to match with the GCN representations. i.e, static_h:3D array of shape(nsites, 1, h):
          holds the static feature encoding of the site locations.
    """
    def __init__(self, **kwargs):
        super(SpatialEncoder, self).__init__()
        """
            - Z:2D array of shape(nsites, 2): holds the normalized site locations (longitude, latitude).
        """
        self.device = kwargs["device"]
        self.site_fdim = kwargs["site_fdim"]
        self.hidden_size = kwargs["hidden_size"]

        """
        - Layers to embed of the site locations to features
        """
        self.Z_transform = nn.Sequential(
            nn.Linear(self.site_fdim, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, Z):
        """
        - Z:2D array of shape(nsites, 2): holds the normalized site locations (longitude, latitude).
        Embedding the site locations
        """
        Z = torch.tensor(Z, dtype=torch.float32, device=self.device)

        static_h = self.Z_transform(Z)
        """ We transform in output to a 3D tensor to match with the GCN representations. 
        i.e, static_h:3D array of shape(nsites, 1, h): holds the static feature encoding of the site locations. """
        static_h = static_h.view(Z.shape[0], 1, -1)

        return static_h


if __name__ == "__main__":
    # load the normalized location vectors
    zfile = np.load(f"{dc.data_output_dir}/{rc.normalized_locs}")
    Z = zfile["static_features"]

    print(f"Shape of Z: {Z.shape}") #should be -> nsites X 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_encoder = SpatialEncoder(site_fdim=Z.shape[-1], ndays=982, hidden_size=mc.hidden_dim, device=device)

    print(spatial_encoder)

    spfeat_emb = spatial_encoder(Z)
    print(f"Spacial feature embedding shape: {spfeat_emb.shape}") #should be -> nsites x 1 x hiddensize