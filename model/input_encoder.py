import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#print(sys.path)
#exit()

import configs.data_config as dc
import configs.model_config as mc
import configs.run_config as rc

from data.dataprocessing import GBRDataset

"""
This scirpt corresponds to the input_encoder of the STZipN model. The input_encoder is a one-hidden layer network for 
input representation learning. It takes the feature_matrix as input and generates a higer dimensional represnetation of
the inputs. The output of the input_encoder is feed into the temporal or spatial encoder based on the desired 
architecture.
"""
class InputEncoder(nn.Module):
    def __init__(self, **kwargs):
        """

        :param kwargs: should include the input dim - should be the fdim and the number of hidden units.
        """
        super(InputEncoder, self).__init__()

        self.hidden_size = kwargs["hidden_size"]
        self.fdim = kwargs["fdim"]

        # input layers:
        # a one-hidden layer network for input representation learning
        # to feed into the RNN

        self.input_layers = nn.Sequential(
            nn.Linear(self.fdim, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
        )

    def forward(self, cots):
        """
            cots:
                - 2D tensor of shape (num_days, num_sites) or
                - 3D tensor of shape (num_days, num_sites, 2) - 2 refers to (ncots, days since last visit)
            output: encoding of the COTS
                - shape: num_sites, num_days, hidden_size
        """

        # We tranpose to learn feautres per site in the next sub network
        h = torch.transpose(cots, 0, 1).view(cots.shape[1], -1, self.fdim)
        # h: num_sites x num_days x 1

        input_emb = self.input_layers(h)

        return input_emb


if __name__ == "__main__":
    # run parameters
    # fdim specifies the dimension of the features required by the input layer
    fdim = rc.fdim
    # hidden_size is required by the InputEncoder class. If defines the number of units in the layers.
    hidden_size = mc.hidden_dim

    # load the feature matrix
    features = np.load(f"{dc.data_output_dir}/{rc.feature_file}")
    X = features['X']
    Y = features['Y']

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    input_encoder = InputEncoder(hidden_size=hidden_size, fdim=fdim)
    print(input_encoder)

    X = torch.tensor(X.reshape(-1, X.shape[1], X.shape[2]),
                     dtype=torch.float32,
                     requires_grad=True)
    input_emb = input_encoder(X)
    print(f"input embedding shape: {input_emb.shape}")

    for name, params in input_encoder.named_parameters():
        print(f"{name} - dim {params.shape}")

    print(X[3])
    print(input_emb[3])

    """
    data_obj = GBRDataset(fileloc="/data/rawdata/Cull Data 211011.xlsx",
                          poissonmodel=True,
                          fdim=fdim,
                          simulate_data=""
                          )
    """