import torch.nn as nn
import torch

import numpy as np
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))


def zip_loss(pred, y, log_output=True):
    """things to consider
    1. values are not normalised
    2. Unobserved sites are -1.
    """
    splus_zero = nn.Softplus()(pred[..., 0][y == 0])
    splus_nonzero = nn.Softplus()(pred[..., 0][y > 0])
    splus = torch.hstack([splus_zero, splus_nonzero])

    nll_zero = (
        nn.PoissonNLLLoss(full=True, reduction="none")(pred[..., 1][y == 0], y[y == 0])
        + splus_zero
    )
    nll_nonzero = (
        nn.PoissonNLLLoss(full=True, reduction="none")(pred[..., 1][y > 0], y[y > 0])
        + splus_nonzero
    )

    logpi_zero = pred[..., 0][y == 0] - splus_zero
    logpi_nonzero = pred[..., 0][y > 0] - splus_nonzero

    nll_zero = -torch.logsumexp(torch.stack((logpi_zero, -nll_zero)), 0)
    nll = torch.hstack([nll_zero, nll_nonzero])

    """
    - Usually we average the loss. Here, we are using sum so that the magnitude of loss is preserved.
    - Will add a regularization term to penalty equal predictions for pred
    """
    totalloss = torch.sum(nll)

    return totalloss


def nll_loss(pred, y):
    """
    The negative likelihood function (Bernoulli model)
    Only observed spatial-temporal coordinates are computed
    """
    mask = (y >= 0).float()
    err = nn.Softplus()(-pred) + pred - pred * y

    _sum = torch.sum(mask * err)
    return _sum


def masked_pnllloss(pred, y):
    """
    poisson loss
    Only observed spatial-temporal coordinates are computed
    """
    nllobserved = nn.PoissonNLLLoss(full=True)(pred[..., 1][y > -1], y[y > -1])

    return nllobserved


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def zMAE(pred, y, alpha=0.5, baseline=False):
    ntest = pred.shape[0]

    print(pred.shape, y.shape)

    if baseline:
        ncots_pred = pred
    else:
        ncots_pred = np.exp(pred[..., 1]) * (1 - sigmoid(pred[..., 0]))

    print(ncots_pred.shape)
    summae = 0
    zero_count = 0
    for i in range(ntest):
        ins_y = y[i, :]
        ins_yhat = ncots_pred[i, :]

        y_zero = ins_y[ins_y == 0]
        y_nonzero = ins_y[ins_y > 0]
        haty_zero = ins_yhat[ins_y == 0]
        haty_nonzero = ins_yhat[ins_y > 0]

        if len(y_zero) == 0 and len(y_nonzero) == 0:
            zero_count += 1
        else:
            zeropart = 0
            if len(y_zero) > 0:
                zeropart = alpha * np.sum(haty_zero)
                zeropart = zeropart / len(y_zero)

            nonzeropart = 0
            if len(y_nonzero) > 0:
                nonzeropart = (1 - alpha) * np.sum(abs(y_nonzero - haty_nonzero))
                nonzeropart = nonzeropart / len(y_nonzero)

            summae = summae + zeropart + nonzeropart

    return summae / (ntest - zero_count)


def zRMSE(pred, y, alpha=0.5, baseline=False):
    ntest = pred.shape[0]

    if baseline:
        ncots_pred = pred
    else:
        ncots_pred = np.exp(pred[..., 1]) * (1 - sigmoid(pred[..., 0]))
    # ncots_pred = np.exp(pred[..., 1]) * (1 - sigmoid(pred[..., 0]))

    sumrmse = 0
    zero_count = 0
    for i in range(ntest):
        ins_y = y[i, :]
        ins_yhat = ncots_pred[i, :]

        y_zero = ins_y[ins_y == 0]
        y_nonzero = ins_y[ins_y > 0]
        haty_zero = ins_yhat[ins_y == 0]
        haty_nonzero = ins_yhat[ins_y > 0]

        if len(y_zero) == 0 and len(y_nonzero) == 0:
            zero_count += 1
        else:
            zeropart = 0
            if len(y_zero) > 0:
                zeropart = alpha * np.sum(haty_zero**2)
                zeropart = zeropart / len(y_zero)

            nonzeropart = 0
            if len(y_nonzero) > 0:
                nonzeropart = (1 - alpha) * np.sum((y_nonzero - haty_nonzero) ** 2)
                nonzeropart = nonzeropart / len(y_nonzero)

            sumrmse = sumrmse + math.sqrt(zeropart + nonzeropart)

    return sumrmse / (ntest - zero_count)


if __name__ == "__main__":
    import torch
    import data.dataprocessing as dp
    from configs import data_config as dc

    data_obj = dp.GBRDataset(
        fileloc=dc.rawdata_loc,
        poissonmodel=True,
        fdim=2,
        normalize=False,
        savefeatures=True,
    )

    gbrdata = np.load(
        r"data/processed-data-pnas/feature_mat_dim_2_norm_False_20240905.npz"
    )

    import data.datautils as dutil

    datadic = dutil.get_train_test_valid(
        gbrdata, testratio=0.25, valratio=0.0, validation=False
    )
    xtrain, ytrain = datadic["trainX"], datadic["trainY"]
    xtest, ytest = datadic["testX"], datadic["testY"]

    # load model
    fdim = 2
    poissonmodel = True
    kernelwidth = 1400
    threshold = 1e-5
    model_name = "GCN"

    modelparams = dict(
        hidden_dims=[128, 64, 64, 256],
        fdim=fdim,
        poissonmodel=poissonmodel,
        nsites=xtrain.shape[1],
        kernelwidth=kernelwidth,
        threshold=threshold,
        D=data_obj.get_distmat(),
        model_name=model_name,
        device="cpu",
        site_fdim=2,
        ndays=xtrain.shape[0],
    )

    modelparams.keys()

    import model.gbrmodel as mdl

    model = mdl.GBRModel(**modelparams)

    # model = model = mdl.GBRModel(**modelparams)
    model.load_state_dict(torch.load("./artifacts/best_model.mdl"))

    # need to build the context matrix
    X_all = np.vstack([xtrain, xtest])
    X = torch.tensor(X_all[:], dtype=torch.float32)

    model.eval()
    model.temp_encoder.reset_context()

    """
    divide the data into batches
    """
    batchsize = 40
    num_batches = int(np.ceil(X.shape[0] / batchsize))

    with torch.no_grad():
        predictions = []
        for i in range(num_batches):
            X_batch = X[i * batchsize : (i + 1) * batchsize]
            Y_batch, A = model(X_batch, data_obj.Z)

            predictions.append(Y_batch.detach())

    N_test = xtest.shape[0]
    Y_pred = torch.vstack(predictions)[-N_test:]
    Y_true = torch.tensor(ytest, dtype=torch.float32)

    count = torch.sum((Y_true >= 0).float())

    zmae = zMAE(Y_pred.detach().numpy(), Y_true.detach().numpy())
    zrmse = zRMSE(Y_pred.detach().numpy(), Y_true.detach().numpy())
    print(zmae, zrmse)
