import torch
import torch.nn as nn

def absolute_error( pred, y, poissonmodel=True ):
    '''
    absolute error
    '''
    mask = ( y >= 0 ).float()

    if poissonmodel:
        '''
        get the predicted number of cots
        '''
        ncots_pred = torch.exp(pred[..., 1]) * (1 - nn.Sigmoid()(pred[..., 0]))

        _sum = torch.sum(torch.abs(ncots_pred - y) * mask)
    else:
        _sum = torch.sum(torch.abs(nn.Sigmoid()(pred) - y) * mask)

    return _sum