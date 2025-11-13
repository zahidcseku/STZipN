import model.loss_fns as lfn
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
import torch.nn as nn


class Trainer:
    def __init__(self, model, modelname, Z, optimizer, traindata, testdata, batchsize) -> None:
        self.optimizer = optimizer
        self.model = model
        self.model_name = modelname
        self.train_data = traindata
        self.valid_data = testdata
        #self.savemodel = savemodel
        self.batchsize = batchsize
        self.device = "cpu"
        self.Z = Z

    def get_model(self):
        return self.model
    
    def train_epoch(self):
        """
            Training logic for an epoch
            :param epoch: Integer, current training epoch.
            :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        
        # reset temporal context vector
        if self.model_name in ["temp-LSTM", "temp-GRU", "GCN"]:
            self.model.temp_encoder.reset_context()

        """
        divide the data into batches
        """
        X = torch.tensor(self.train_data[0], dtype=torch.float32, device=self.device, requires_grad=True)
        Y = torch.tensor(self.train_data[1], dtype=torch.float32, device=self.device, requires_grad=True)

        num_batches = int(np.ceil(X.shape[0] / self.batchsize))
        start_idx = 0

        predictions = []
        """------------training loop----------"""
        for i in range(num_batches):
            X_batch = X[start_idx + i * self.batchsize: start_idx + (i + 1) * self.batchsize]
            Y_batch = Y[start_idx + i * self.batchsize: start_idx + (i + 1) * self.batchsize]

            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

            self.optimizer.zero_grad()

            Y_pred, A = self.model(X_batch, self.Z) if self.model_name in ["SP", "GCN"] else self.model(X_batch)

            loss = lfn.zip_loss(Y_pred, Y_batch) / self.batchsize

            loss.backward()
            self.optimizer.step()

            predictions.append(Y_pred)

        #print(f"y pred shape: {Y_pred.shape}")
        Y_pred = torch.vstack(predictions)
        Y_true = Y[start_idx:]
        count = torch.sum((Y_true >= 0).float())
        loss = lfn.zip_loss(Y_pred, Y_true) / count

        mae = self.zip_mae(Y_true, Y_pred)
        #lr_scheduler.step()

        return {"loss": loss.item(), "mae": mae}
    

    def valid_epoch(self):
        """
            Validate after training an epoch
            :param epoch: Integer, current training epoch.
            :return: A log that contains information about validation
        """

        # need to build the context matrix
        X_all = np.vstack([self.train_data[0], self.valid_data[0]])
        X = torch.tensor(X_all[:], dtype=torch.float32, device=self.device)

        self.model.eval()
        if self.model_name in ["temp-LSTM", "temp-GRU", "GCN"]:
            self.model.temp_encoder.reset_context()

        """
            divide the data into batches
        """
        num_batches = int(np.ceil(X.shape[0] / self.batchsize))
        
        with torch.no_grad():
            predictions = []
            for i in range(num_batches):
                X_batch = X[i * self.batchsize: (i + 1) * self.batchsize]
                Y_batch, A = self.model(X_batch, self.Z) if self.model_name in ["SP", "GCN"] else self.model(X_batch)

                predictions.append(Y_batch.detach())

            """save adjacency matrix"""
            #np.savez("adjacenymat.npz", adj=A.detach().to('cpu').numpy())

        N_valid = self.valid_data[0].shape[0]
        Y_pred = torch.vstack(predictions)[-N_valid:]
        Y_true = torch.tensor(self.valid_data[1], dtype=torch.float32, device=self.device)

        count = torch.sum((Y_true >= 0).float())
        loss = lfn.zip_loss(Y_pred, Y_true) / count

        mae = self.zip_mae(Y_true, Y_pred)
    
        return {"vloss": loss.item(), "vmae": mae}
    
    
    def zip_mae(self, ytrue, ypreds):
        gt = ytrue[ytrue > 0].detach().numpy()
        preds = torch.exp(ypreds[ytrue > 0][..., 1]) * (1 - nn.Sigmoid()(ypreds[ytrue > 0][..., 0]))
        preds = preds.detach().numpy()

        #print(gt.shape, preds.shape)
        #print(gt.max(), gt.min())
        #print(preds.max(), preds.min())

        return mean_absolute_error(preds, gt)
    
    
    def absolute_error(self, pred, y, poissonmodel=True ):
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


