import numpy as np
import warnings

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from data.dataprocessing import GBRDataset

from loss_fns import zMAE, zRMSE

class BaselineModels:
    def __init__(self, **kwargs):
        self.X_train = kwargs["X_train"]
        self.X_test = kwargs["X_test"]
        self.fdim = kwargs["fdim"]
        self.verbose = kwargs["varbose"]

        if self.fdim > 1:
            self.X_train = self.X_train[..., 0]
            self.X_test = self.X_test[..., 0]
        #print(self.X_test.shape)
        #print(self.X_train.shape)


    def model_global_average(self):
        '''
        (1st baseline) predict COTS population using global average
        '''

        if self.verbose: print('computing baseline using global average...')

        # average of non-negative entries (all valid entries)
        averageX = np.nanmean(np.where(self.X_train >= 0, self.X_train, np.nan))
        if self.verbose: print(f'    average normalized density: {averageX:.3f}')

        # compute absolute error in all valid entries
        X_test_nan = np.where(self.X_test >= 0, self.X_test, np.nan)
        err = np.nanmean(np.abs(X_test_nan - averageX))

        if self.verbose: print(f'    absolute error: {err:.3f}')

        average_preds = np.zeros_like(X_test_nan)
        average_preds[...] = averageX

        #print(average_preds.shape, X_test_nan.shape)
        zmae = zMAE(average_preds, X_test_nan, baseline=True)
        zrmse = zRMSE(average_preds, X_test_nan, baseline=True)

        return err, zmae, zrmse


    def model_site_average(self):
        '''
        (2nd baseline) predict based on site average population
        '''
        if self.verbose: print('computing baseline using site average...')

        # avearge of each site (ignore devision by 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_siteX = np.nanmean(np.where(self.X_train >= 0, self.X_train, np.nan), axis=0)

        # replace nan with 0
        average_siteX = np.nan_to_num(average_siteX)

        if self.verbose:
            print(f'minimum normalized density: {min(average_siteX):.5f}')
            print(f'maximum normalized density: {max(average_siteX):.3f}')

        X_test_nan = np.where(self.X_test >= 0, self.X_test, np.nan)

        err = np.nanmean(np.abs(X_test_nan - average_siteX))

        if self.verbose: print(f'absolute error: {err:.3f}')

        #saverage_preds = np.zeros_like(X_test_nan)
        #np.repeat(arr[np.newaxis, :], 3, axis=0)
        saverage_preds = np.repeat(average_siteX[np.newaxis, :], X_test_nan.shape[0], axis=0)

        zmae = zMAE(saverage_preds, X_test_nan, baseline=True)
        zrmse = zRMSE(saverage_preds, X_test_nan, baseline=True)

        return err, zmae, zrmse


    def model_last_seen(self):
        '''
        (3rd baseline) Predict site density based on last observed value
        This is an online model.
        '''

        if self.verbose: print('computing baseline using most recent observation ...')

        X_train_nan = np.where(self.X_train >= 0, self.X_train, np.nan)
        ## do we take avg?
        averageX = np.nanmean(X_train_nan)

        # this is the most recently observed density value of each site
        last = np.ones(self.X_train.shape[1]) * averageX

        for x in X_train_nan:
            last = np.where(np.isnan(x), last, x)

        # testing
        X_test_nan = np.where(self.X_test >= 0, self.X_test, np.nan)

        err = []
        zmae = []
        zrmse = []
        for x in X_test_nan:
            err.append(np.abs(x - last))
            zmae.append(zMAE(last[None, :], x[None, :], baseline=True))
            zrmse.append(zRMSE(last[None, :], x[None, :], baseline=True))
            last = np.where(np.isnan(x), last, x)
            #preds.append(last)

        err = np.array(err)
        err = np.nanmean(err)

        if self.verbose: print(f'absolute error: {err:.3f}')

        #preds = np.array(preds)
        #preds_nan = np.where(preds >= 0, preds, np.nan)
        #print(preds[0, :])
        #print(preds.shape, X_test_nan.shape)
        #zmae = zMAE(preds, X_test_nan, baseline=True)
        #zrmse = zRMSE(preds, X_test_nan, baseline=True)
        zmae = np.array(zmae)
        zmae = np.nanmean(zmae)
        zrmse = np.array(zrmse)
        zrmse = np.nanmean(zrmse)


        return err, zmae, zrmse


if __name__ == "__main__":
    data_obj = GBRDataset(fileloc=r"./data/rawdata/gbr_cots_culldata.xlsx",
                          poissonmodel=True,
                          fdim=2,                          
                          normalize=False,
                          savefeatures=False
                          )

    
    #datadic = dutil.get_train_test_valid(gbrdata, testratio=.25, valratio=0., validation=False )
    #xtrain, ytrain = datadic['trainX'], datadic['trainY']
    #xtest, ytest = datadic['testX'], datadic['testY']

    """get the temporal feature embeddings"""
    X = data_obj[...]['x']
    Y = data_obj[...]['y']

    testsize = 200

    trainX = X[:-testsize]
    testX = X[-testsize:]

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"train X shape: {trainX.shape}")
    print(f"test X shape: {testX.shape}")

    baselines = BaselineModels(X_train=trainX, X_test=testX, fdim=2, varbose=True)

    gmae, gzmae, gzrmse = baselines.model_global_average()
    sav_mae, sav_zmae, sav_zrmse = baselines.model_site_average()
    lseen_mae, lseen_zmae, lseen_zrmse = baselines.model_last_seen()

    print(f"Global average MAE: {gmae:.2f}, zMAE: {gzmae:.2f}, zRMSE: {gzrmse:.2f}")
    print(f"Site average MAE: {sav_mae:.2f}, zMAE: {sav_zmae:.2f}, zRMSE: {sav_zrmse:.2f}")
    print(f"Last seen average MAE: {lseen_mae:.2f}, zMAE: {lseen_zmae:.2f}, zRMSE: {lseen_zrmse:.2f}")