import numpy as np


def split_train_test_data(X, testratio, split_space=False, split_from_tail=True):
    """
                  X -- input data matrix of shape num_days x num_sites
          testratio -- ratio of testing dataset (e.g. 0.2)
    split_from_tail -- Use future data for test
       split_space -- will split like
                     -- train: 0, 1, 3, 4, 6, 7, ...
                     -- test: 2, 5
    """
    train_ids = np.ones(X.shape[0], dtype=bool)

    test_N = int(X.shape[0] * testratio)

    if not split_space:
        test_N = max(test_N, 1)
        train_N = X.shape[0] - test_N

        if split_from_tail:
            train_ids[train_N:] = 0
        else:
            train_ids[:test_N] = 0
    else:
        space_step = int(X.shape[0] / test_N)

        train_ids[space_step - 1::space_step, :] = 0

    return train_ids


def get_train_test_valid(gbrdata, testratio, valratio, validation, split_space=False, split_from_tail=True):
    #print(gbrdata.files)
    X, Y = gbrdata["X"], gbrdata["Y"]
    trainids = split_train_test_data(X, testratio, split_space=split_space, split_from_tail=split_from_tail)

    trainX, trainY = X[trainids], Y[trainids]
    testX, testY = X[~trainids], Y[~trainids]

    validX, validY = None, None

    if validation:
        trainids = split_train_test_data(trainX, valratio, split_space=split_space, split_from_tail=split_from_tail)
        validX, validY = trainX[~trainids], trainY[~trainids]
        trainX, trainY = trainX[trainids], trainY[trainids]

    return dict(trainX=trainX, trainY=trainY, testX=testX, testY=testY, validX=validX, validY=validY)