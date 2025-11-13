#!/usr/bin/env python
from pathlib import Path
import joblib
import numpy as np
import torch

import time, datetime
import argparse
from datetime import datetime

def runmodel(D,
             threshold,
             Z,
             X,
             Y,
             trainids,
             lrate,
             epochs,
             hidden,
             backward,
             forward,
             temporalonly,
             spacialonly,
             logfile,
             poissonmodel,
             kernelwidth=1400,
             temporalunit=None,
             savemodel=False,
             verbose=False):
    A = np.exp(-D ** 2 / kernelwidth ** 2)
    A = A * (A > threshold)

    print(X.shape)
    print(trainids.shape)
    X_train = X[trainids, :, :].reshape(-1, X.shape[1], X.shape[2])
    Y_train = Y[trainids, :].reshape(-1, Y.shape[1])

    X_test = X[~trainids, :, :].reshape(-1, X.shape[1], X.shape[2])
    Y_test = Y[~trainids, :].reshape(-1, Y.shape[1])

    # running baselines
    '''
    baseline1 = model_global_average(X_train, X_test, verbose=False)
    baseline2 = model_site_average(X_train, X_test, verbose=False)
    baseline3 = model_last_seen(X_train, X_test, verbose=False)

    print(f"X train shape: {X_train.shape} --- X test shape {X_test.shape}")
    '''

    model, lcurve = train(X,
                          Y,
                          trainids,
                          A,
                          Z,
                          lrate,
                          epochs,
                          hidden,
                          backward,
                          temporalonly,
                          spacialonly,
                          poissonmodel=poissonmodel,
                          logfile=logfile,
                          temporalunit=temporalunit,
                          savetrainlog=savemodel
                          )

    loss, err, prediction, prediction_all = test(X, Y, trainids, model, backward, poissonmodel=poissonmodel)

    '''---- write logs---'''
    if savemodel:
        filename = f'{logfile}-model-trainX-testX-trainY-test_Y-lcurve-loss-err-prediction.npz'
        np.savez(filename,
                 trainX=X_train,
                 trainY=Y_train,
                 testX=X_test,
                 testY=Y_test,
                 lcurve=lcurve,
                 loss=loss,
                 err=err,
                 prediction=prediction,
                 predictionall=prediction_all
                 )

    '''Write baseline performance'''
    if verbose:
        with open(f"{logfile}-performance.txt", "w", encoding='utf-8') as f:
            '''
            f.write(f"Baseline - global average: {baseline1}\n")
            f.write(f"Baseline - site average: {baseline2}\n")
            f.write(f"Baseline - last observed: {baseline3}\n")
            '''
            f.write('------------------------------------\n')
            f.write(f'final testing loss (cross-entropy): {loss}\n')
            f.write(f'final testing absolute error (MAE): {err}\n')

    return dict(loss=loss, err=err, prediction=prediction,
                #baseline1=baseline1, baseline2=baseline2, baseline3=baseline3
                )


def run_paramsearch(D,
                    Z,
                    X,
                    Y,
                    trainids,
                    validids,
                    forward,
                    threshold,
                    spacialonly,
                    temporalonly,
                    dbbackend,
                    ntrials,
                    poissonmodel,
                    epochs,
                    temporalunit=None,
                    ipaddress=None,
                    port=None,
                    logfile=None,
                    foldid=None,
                    studyname=None,
                    savestudy=False):
    '''Create study or load study based on dbbackend arg'''

    print(f"dbbackend:{dbbackend}, studyname: {studyname}, ipaddress:{ipaddress}, port:{port}")

    if dbbackend:
        '''load study'''
        lstorage = f"postgresql://isl020@{ipaddress}:{port}/{studyname}"
        lstudy_name = f"{studyname}-{foldid}"

        study = optuna.load_study(storage=lstorage, study_name=lstudy_name)

        print(f"Storage: {lstorage}")
        print(f"Study name: {lstudy_name}")

    else:
        '''create study'''
        studyname = f"SP_{spacialonly}-TEMP_{temporalonly}-NTRIALs_{ntrials}-fold_{foldid}"
        study = optuna.create_study(study_name=studyname)

    '''load the param search sapce: four options:
        - gcn
        - spacialonly
        - temporal (lstm)
        - temporal (gru)
    '''
    if spacialonly:
        modelname = "spacial"
    elif temporalonly:
        modelname = f"temporal-{temporalunit}"
    else:
        modelname = "gcn"

    paramspace_lst = define_hpspace(modelname)

    study.optimize(lambda trial: objective(trial,
                                           D,
                                           Z,
                                           X,
                                           Y,
                                           trainids,
                                           validids,
                                           poissonmodel=poissonmodel,
                                           epochs=epochs,
                                           threshold=threshold,
                                           forward=forward,
                                           spacialonly=spacialonly,
                                           temporalonly=temporalonly,
                                           paramspace_lst=paramspace_lst,
                                           logfile=logfile
                                           ),
                   n_trials=ntrials
                   )

    besttrail_ = study.best_trial

    hidden = []
    for i in range(besttrail_.params["hidden_n_layers"]):
        hidden.append(besttrail_.params["n_units"])

    '''save trials'''
    if savestudy:
        studyname = f"{logfile[:logfile.rindex('/')]}/{studyname}.pkl"
        joblib.dump(study, studyname)



    dic_result = dict(lrate=besttrail_.params["lrate"],
                      epochs=besttrail_.params["epochs"],
                      backward=besttrail_.params["backward"],
                      hidden=hidden,
                      bestvalue=study.best_value
                      )

    '''no kernel width in case of Temporal only models'''
    if not temporalonly:
        dic_result['kernelwidth'] = besttrail_.params["kernelwidth"]

    return dic_result


def objective(trial,
              D,
              Z,
              X,
              Y,
              trainids,
              validids,
              epochs,
              poissonmodel,
              spacialonly,
              temporalonly,
              paramspace_lst,
              threshold,
              forward,
              logfile=None):
    params = dict()

    for p in paramspace_lst:
        if p["type"] == "int":
            params[p["name"]] = trial.suggest_int(p["name"], p["low"], p["high"])
        elif p["type"] == "loguniform":
            params[p["name"]] = trial.suggest_loguniform(p["name"], p["low"], p["high"])

    hidden_units = trial.suggest_categorical(f"n_units", [32, 64, 128, 200, 256])
    hiddenl_sizes = [hidden_units] * params["hidden_n_layers"]

    params["hidden"] = hiddenl_sizes
    params["epochs"] = trial.suggest_int("epochs", epochs, epochs)

    del params['hidden_n_layers']

    '''' in parameter search X is trainX and it is divided into train and valid '''
    train_X = X[trainids].reshape(-1, X.shape[1], 2)
    print(f'X shape {X.shape}')
    print(f'train X shape {train_X.shape}')
    print(f'train id shape {trainids.shape}')
    train_Y = Y[trainids].reshape(-1, X.shape[1])

    model_result = runmodel(D=D,
                            Z=Z,
                            X=train_X,
                            Y=train_Y,
                            trainids=validids,
                            poissonmodel=poissonmodel,
                            forward=forward,
                            threshold=threshold,
                            spacialonly=spacialonly,
                            temporalonly=temporalonly,
                            logfile=logfile,
                            **params
                            )

    return model_result['loss']


def main():
    parser = argparse.ArgumentParser(description='COTS population prediction')

    parser.add_argument('culling', type=str, help='filename of the Culling data')
    parser.add_argument('--manta', type=str, help='filename of the Manta-tow data')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train')
    parser.add_argument('--kernelwidth', type=float, default=1600, help='kenrel width to compute the graph')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256], help='size of all hidden layers')
    parser.add_argument('--testratio', type=float, default=0.2, help='ratio of testing data')
    parser.add_argument('--backward', type=int, default=40, help='how many previous days of data for the prediction')
    parser.add_argument('--forward', type=int, default=3, help='how many days to predict')
    parser.add_argument('--lrate', type=float, default=0.003, help='initial learning rate')
    parser.add_argument('--threshold', type=float, default=1e-5, help='threshold to sparsify the adjacency matrix')

    parser.add_argument('--temporalonly', action='store_true',
                        help='skip spatial information and perform temporal regression (GRU) only')
    parser.add_argument('--gruunit', action='store_true', help='perform temporal regression wtih GRU units only')
    parser.add_argument('--lstmunit', action='store_true', help='perform temporal regression wtih LSTM units only')
    parser.add_argument('--spacialonly', action='store_true',
                        help='skip temporal information and perform spacial regression (MLP) only')

    parser.add_argument('--poissonmodel', action='store_true', help='model the output as counts')

    parser.add_argument('--ncv', type=int, default=1, help='number of folds for nested cross-validation')
    parser.add_argument('--dbbackend', action='store_true', help='indicate whether to run as a single process')
    parser.add_argument('--savestudy', action='store_true', help='whether to save the studies when dbbackend is false.')

    parser.add_argument('--paramsearch', action='store_true',
                        help='determine the running mode. If true, run the paramter search.')
    parser.add_argument('--ntrials', type=int, default=200, help='the number of trails to run for parameter search.')

    parser.add_argument('--ipaddress', default="", type=str, help='the ip of the machine running the db.')
    parser.add_argument('--port', default=0, type=int, help='the ip of the machine running the db.')
    parser.add_argument('--studyname', default="", type=str, help='the name of the study.')

    parser.add_argument('--spacedsplit', action='store_true', help='split the train and test into spaced intervals.')

    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gbrdata = GBRDataset(args.culling, args.poissonmodel)

    '''Logs'''
    start_t = time.time()

    '''creates a directory with current time stamp'''
    log_prefix = f'{datetime.now().day}-{datetime.now().month}-{datetime.now().hour}-SP_{args.spacialonly}-TMP_{args.temporalonly}-' \
                 f'LSTM_{args.lstmunit}-GRU_{args.gruunit}-ntrails_{args.ntrials}-ZIP_{args.poissonmodel}-SS_{args.spacedsplit}-epoch_{args.epochs}'

    Path(f"experiments/{log_prefix}").mkdir(parents=True, exist_ok=True)

    ''''Define model parameters'''
    model_params = dict(forward=args.forward,
                        threshold=args.threshold
                        )

    model_params['spacialonly'] = args.spacialonly
    model_params['temporalonly'] = args.temporalonly
    model_params['epochs'] = args.epochs

    with open(f"experiments/{log_prefix}/best-params.txt", "a", encoding='utf-8') as f:
        f.write(f"Data size: {gbrdata[:]['x'].shape}\n")

    if args.temporalonly:
        if args.lstmunit:
            model_params['temporalunit'] = 'lstm'
        elif args.gruunit:
            model_params['temporalunit'] = 'gru'
        else:
            print(f"Temporal unit type undefiend!")
            exit(0)

    '''add model params from args if not paramsearch'''
    if not args.paramsearch:
        model_params['lrate'] = args.lrate
        model_params['hidden'] = args.hidden
        model_params['backward'] = args.backward

        if args.kernelwidth is None:
            model_params['kernelwidth'] = gbrdata.kernelwidth
        else:
            model_params['kernelwidth'] = args.kernelwidth
    else:
        '''define hyper parameter search parameters'''
        hp_params = dict(ntrials=args.ntrials, dbbackend=args.dbbackend)

        '''initilize best param log file'''
        with open(f"experiments/{log_prefix}/best-params.txt", "a", encoding='utf-8') as f:
            f.write(f"--" * 20 + "\n")
            f.write(f"Parameter search results\n")
            f.write(f"--" * 20 + "\n")

        if args.dbbackend:
            if args.ipaddress == "":
                print("IP address of database server is not defined!!")
                exit(0)
            elif args.port == 0:
                print("Port of database server is not defined!!")
                exit(0)
            elif args.studyname == "":
                print("Study name is not defined!!")
                exit(0)
            else:
                hp_params['ipaddress'] = args.ipaddress
                hp_params['port'] = args.port
                hp_params['studyname'] = args.studyname
        else:
            '''dbbackend is false check savestudy'''
            hp_params['savestudy'] = args.savestudy

    '''define the search keys according to model. Temporal models do not have kernelwidth'''
    searhkeys = ['lrate', 'hidden', 'backward']

    if not model_params['temporalonly']:
        searhkeys.append('kernelwidth')

    '''divide the data into ncv folds'''
    cvratio = 1 / args.ncv
    cvratios = [1 - i * cvratio for i in range(args.ncv)]
    cvratios.reverse()

    fold_results = []
    run_args = dict()

    print(f"cvratios: {cvratios}")
    for foldid, ratio in enumerate(cvratios):
        print("===" * 20)
        print(f"Started processing fold {foldid}")
        print("===" * 20)

        '''
        for each fold:
            1. divide the data into train and test
            2. use train to find the best params.
                    2.a divide train into train and validation
                    2.b train model using train
                    2.c evaluate using valid
            3. train the model with best parameters and train.
            4. evaluate using test set and record loss                
        '''
        foldsize = int(len(gbrdata) * ratio)

        X = gbrdata[:foldsize]['x']
        Y = gbrdata[:foldsize]['y']

        #print(X[0])
        #print(Y[0])
        #exit(0)

        trainids = split_train_test_data(X,
                                         args.testratio,
                                         split_space=args.spacedsplit,
                                         split_from_tail=True
                                         )
        run_args['D'] = gbrdata.D
        run_args['Z'] = gbrdata.Z

        run_args['X'] = X
        run_args['Y'] = Y
        run_args['trainids'] = trainids


        run_args['poissonmodel'] = args.poissonmodel

        run_args['logfile'] = f"experiments/{log_prefix}/fold_{foldid}"

        '''setting up parameter search'''
        if args.paramsearch:
            # ------divide the dataset into train and test------
            train_X = X[trainids].reshape(-1, X.shape[1], X.shape[2])
            validids = split_train_test_data(train_X,
                                             args.testratio,
                                             split_from_tail=True
                                             )

            hp_params['validids'] = validids
            hp_params['foldid'] = foldid

            print(f"Running model with paramsearch with ratio {ratio}")

            '''remove the search results from previous run'''
            for key in searhkeys:
                if key in model_params.keys():
                    del model_params[key]

            hp_best_params = run_paramsearch(**run_args, **model_params, **hp_params)

            '''assign the model params according to hp searh result'''
            for key in searhkeys:
                model_params[key] = hp_best_params[key]

            '''log best params'''
            with open(f"experiments/{log_prefix}/best-params.txt", "a", encoding='utf-8') as f:
                f.write(f"Fold: {foldid}\n")
                # f.write(f"Train X size: {train_X.shape}\n")
                # f.write(f"Train Y size: {train_Y.shape}\n")
                # f.write(f"Test X size: {test_X.shape}\n")
                # f.write(f"Test Y size: {test_Y.shape}\n")

                # f.write(f"HP search X Train size: {ttrain_X.shape}\n")
                # f.write(f"HP search Y Train size: {ttrain_Y.shape}\n")
                # f.write(f"Validation X size: {valid_X.shape}\n")
                # f.write(f"Validation Y size: {valid_Y.shape}\n")

                f.write(f"{' '.join(key + ': ' + str(val) for key, val in model_params.items())}")
                f.write("\n")
                f.write(f"Trial loss: {hp_best_params['bestvalue']}")
                f.write("\n\n")

            '''
            For running the model with best params, 
                 - remove X_valid and Y_valid
                 - Make X_train the previous state: 
            '''
            # run_args['X_train'] = train_X
            # run_args['Y_train'] = train_Y
        else:
            print(f"Running model without param search with ratio {ratio}")
            run_args['verbose'] = True

        '''----Write model params ------'''
        with open(f"experiments/{log_prefix}/model-params.txt", "w", encoding='utf-8') as f:
            if not model_params['temporalonly']:
                f.write(f"Kernel width: {model_params['kernelwidth']}\n")

            f.write(f"threshold: {model_params['threshold']}\n")
            f.write(f"lrate: {model_params['lrate']}\n")
            f.write(f"epochs: {model_params['epochs']}\n")
            f.write(f"hidden layers: {model_params['hidden']}\n")
            f.write(f"backward: {model_params['backward']}\n")
            f.write(f"forward: {model_params['forward']}\n")
            f.write(f"temporalonly: {model_params['temporalonly']}\n")
            f.write(f"spacialonly: {model_params['spacialonly']}\n")
            f.write(f"ZIPmodel: {run_args['poissonmodel']}\n")

            if model_params['temporalonly']:
                f.write(f"temporalunit:{model_params['temporalunit']}\n")

            '''may write the model summary later'''

        results = runmodel(**run_args, **model_params, savemodel=True)

        '''-------log fold results----------'''
        print(f"loss: {results['loss']}, error: {results['err']}")

        fold_results.append(dict(error=results["err"],
                                 loss=results["loss"],
                                 #baseline1=results["baseline1"],
                                 #baseline2=results["baseline2"],
                                 #baseline3=results["baseline3"]
                                 )
                            )

    '''---log average results'''
    with open(f"experiments/{log_prefix}/performance_summary.txt", "w", encoding='utf-8') as f:
        f.write(f"Average results across folds: \n")
        f.write(f"Baseline - global average: {sum(d['baseline1'] for d in fold_results) / len(fold_results)}\n")
        f.write(f"Baseline - site average: {sum(d['baseline2'] for d in fold_results) / len(fold_results)}\n")
        f.write(f"Baseline - last observed: {sum(d['baseline3'] for d in fold_results) / len(fold_results)}\n")
        f.write('------------------------------------\n')

        losstype = "ZIP loss" if run_args['poissonmodel'] else "BCE loss"

        f.write(f"Final testing loss ({losstype}): {sum(d['loss'] for d in fold_results) / len(fold_results)}\n")
        f.write(f"Final testing absolute error (MAE): {sum(d['error'] for d in fold_results) / len(fold_results)}\n")
        f.write(f"Time taken: {(time.time() - start_t) / 3600} hours\n")


if __name__ == '__main__':
    main()
