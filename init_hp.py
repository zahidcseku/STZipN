def define_hpspace(model):
    '''
    This method will define a hyper-parameter search space based on passed model.
    The search space is defined based on previous experiments involving 350 trials.

    Input:
        model - string: {gcn, spatial, temporal-gru, temporal-lstm}
    Output:
        list_params - a list of dictionaries of parameters. Each dictionary has
                    four fields - {name, type, low, high}
    '''
    list_params = []
    if model == "gcn":
        #list_params.append(dict(name="epochs", type="int", low=850, high=1150))
        list_params.append(dict(name="kernelwidth",type="int", low=1050, high=1200))
        list_params.append(dict(name="hidden_n_layers", type="int", low=2, high=8))
        list_params.append(dict(name="backward", type="int", low=40, high=60))
        list_params.append(dict(name="lrate", type="loguniform", low=0.0001, high=0.001))
    elif model == "spacial":
        #list_params.append(dict(name="epochs", type="int", low=400, high=1000))
        #list_params.append(dict(name="epochs", type="int", low=5, high=10))
        list_params.append(dict(name="hidden_n_layers", type="int", low=3, high=8))
        list_params.append(dict(name="backward", type="int", low=15, high=60))
        list_params.append(dict(name="lrate", type="loguniform", low=0.0001, high=0.001))
        list_params.append(dict(name="kernelwidth", type="int", low=450, high=700))
    elif model == "temporal-gru":
        #list_params.append(dict(name="epochs", type="int", low=400, high=750))
        #list_params.append(dict(name="epochs", type="int", low=4, high=6))
        list_params.append(dict(name="hidden_n_layers", type="int", low=3, high=8))
        list_params.append(dict(name="backward", type="int", low=40, high=60))
        list_params.append(dict(name="lrate", type="loguniform", low=0.0001, high=0.001))
    elif model == "temporal-lstm":
        #list_params.append(dict(name="epochs", type="int", low=750, high=1000))
        #list_params.append(dict(name="epochs", type="int", low=7, high=10))
        list_params.append(dict(name="hidden_n_layers", type="int", low=3, high=8))
        list_params.append(dict(name="backward", type="int", low=25, high=60))
        list_params.append(dict(name="lrate", type="loguniform", low=0.00001, high=0.0001))
    else:
        print(f"model param space is not defined for {model}!!!!")
        exit(0)


    return list_params