from rsl_rl import OnPolicyRunner




def main():
    # interpret run args
    run_with_gui = False # dummy variable
    load_existing_log = False # dummy variable
    
    # load configs
    configs = config.load({
        'model':'default',
        'env':'static'
        'obs':'default',
        'reward':'simple',
        'command':'random_walk'
    })
    # load model via rsl-rl OR initialize fresh one
    if load_existing_log:
        model = model.load(load_existing_log)
    else:
        model = model.initialize(configs)
    # decide whether or not to do a GUI 

    # run a training loop

    # do tensorboard logging, live checkpoint updates

    # quit when done

    pass