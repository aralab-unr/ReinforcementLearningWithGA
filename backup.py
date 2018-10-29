#function created to make it run with GA, since it requires many calls to fitness function
def default_params():


    DEFAULT_PARAMS1 = {
        # env
        'max_u': 1.,  # max absolute value of actions on different coordinates
        # ddpg
        'layers': 3,  # number of layers in the critic/actor networks
        'hidden': 256,  # number of neurons in each hidden layers
        'network_class': 'baselines.her.actor_critic:ActorCritic',
        'Q_lr': 0.001,  # critic learning rate
        'pi_lr': 0.001,  # actor learning rate
        'buffer_size': int(1E6),  # for experience replay
        'polyak': 0.95,#round(random.uniform(0, 1), 3),  # polyak averaging coefficient 0.95
        'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
        'clip_obs': 200.,
        'scope': 'ddpg',  # can be tweaked for testing
        'relative_goals': False,
        # training
        'n_cycles': 50,  # per epoch
        'rollout_batch_size': 2,  # per mpi thread
        'n_batches': 40,  # training batches per cycle
        'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
        'test_with_polyak': False,  # run test episodes with the target network
        # exploration
        'random_eps': 0.3,  # percentage of time a random action is taken
        'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        # HER
        'replay_strategy': 'future',  # supported modes: future, none
        'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
        # normalization
        'norm_eps': 0.01,  # epsilon used for observation normalization
        'norm_clip': 5,  # normalized observations are cropped to this values
    }

    return DEFAULT_PARAMS1


    params=default_params()
    kwargs['max_u'] = params['max_u']
    kwargs['buffer_size'] = params['buffer_size']
    kwargs['hidden'] = params['hidden']
    kwargs['layers'] = params['layers']
    kwargs['network_class'] = params['network_class']
    kwargs['batch_size'] = params['batch_size']
    kwargs['Q_lr'] = params['Q_lr']
    kwargs['pi_lr'] = params['pi_lr']
    kwargs['norm_eps'] = params['norm_eps']
    kwargs['norm_clip'] = params['norm_clip']
    kwargs['action_l2'] = params['action_l2']
    kwargs['clip_obs'] = params['clip_obs']
    kwargs['scope'] = params['scope']
    kwargs['relative_goals'] = params['relative_goals']
    kwargs['replay_k'] = params['replay_k']

    # remove this
    params['ddpg_params'] = dict()
    params['make_env'] = 'test'




