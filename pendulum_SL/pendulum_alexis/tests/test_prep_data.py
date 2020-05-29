from pendulum_alexis import prep_data
import numpy as np


def test_prep_data():
    npz = np.load('../data/single_random_pendulum_data.npz', allow_pickle=True)
    # 50000 states of the pendulum
    states_data = npz['states']

    # # Reward information
    # rewards_data = npz['rewards']
    # 50000 corresponding times for each state of the pendulum
    time_data = npz['time']

    # using sin(theta) data
    theta = states_data[:, 1]

    set_size = 1500  # reduce set size to accommodate computational limitations

    # Number of random samples should not be larger than the data size
    try:
        N_u = set_size + 10

        # Collocation points size, where we’ll check for f = 0
        N_f = 1500

        t, theta, t_train, theta_train, t_f, ub, lb = \
            prep_data.prep_data(theta[:set_size], time_data[:set_size],
                                N_u, N_f, noise=0.1)
    except (ValueError):
        print("1. Passed when random size is larger than data size.")
        pass
    else:
        raise Exception("Sample size larger than data size")

    N_u = int(set_size / 10)
    N_f = set_size

    t, theta, t_train, theta_train, t_f, ub, lb = \
        prep_data.prep_data(theta[:set_size], time_data[:set_size],
                            N_u, N_f, noise=0.1)

    # Train data should have same shape
    assert t_train.shape == theta_train.shape, \
        "Both training data should have the same shape."

    print("2. Both training data are same size.")

    # Train data should be same size as N_u
    assert t_train.shape[0] == N_u and theta_train.shape[0] == N_u, \
        "Training data should be same size as N_u."

    print("3. Both training data are size of N_u.")
