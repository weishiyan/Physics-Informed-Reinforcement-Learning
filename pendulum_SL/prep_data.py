import numpy as np
import matplotlib.pyplot as plt 


def prep_data(N_u, N_f, noise=0.1):
    npz = np.load(
        'data/single_action_5_pendulum_data_L100.npz',
        allow_pickle=True)
    # 50000 states of the pendulum
    states_data = npz['states']
    # 50000 corresponding times for each state of the pendulum
    time_data = npz['time']
    theta = states_data[:, 1]
    theta = theta[:N_f]
    time = time_data[:N_f]
    lb = np.min(time)  # lowerbound
    ub = np.max(time)  # upperbound
    X_f = time
    Exact_u = theta
    idx = np.random.choice(time.shape[0], N_u, replace=False)
    noise_mod = np.random.normal(0, noise, theta.shape)
    u_train = theta + noise_mod
    u_train = u_train[idx]
    X_u_train = time[idx]

    plt.title("Training Dataset")
    plt.ylabel("Theta")
    plt.xlabel("Time (s)")
    plt.scatter(X_u_train, u_train, marker='.',c='g')
    plt.savefig("plots/Training_Dataset.png")
    return X_f, Exact_u, X_u_train, u_train, lb, ub
