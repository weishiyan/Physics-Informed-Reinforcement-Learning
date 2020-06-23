import numpy as np

# def test_prep_data():
# Collected data from file
npz = np.load("data/single_random_pendulum_data.npz", allow_pickle=True)
states_data = npz["states"]
rewards_data = npz['rewards']
# 50000 corresponding times for each state of the pendulum
time_data = npz['time']

# Reduce data set size for testing
data_set_size = 100
theta = states_data[:data_set_size, 1]
time = time_data[:data_set_size]

# Build distribution
# dist_size = 10
# dist_theta = np.empty((dist_size,len(theta)), dtype=float)

stdev = 1
theta_dist = []
for th in range(len(theta)):
    theta_dist.append(np.random.normal(theta[th], stdev))
print(len(theta_dist))
