import numpy as np 
import matplotlib.pyplot as plt 

# NOTE: column1 = mse NN, column2 = mse PINN, column3 = mse NN + mse PINN
p = np.load("single_random_pendulum_data2.npz", allow_pickle=True)
p_states = p["states"]
p_time = p["time"]
mu, sigma = 0, 0.1
random_noise = np.random.normal(mu, sigma, [len(p_states)]) # give same shape as data
theta = np.arcsin(p_states[:,1]) + random_noise
fig = plt.plot(figsize=(12,7))
set_size = 500
# c_inference
#x = np.linspace(0,p_array.shape[0],p_array.shape[0])
plt.title("Single Action Pendulum Data (%i points from data set)" %set_size)
plt.ylabel("Theta")
plt.xlabel("Time")
plt.plot(p_time[:set_size],theta[:set_size], c="orange") #, label="$MSE_{NN}$")  # mse NN
#axs.legend()

plt.savefig("pendulum_data_single_action_noise2.png")
