import numpy as np 
import tensorflow as tf
import sys



# Collect data
npz = np.load('data/single_random_pendulum_data.npz', allow_pickle=True)
# 50000 states of the pendulum
states_data = npz['states']
rewards_data = npz['rewards']
# 50000 corresponding times for each state of the pendulum
time_data = npz['time']

#theta = states_data[:,1] # sin(theta)
theta_dot = states_data[:,2]
theta_cos1 = [np.arccos(x) for x in states_data[:,0]]
theta = np.array(theta_cos1)

set_size = 10

time = np.array(time_data[:set_size])
theta = np.array(theta[:set_size])
theta_dot = theta_dot[:set_size]
data = np.vstack((time,theta)) #zip(time, theta)# np.array(zip(np.array(time),np.array(theta)))
#print(data.shape)
print(data[:,0])
print(data.shape)

# ========== FROM: https://www.tensorflow.org/api_docs/python/tf/GradientTape ====================
#x = tf.constant(3.0)
#with tf.GradientTape() as g:
#  g.watch(x)
#  y = x * x
#dy_dx = g.gradient(y, x) # Will compute to 6.0
#print(dy_dx)
#
#x = tf.constant(3.0)
#with tf.GradientTape() as g:
#  g.watch(x)
#  with tf.GradientTape() as gg:
#    gg.watch(x)
#    y = x * x
#  dy_dx = gg.gradient(y, x)     # Will compute to 6.0
#d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
#print(dy_dx)
#print(d2y_dx2)

# The code above works
# =================================================================================================


#time = tf.constant(0.05)
time = tf.convert_to_tensor(time[:,None], dtype=tf.float32)
#print(type(time))
#print(theta)
#theta = tf.convert_to_tensor(data, dtype=tf.float32)
#print(type(theta))
def model(time):
    npz = np.load('data/single_random_pendulum_data.npz', allow_pickle=True)
    # 50000 states of the pendulum
    states_data = npz['states']    
    #theta = states_data[:,1] # sin(theta)
    theta_cos1 = [np.arccos(x) for x in states_data[:,0]]
    theta = np.array(theta_cos1)

    set_size = 10

    theta = np.array(theta[:set_size])
    theta = tf.convert_to_tensor(theta, dtype=tf.float32)
    return theta

with tf.GradientTape() as tape:    
#    
#    # Watching time input
    tape.watch(time)
#    # Deriving INSIDE the tape
    #for i in range(set_size-1):
    #y = time #np.cos(np.sqrt(10)*time)
    u = model(time)
    u_t = tape.gradient(u, time) # d_theta/d_time
    tf.print(u_t)
# Getting the second derivative
#u_tt = tf.GradientTape(u_t, time) # d2_theta/d_time2
# Letting the tape go
# Buidling the PINNs
# using small-angle apprximation
#(u_tt + 10.0*u) #10.0 holding value for now (g/l)
#tf.print(u_tt)
