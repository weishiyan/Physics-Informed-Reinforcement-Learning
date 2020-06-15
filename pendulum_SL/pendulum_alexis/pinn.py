import numpy as np
import tensorflow as tf
import lbfgs
import matplotlib.pyplot as plt
from IPython import display


nt_config = lbfgs.Struct()
nt_config.learningRate = 0.8
nt_config.maxIter = 0 #10 #nt_epochs
nt_config.nCorrection = 50
nt_config.tolFun = 1.0 * np.finfo(float).eps


class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, t_f, ub, lb):
    # Descriptive Keras model [2, 20, …, 20, 1]
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.u_model.add(tf.keras.layers.Lambda(
      lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(
          width, activation=tf.nn.tanh,
          kernel_initializer='glorot_normal'))

    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        self.sizes_w.append(int(width * layers[1]))
        self.sizes_b.append(int(width if i != 0 else layers[1]))

    self.optimizer = optimizer
    self.logger = logger

    self.dtype = tf.float32

    # Collocation coordinates
    self.t_f = tf.convert_to_tensor(t_f[:, None], dtype=self.dtype)



  # Defining custom loss
  def __loss(self, theta, theta_pred):
    f_pred = self.f_model()
    u_loss = tf.reduce_mean(tf.square(theta - theta_pred))
    f_loss = tf.reduce_mean(tf.square(f_pred))
    #list_loss.append([u_loss, f_loss, (u_loss+f_loss)]) # NN, PINN, Sum_of_PINN/NN
    return u_loss, f_loss

  def __grad(self, t, theta):
    with tf.GradientTape() as tape:
      u_loss, f_loss = self.__loss(theta, self.u_model(t))
      loss_value = u_loss + f_loss
    return u_loss, f_loss, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_model.trainable_variables
    return var

  # The actual PINN
  def f_model(self):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
      # Watching time input
      tape.watch(self.t_f)
      # Getting the prediction
      u = self.u_model(self.t_f) # theta prediction from model
      print("\t predicted theta", u[0]) ##u[0])
      # Deriving INSIDE the tape
      u_t = tape.gradient(u, self.t_f) # d_theta/d_time
      print("\t u_t:", u_t[0])

    # Getting the second derivative
    u_tt = tape.gradient(u_t, self.t_f) # d2_theta/d_time2
    #print("u_tt: ", u_tt)

    # Letting the tape go
    del tape

    # Buidling the PINNs
      # using small-angle apprximation
    return (u_tt + 10.0*np.sin(u)) #10.0 holding value for now (g/l)

  #def get_params(self, numpy=False):
  #  return self.g, self.l

  def get_weights(self):
    w = []
    for layer in self.u_model.layers[1:]:
      weights_biases = layer.get_weights()
      weights = weights_biases[0].flatten()
      biases = weights_biases[1]
      w.extend(weights)
      w.extend(biases)
    return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    for i, layer in enumerate(self.u_model.layers[1:]):
      start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)

  def summary(self):
    return self.u_model.summary()

  # The training function
  def fit(self, t_train, theta, nt_config, tf_epochs=5000):
    self.logger.log_train_start(self)
    #self.u_model.compile(optimizer=self.optimizer, loss='mse')
    #self.u_model.fit(t_train,theta)
    # Creating the tensors
    t_train = tf.convert_to_tensor(t_train[:,None], dtype=self.dtype)
    theta = tf.convert_to_tensor(theta[:,None], dtype=self.dtype)

    # Reporting loss and logging
    fnew = open("logs/Loss_report_tf.out", "w+")
    fnew.write("u_loss \t \t f_loss \t \t sum_loss \n")
    loss_list = []
    self.logger.log_train_opt("Adam")

    for epoch in range(tf_epochs):
      # Optimization step
      u_loss, f_loss, grads = self.__grad(t_train, theta)
      loss_value = u_loss + f_loss
      fnew.write("%s \t %s \t %s \n" %("{:0.4e}".format(u_loss), "{:0.4e}".format(f_loss), "{:0.4e}".format(loss_value)))
      loss_list.append([u_loss, f_loss, loss_value])
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))

      self.logger.log_train_epoch(epoch, loss_value)

      #if epoch % 1000 == 0:
      #  #plt.clf()
      #  plt.scatter(t_train, theta, marker='.')
      #  plt.scatter(t_train, self.u_model(t_train), marker='.')
      #  display.display(plt.gcf())
      #  display.clear_output(wait=True)
      #  plt.savefig("train.png")
      #  #plt.show()

    self.logger.log_train_opt("LBFGS")
    fnew = open("logs/Loss_report_nt.out", "w+")
    fnew.write("u_loss \t \t f_loss \t \t sum_loss \n")
    def loss_and_flat_grad(w):

      with tf.GradientTape() as tape:
        self.set_weights(w)
        u_loss, f_loss = self.__loss(theta, self.u_model(t_train))
        loss_value = u_loss + f_loss
        fnew.write("%s \t %s \t %s \n" %("{:0.4e}".format(u_loss), "{:0.4e}".format(f_loss), "{:0.4e}".format(loss_value)))
      grad = tape.gradient(loss_value, self.u_model.trainable_variables)
      grad_flat = []
      for g in grad:
        grad_flat.append(tf.reshape(g, [-1]))
      grad_flat =  tf.concat(grad_flat, 0)
      return loss_value, grad_flat
    # tfp.optimizer.lbfgs_minimize(
    #   loss_and_flat_grad,
    #   initial_position=self.get_weights(),
    #   num_correction_pairs=nt_config.nCorrection,
    #   max_iterations=nt_config.maxIter,
    #   f_relative_tolerance=nt_config.tolFun,
    #   tolerance=nt_config.tolFun,
    #   parallel_iterations=6)
    lbfgs.lbfgs(loss_and_flat_grad,
      self.get_weights(),
      nt_config, lbfgs.Struct(), True,
      lambda epoch, loss, is_iter:
        self.logger.log_train_epoch(nt_config.epoch, loss, "", is_iter))


    self.logger.log_train_end(tf_epochs + nt_config.maxIter)
    loss_list = np.array(loss_list)
    np.savez("pendulum_loss", loss=loss_list)
    fnew.close()

  def predict(self, t):

    theta_pred = self.u_model.predict(t)
    f_pred = self.f_model()
    return theta_pred, f_pred
