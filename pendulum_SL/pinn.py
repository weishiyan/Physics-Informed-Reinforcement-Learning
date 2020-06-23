import tensorflow as tf
import matplotlib.pyplot as plt


class PhysicsInformedNN(object):
    def __init__(self, layers, optimizer, logger, X_f, ub, lb, g, l):
        # Descriptive Keras model [1, 20, …, 20, 1]
        self.u_model = tf.keras.Sequential()
        self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        self.u_model.add(
            tf.keras.layers.Lambda(lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
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
        self.len = l
        self.nu = g/l
        self.optimizer = optimizer
        self.logger = logger
        self.tolX = 5e-2
        self.dtype = tf.float32
        self.t_f = tf.convert_to_tensor(X_f, dtype=self.dtype)

    def __loss(self, u, u_pred):
        f_pred = self.f_model()
        return tf.reduce_sum(
            tf.square(u - u_pred)) # + tf.reduce_mean(tf.square(f_pred))

    def __grad(self, X, u):
        with tf.GradientTape() as tape:
            loss_value = self.__loss(u, self.u_model(X))
        return loss_value, tape.gradient(
            loss_value, self.__wrap_training_variables())

    def __wrap_training_variables(self):
        var = self.u_model.trainable_variables
        return var

    # The actual PINN
    def f_model(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the input we’ll need later, t
            tape.watch(self.t_f)
            # Getting the prediction
            u = self.u_model(self.t_f)
            u_t = tape.gradient(u, self.t_f)

        # Getting the other derivatives
        u_tt = tape.gradient(u_t, self.t_f)
        nu = self.get_params(numpy=True)

        # Buidling the PINNs... assuming small angle approx.
        return nu*(u) + u_tt

    def get_params(self, numpy=False):
        return self.nu

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

    def fit(self, X_u, u, tf_epochs=5000):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
        u = tf.convert_to_tensor(u, dtype=self.dtype)
        self.logger.log_train_opt("Adam")

        # Determining the number of images to output
        fps = 24
        time = 10
        mod = int(tf_epochs / (fps * time))
        for epoch in range(tf_epochs):
            # Optimization step
            loss_value, grads = self.__grad(X_u, u)
            self.optimizer.apply_gradients(
                zip(grads, self.u_model.trainable_variables))
            self.logger.log_train_epoch(epoch, loss_value)
            if epoch % mod == 0:
                plt.clf()
                x = X_u
                exact = u
                predict = self.u_model(X_u)
                ratio = 1
                plt.scatter(x, exact, marker='.')
                plt.scatter(x, ratio*predict, marker='.')
                plt.xlabel("Time (s)")
                plt.ylabel("Theta")
                plt.title("%s Epochs (pendulum length = %s)" % (
                    str(epoch), str(self.len)))
                plt.savefig("plots/%s_PINN_Epochs.png" % str(epoch), dpi=300)
                plt.close()
            else:
                pass
            if tf.abs(loss_value) < self.tolX:
                plt.clf()
                x = X_u
                exact = u
                predict = self.u_model(X_u)
                ratio = 1

                plt.scatter(x, exact, marker='.')
                plt.scatter(x, ratio*predict, marker='.')
                plt.xlabel("Time (s)")
                plt.ylabel("Theta")
                plt.title("%s Epochs (Final)" % str(epoch))
                plt.savefig(
                    "plots/PINN_Opt_Completed_%s_Epochs.png" % str(epoch))
                plt.close()
                break
            else:
                pass
        self.logger.log_train_end(tf_epochs)

    def predict(self, X_star):
        u_star = self.u_model(X_star)
        f_star = self.f_model()
        return u_star, f_star
