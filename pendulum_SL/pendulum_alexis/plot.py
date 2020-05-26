import numpy as np 
import matplotlib.pyplot as plt 

# NOTE: column1 = mse NN, column2 = mse PINN, column3 = mse NN + mse PINN

def exact_vs_pred(t, t_train, theta, theta_train, theta_pred):
    plt.plot(t,theta, "b-", label = "Exact")
    end = theta_pred.shape[0]
    plt.plot(t,theta_pred, "r--", label="Prediction")
    #plt.scatter(t_train,theta_train,c="g", label="Training Points")
    plt.ylabel("Theta")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig("plots/pred_vs_exact.png")
    return

def loss(file):
#file = "pendulum_loss"
    p_loss = np.load("%s.npz" %file, allow_pickle=True)
    p_loss_array = p_loss["loss"]
    #print("p_loss_array", p_loss_array.shape[0])
    fig, axs = plt.subplots(1,3,sharey='row', figsize=(12,7))
    set_size = 950
    # c_inference
    x = np.linspace(0,p_loss_array.shape[0],p_loss_array.shape[0])
    axs[0].set_title("$MSE_{NN}$ (tf)")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("epoch")
    axs[0].plot(x[:],p_loss_array[:,0], c="g") #, label="$MSE_{NN}$")  # mse NN
    axs[1].set_title("$MSE_{PINN}$ (tf)")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("epoch")
    axs[1].plot(x[:],p_loss_array[:,1], c="b") #, label="$MSE_{PINN}$")  # mse PINN
    axs[2].set_title("$MSE_{NN}$+$MSE_{PINN}$ (tf)")
    axs[2].set_ylabel("Loss")
    axs[2].set_xlabel("epoch")
    axs[2].plot(x[:],p_loss_array[:,2], c="r") #, label="$MSE_{NN}$+$MSE_{PINN}$")  # mse NN + mse PINN
    #axs.legend()
    fig.tight_layout()
    plt.savefig("plots/pendulum_loss.png")
    return


        