import numpy as np 
import matplotlib.pyplot as plt 

# NOTE: column1 = mse NN, column2 = mse PINN, column3 = mse NN + mse PINN
def plotting(file):
    p_loss = np.load("%s.npz" %file, allow_pickle=True)
    p_loss_array = p_loss["loss"]

    print("p_loss_array", p_loss_array.shape[0])

    fig, axs = plt.subplots(1,3,figsize=(12,7))
    set_size = 950
    # c_inference
    x = np.linspace(0,p_loss_array.shape[0],p_loss_array.shape[0])
    axs[0].set_title("$MSE_{NN}$ (tf)")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("epoch")
    axs[0].scatter(x[:],p_loss_array[:,0], s=5, c="g") #, label="$MSE_{NN}$")  # mse NN
    axs[1].set_title("$MSE_{PINN}$ (tf)")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("epoch")
    axs[1].scatter(x[:],p_loss_array[:,1], s=5, c="b") #, label="$MSE_{PINN}$")  # mse PINN
    axs[2].set_title("$MSE_{NN}$+$MSE_{PINN}$ (tf)")
    axs[2].set_ylabel("Loss")
    axs[2].set_xlabel("epoch")
    axs[2].scatter(x[:],p_loss_array[:,2], s=5, c="r") #, label="$MSE_{NN}$+$MSE_{PINN}$")  # mse NN + mse PINN

    #axs.legend()
    fig.tight_layout()

    plt.savefig("pendulum_loss.png")
