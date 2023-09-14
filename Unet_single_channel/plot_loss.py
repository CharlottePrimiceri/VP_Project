import torch
import matplotlib.pyplot as plt
import os

# function to plot the losses after training

def plot_train_and_val_loss(loss_vec, steps):

    epoch_ax = torch.arange(1, steps + 1)
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 1, 1)
    plt.plot(epoch_ax, loss_vec)
    plt.grid()

    plt.title("Training Loss"),
    plt.ylabel("Training Loss"),
    plt.xlabel("Steps, epochs")

    return plt.show()

def obtain_loss_vector(loss_file_path):
    with open(loss_file_path) as f:
        loss_values = f.readlines()
        for i in range(len(loss_values)):
            loss_values[i] = float(loss_values[i])
    return loss_values


#loss_vector = obtain_loss_vector("C:/Users/gnard/Documents/v/checkpoints for GIT/unet_loss_nodepth_50_epochs.txt")
#plot_train_and_val_loss(loss_vector, 9300)

#loss_vector = obtain_loss_vector("C:/Users/gnard/Documents/v/checkpoints for GIT/unet_loss_double_chann_50_epochs.txt")
#plot_train_and_val_loss(loss_vector, 9300)

#loss_vector = obtain_loss_vector("C:/Users/gnard/Documents/v/checkpoints for GIT/unet_loss_double_chann_15_epochs.txt")
#plot_train_and_val_loss(loss_vector, 2790)
