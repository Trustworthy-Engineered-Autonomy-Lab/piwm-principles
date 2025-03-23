import torch
import numpy as np

def preprocess_data(npz_file, device):
    data = np.load(npz_file)
    images_pole = data['obs'][:, 0, :, :, :]  # Shape: (46326, 96, 96, 3)
    images_pole = np.mean(images_pole, axis=3, keepdims=True)  # Convert RGB to grayscale (N, 96, 96, 1)
    # images_pole = images_pole / 255.0  # Normalize to [0, 1]
    # images_pole = np.transpose(images_pole, (0, 3, 1, 2))  # Convert to (N, 1, 96, 96)
    images_pole = torch.tensor(images_pole, dtype=torch.float32).to(device)

    states = np.stack([
        data['Cart_Position'],
        data['Cart_Velocity'],
        data['Pole_Angle'],
        data['Pole_Angular_Velocity']
    ], axis=1)  # Shape: (46326, 4)
    states = torch.tensor(states, dtype=torch.float32).to(device)

    return images_pole, states


# load in cpu
def divided_data(npz_file):
    data = np.load(npz_file)
    images_pole = data['obs']
    images_pole = images_pole / 255.0  # Normalize to [0, 1]
    images_pole = np.transpose(images_pole, (0, 1, 4, 2, 3))  # Convert to (N, ID, C, H, W)
    images_pole = torch.tensor(images_pole, dtype=torch.float32)#.to(device)

    states = np.stack([
        data['Cart_Position'],
        data['Cart_Velocity'],
        data['Pole_Angle'],
        data['Pole_Angular_Velocity']
    ], axis=1)  # Shape: (46326, 4)
    states = torch.tensor(states, dtype=torch.float32)#.to(device)
    
    indices = torch.randperm(images_pole.shape[0])
    images_pole = images_pole[indices]
    states = states[indices]
    return images_pole, states
