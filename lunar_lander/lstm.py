
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
# Define the VAE architecture
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random
from nltk.corpus import words
import nltk

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define VAE class with customizable latent size
class VAE(nn.Module):
    def __init__(self, latent_size=128):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x).view(batch_size, -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.fc_decode(z).view(batch_size, 64, 10, 15)
        reconstructed = self.decoder(decoded)
        return reconstructed, mu, logvar
#
class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, output_size=10):

        super(LSTMTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        batch_size = x.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out)

        return out

def train_vae(predictor,model, train_loader, optimizer, device):
    model.eval()
    predictor.train()
    total_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader, desc="Training Progress")):
        input, label, action, use1, use2 = data
        x = input.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        recon_x2, mu2, logvar2 = model(label)

        z2 = predictor(mu.view(128,1,64))
        loss = nn.functional.mse_loss(z2, mu2.view(128,1,64), reduction='sum')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--train', type=str,default="/cartpole/train/", help='Path to train file')
    parser.add_argument('--test', type=str,default="/cartpole/test/", help='Path to test file')
    parser.add_argument('--save', type=str, default='./',
                        help='Path to output dir ')
    parser.add_argument('--latent', type=int, default=8,
                        help='latent size')

    args = parser.parse_args()
    from zeroloader import VaeDataset, SequenceDataset

    traindataset = VaeDataset(root=args.train)
    TrainLoader = DataLoader(traindataset, batch_size=128, shuffle=True, drop_last=True)
    # Training loop with custom latent size
    latent_size = 64  # Set your desired latent size here
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    vae_model = VAE(latent_size=latent_size).to(device)
    Predictor = LSTMTimeSeries(input_size=64, hidden_size=64, num_layers=2, output_size=64).to(device)

    optimizer = optim.Adam(Predictor.parameters(), lr=1e-3)
    vae_model.load_state_dict(torch.load("vae.pth",map_location=torch.device('cpu')))
    vae_model=vae_model.to(device)
    epochs = 200
    list1 = []
    list2 = []
    for epoch in range(epochs):

        train_loss = train_vae(Predictor,vae_model, TrainLoader,optimizer, device)

        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')

    torch.save(Predictor.state_dict(), 'p1lstm.pth')
