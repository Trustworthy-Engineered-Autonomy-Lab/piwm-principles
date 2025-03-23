"""
reconstruct image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime


class Encoder(nn.Module):
    def __init__(self, latent_size=4):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(64 * 24 * 24, latent_size)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self, latent_size=4):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 64 * 24 * 24)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 64, 24, 24)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # Output normalized to [0, 1]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class I2I(nn.Module):
    def __init__(self, latent_size=4):
        super(I2I, self).__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def train_model(self, images_pole, device, epochs=10, batch_size=64, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        dataset = torch.utils.data.TensorDataset(images_pole)  # Only images are needed
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for batch_images, in dataloader:
                batch_images = batch_images.to(device)

                optimizer.zero_grad()
                outputs = self(batch_images)

                loss = criterion(outputs, batch_images)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

    def save_model(self, path=datetime.now().strftime("%m_%d_%H_%M") + ".pth"):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"Model loaded from {path}")









# """
# reconstruct image
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from datetime import datetime


# class Encoder(nn.Module):
#     def __init__(self, latent_size=4):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=4, padding=1)
#         self.fc = nn.Linear(64 * 24 * 24, latent_size)
#         self._initialize_weights()

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)


# class Decoder(nn.Module):
#     def __init__(self, latent_size=4):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(latent_size, 64 * 24 * 24)
#         self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=4, padding=1, output_padding=1)
#         self._initialize_weights()

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.reshape(-1, 64, 24, 24)
#         x = F.relu(self.deconv1(x))
#         x = torch.sigmoid(x)  # Output normalized to [0, 1]
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)


# class I2I(nn.Module):
#     def __init__(self, latent_size=4):
#         super(I2I, self).__init__()
#         self.encoder = Encoder(latent_size)
#         self.decoder = Decoder(latent_size)

#     def forward(self, x):
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         return reconstructed

#     def train_model(self, images_pole, device, epochs=10, batch_size=64, learning_rate=0.001):
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(self.parameters(), lr=learning_rate)

#         dataset = torch.utils.data.TensorDataset(images_pole)  # Only images are needed
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         for epoch in range(epochs):
#             self.train()
#             total_loss = 0

#             for batch_images, in dataloader:
#                 batch_images = batch_images.to(device)

#                 optimizer.zero_grad()
#                 outputs = self(batch_images)

#                 loss = criterion(outputs, batch_images)
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()

#             avg_loss = total_loss / len(dataloader)
#             print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

#     def save_model(self, path=datetime.now().strftime("%m_%d_%H_%M") + ".pth"):
#         torch.save(self.state_dict(), path)
#         print(f"Model saved to {path}")

#     def load_model(self, path, device):
#         self.load_state_dict(torch.load(path, map_location=device))
#         self.to(device)
#         print(f"Model loaded from {path}")
