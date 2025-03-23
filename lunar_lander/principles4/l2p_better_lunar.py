import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_msssim import ssim

class L2P_better_Decoder(nn.Module):
    def __init__(self, state_size=8, m_channel=3, output_channel=3):
        super(L2P_better_Decoder, self).__init__()
        self.m_channel = m_channel
        self.fc = nn.Linear(state_size, 3 * m_channel * 20 * 30)
        self.fc2=nn.Linear(state_size,64)
        self.fc3=nn.Linear(64,128)
        self.fc4=nn.Linear(128,m_channel * 20 * 30)
        self.conv1 = nn.ModuleList([
            nn.ConvTranspose2d(m_channel, m_channel, kernel_size=4, stride=2, padding=1)
            for _ in range(3)
        ])

        self.conv2 = nn.ModuleList([
            nn.ConvTranspose2d(m_channel, output_channel, kernel_size=4, stride=2, padding=1)
            for _ in range(3)
        ])
        
        self.conv3 = nn.ConvTranspose2d(m_channel, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(128, output_channel, kernel_size=4, stride=2, padding=1)
     
        self._initialize_weights()

    def forward(self, states):
        batch_size = states.size(0)
        x = self.fc(states)
        x = x.view(batch_size, 3, self.m_channel, 20, 30)

        outputs = []
        for i in range(3):
            out = F.relu(self.conv1[i](x[:, i]))
            out = self.conv2[i](out)
            outputs.append(out.unsqueeze(1))
            
        x = F.relu(self.fc2(states))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(batch_size,self.m_channel, 20, 30)
        out = self.conv3(x)
        out = self.conv4(out)
        outputs.append(out.unsqueeze(1))

        x = torch.cat(outputs, dim=1)
        x = torch.sigmoid(x)
        img = x.sum(dim=1)
        return x, img

    def train_model(self, part, combined, states, device, epochs=10, batch_size=64, learning_rate=0.001, lambda_size=0.0):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.TensorDataset(states, part, combined)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')  # Initialize best_loss

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0
            total_l2 = 0
            total_ssim_loss = 0
            for batch_states, part_images, combined_images in train_loader:
                part_images, combined_images, batch_states = (
                    part_images.to(device),
                    combined_images.to(device),
                    batch_states.to(device),
                )
                
                optimizer.zero_grad()
                outputs , img = self(batch_states)
                
                l1 = criterion(outputs, part_images)
                l2 = criterion(img, combined_images)
                loss = (1 - lambda_size) * l1 + lambda_size * l2
                loss.backward()
                optimizer.step()
                
                # Compute SSIM loss (only for debugging, not included in training loss)
                ssim_loss = 1 - ssim(img, combined_images, data_range=1.0, size_average=True)
                
                total_train_loss += loss.item()
                total_l2 += l2.item()
                total_ssim_loss += ssim_loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            avg_l2 = total_l2 / len(train_loader)
            avg_ssim_loss = total_ssim_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - img Loss: {avg_l2:.4f} - SSIM Loss: {avg_ssim_loss:.4f}")

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss

        return best_loss

    def save_model(self, path=None):
        if path is None:
            path = datetime.now().strftime("%m_%d_%H_%M") + ".pth"
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"Model loaded from {path}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
