import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime


class L2PDecoder(nn.Module):
    def __init__(self, state_size=4, m_channel=16, output_channel=3):
        super(L2PDecoder, self).__init__()
        self.m_channel = m_channel
        self.fc = nn.Linear(state_size, 3 * m_channel * 24 * 24)
        self.conv = nn.ModuleList([
            nn.ConvTranspose2d(m_channel, output_channel, kernel_size=5, stride=4, padding=1, output_padding=1)
            for _ in range(3)
        ])

        self._initialize_weights()

    def forward(self, states):
        batch_size = states.size(0)
        x = self.fc(states)
        x = x.view(batch_size, 3, self.m_channel, 24, 24)

        outputs = []
        for i in range(3):
            out = F.relu(self.conv[i](x[:, i]))
            outputs.append(out.unsqueeze(1))
        x = torch.cat(outputs, dim=1)
        x = torch.sigmoid(x)
        return x
    
    def train_model(self, images_pole, states, device, epochs=10, batch_size=64, learning_rate=0.001, val_split=0.1, lambda_size=0.0):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        total_size = len(images_pole)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size

        train_dataset = torch.utils.data.TensorDataset(images_pole[:train_size], states[:train_size])
        val_dataset = torch.utils.data.TensorDataset(images_pole[val_size:][:, 0, ::], states[val_size:])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            # Training Phase
            self.train()
            total_train_loss = 0
            for batch_images, batch_states in train_loader:
                real_images, batch_states = batch_images.to(device), batch_states.to(device)

                optimizer.zero_grad()
                outputs = self(batch_states)
                
                reconstructed_images = torch.chunk(outputs, 3, dim=1)
                reconstructed_images = reconstructed_images[0]+reconstructed_images[1]+reconstructed_images[2] - 2
                reconstructed_images = torch.clamp(reconstructed_images, max=1.0).view(-1, 3, 96, 96)
                    
                loss = (1-lambda_size)*criterion(outputs, real_images[:, 1:, ::]) + lambda_size * criterion(reconstructed_images, real_images[:, 0, ::].view(-1, 3, 96, 96))
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            #print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

        self.eval()
        total_val_loss = 0
        best_loss = 114.0
        with torch.no_grad():
            for batch_images, batch_states in val_loader:
                real_images, batch_states = batch_images.to(device).view(-1, 3, 96, 96), batch_states.to(
                    device)

                outputs = torch.chunk(self(batch_states), 3, dim=1)

                reconstructed_images = outputs[0]+outputs[1]+outputs[2]-2
                reconstructed_images = torch.clamp(reconstructed_images, max=1.0).view(-1, 3, 96, 96)

                loss = criterion(reconstructed_images, real_images)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            best_loss = min(best_loss, avg_val_loss)

        print(f"msize = {self.m_channel}, best loss = {best_loss}")
        return best_loss

    def save_model(self, path=datetime.now().strftime("%m_%d_%H_%M") + ".pth"):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"Model loaded from {path}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

