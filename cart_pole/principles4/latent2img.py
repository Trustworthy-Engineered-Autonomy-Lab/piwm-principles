import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime


class L2IDecoder(nn.Module):
    def __init__(self, state_size=4, m_size=3,output_channel=3):
        super(L2IDecoder, self).__init__()
        self.fc_dec = nn.Linear(state_size, m_size * 24 * 24)
        self.dec_conv1 =nn.ConvTranspose2d(m_size, 3, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 =nn.ConvTranspose2d(3, 1, kernel_size=4, stride=2, padding=1)
        self.m_size=m_size

        self._initialize_weights()
        
    def forward(self, states):
        batch_size = states.size(0)
        x = self.fc_dec(states)
        x = x.view(batch_size, self.m_size, 24, 24)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_conv2(x)
        x = torch.sigmoid(x)
        x = x.view(batch_size, 96,96,1)
        velocity = states[:, 1].unsqueeze(1)
        return x ,velocity

    def train_model(self, img, states, device, epochs=10, batch_size=64, learning_rate=0.001, val_split=0.1):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Split data into training and validation sets
        total_size = len(img)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            torch.utils.data.TensorDataset(img, states), [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            # Training Phase
            self.train()
            total_train_loss = 0
            best_loss=114.0
            for batch_images, batch_states in train_loader:
                batch_images, batch_states = batch_images.to(device), batch_states.to(device)

                optimizer.zero_grad()
                outputs,_ = self(batch_states)

                loss = criterion(outputs, batch_images)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation Phase
            self.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch_images, batch_states in val_loader:
                    batch_images, batch_states = batch_images.to(device), batch_states.to(device)

                    outputs,_ = self(batch_states)
                    loss = criterion(outputs, batch_images)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            best_loss=min(best_loss,avg_val_loss)

            #print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"msize = {self.m_size}, best loss = {best_loss}")
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

