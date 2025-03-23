import torch
from L2P import L2PDecoder
from data_utils import divided_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

npz_file = './combined_data.npz'
images_pole, states = divided_data(npz_file,device)

# import matplotlib.pyplot as plt
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# plt.imshow(images_pole[0][3].cpu().permute(1, 2, 0).numpy())
# plt.show()

model = L2PDecoder().to(device)
model.train_model(images_pole, states, device, epochs=20, batch_size=64, learning_rate=0.001)
model.save_model()
