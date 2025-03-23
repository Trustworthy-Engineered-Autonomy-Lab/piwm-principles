import torch
from latent2img import L2IDecoder
from img2img import I2I
from data_utils import preprocess_data
import matplotlib.pyplot as plt

def test_2I(model, input_data, device, title, save_path=None, real_img=None, num_samples=10):
    model.eval()
    
    
    indices = torch.randint(0, input_data.size(0), (num_samples,))
    test_data = input_data[indices].to(device)
    
    
    real_img = test_data if real_img is None else real_img[indices].to(device)
    
    
    with torch.no_grad():
        reconstructed_images = model(test_data)
    
    
    plt.figure(figsize=(10, 2 * num_samples))
    plt.suptitle(title)
    plt.axis('off')
    for i in range(num_samples):
        # Original Image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(real_img[i].cpu().numpy())
        plt.title("Original Image")
        plt.axis('off')

        # Reconstructed Image
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(reconstructed_images[i].cpu().numpy())
        plt.title("Reconstructed Image")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    if save_path!=None:
        plt.savefig(save_path, dpi=300)
    
    
def test_2P(model, input_data, device, title, save_path=None, real_img=None, num_samples=10):
    model.eval()
    
    
    indices = torch.randint(0, input_data.size(0), (num_samples,))
    test_data = input_data[indices].to(device)
    
    
    real_img = test_data if real_img is None else real_img[indices].to(device)
    
    
    with torch.no_grad():
        outputs ,reconstructed_images= model(test_data)
    
    plt.figure(figsize=(10, 2 * num_samples))
    plt.suptitle(title)
    plt.axis('off')
    for i in range(num_samples):
        # Original Image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(real_img[i].cpu().permute(1, 2, 0).numpy())  # (C, H, W) -> (H, W, C)
        plt.title("Original Image")
        plt.axis('off')

        # Reconstructed Image
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(reconstructed_images[i].cpu().permute(1, 2, 0).numpy())
        plt.title("Reconstructed Image")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    if save_path!=None:
        plt.savefig(save_path, dpi=300)    
        

def test_2PS(model, input_data, device, title, save_path=None, real_img=None, num_samples=10):
    model.eval()

    indices = torch.randint(0, input_data.size(0), (num_samples,))
    test_data = input_data[indices].to(device)

    real_img = test_data if real_img is None else real_img[indices].to(device)

    with torch.no_grad():
        outputs,img =model(test_data)

    # plt.figure(figsize=(10, 1 * num_samples))
    plt.suptitle(title)
    
    for i in range(num_samples):
        # Original Image
        
        plt.figure(figsize=(18, 6))
        plt.subplot(num_samples, 5, 5 * i + 1)
        plt.imshow(real_img[i].cpu().permute(1, 2, 0).numpy())  # (C, H, W) -> (H, W, C)
        if i==0:
            plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(num_samples, 5, 5 * i + 2)
        plt.imshow(img[i].cpu().permute(1, 2, 0).numpy())  # (C, H, W) -> (H, W, C)
        if i==0:
            plt.title("Reconstructed Image")
        plt.axis('off')
        
        
        # Ensure we have exactly 3 outputs
        for j in range(3):
            plt.subplot(num_samples, 5, 5 * i + j + 3)  # Adjusting subplot indexing
            plt.imshow(outputs[i][j].view(3,96,96).cpu().permute(1, 2, 0).numpy())  # Correct output access
            if i==0:
                plt.title(f"Reconstructed {j + 1}")
            plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    
    plt.show()


def test_combined(model_2I, model_2P, input_data_2I, input_data_2P, device, title, save_path=None, real_img_2I=None, real_img_2P=None, num_samples=5):
    model_2I.eval()
    model_2P.eval()
    
    indices_2I = torch.randint(0, input_data_2I.size(0), (num_samples,))
    test_data_2I = input_data_2I[indices_2I].to(device)
    real_img_2I = test_data_2I if real_img_2I is None else real_img_2I[indices_2I].to(device)
    
    
    test_data_2P = input_data_2P[indices_2I].to(device)
    real_img_2P = test_data_2P if real_img_2P is None else real_img_2P[indices_2I].to(device)
    
    with torch.no_grad():
        reconstructed_images_2I = model_2I(test_data_2I)
        
        outputs ,reconstructed_images_2P= model_2P(test_data_2P)
    
    plt.figure(figsize=(10, 3 * num_samples))
    plt.suptitle(title)
    
    for i in range(num_samples):
        # Original Image for 2I
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(real_img_2I[i].cpu().permute(1, 2, 0).numpy())
        plt.title("Original Image")
        plt.axis('off')
        
        # Reconstructed Image from model_2I
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(reconstructed_images_2I[i].cpu().permute(1, 2, 0).numpy())
        plt.title("one model(baseline)")
        plt.axis('off')
        

        
        # Reconstructed Image from model_2P
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(reconstructed_images_2P[i].cpu().permute(1, 2, 0).numpy())
        plt.title("Three simpler models")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
