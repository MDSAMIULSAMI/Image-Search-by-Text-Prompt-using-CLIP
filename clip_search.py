import os
import torch
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog

# Use Metal (MPS) on ARM64 (M1-M4), otherwise fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and preprocess using open_clip
model_transform = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
model = model_transform[0]  # Access the model
transform = model_transform[1]  # Access the transform

model.to(device)

def load_images(folder_path):
    """Loads images from a folder and returns a list of (image_path, PIL Image)"""
    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_path = os.path.join(folder_path, file)
            try:
                img = Image.open(image_path).convert("RGB")
                images.append((image_path, img))
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    return images

def find_images_by_prompt(folder_path, prompt, top_k=5):
    """Finds images most similar to the given prompt using open_clip"""
    images = load_images(folder_path)
    
    if not images:
        print("No images found in the folder.")
        return []

    # Encode the text prompt
    text_tokens = open_clip.tokenize([prompt]).to(device)
    text_features = model.encode_text(text_tokens).detach().cpu().numpy()

    similarities = []
    
    for img_path, img in images:
        # Preprocess and encode image
        img_tensor = transform(img).unsqueeze(0).to(device)
        img_features = model.encode_image(img_tensor).detach().cpu().numpy()

        # Compute cosine similarity
        similarity = (text_features @ img_features.T).item()
        similarities.append((img_path, similarity))

    # Sort images by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def show_images(results):
    """Displays the top matching images"""
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    
    for i, (img_path, score) in enumerate(results):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Score: {score:.2f}")
        axes[i].axis("off")
    
    plt.show()

if __name__ == "__main__":
    # Create hidden root window
    root = tk.Tk()
    root.withdraw()
    
    # Select folder dialog
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        print("No folder selected. Exiting...")
        root.destroy()
        exit()
    
    # Prompt input dialog
    prompt = simpledialog.askstring("Search Prompt", "Enter your search prompt:", parent=root)
    root.destroy()
    
    if not prompt:
        print("No prompt entered. Exiting...")
        exit()

    # Process and show results
    results = find_images_by_prompt(folder_path, prompt)
    
    if results:
        print("Matching images found:")
        show_images(results)
    else:
        print("No matching images found.")