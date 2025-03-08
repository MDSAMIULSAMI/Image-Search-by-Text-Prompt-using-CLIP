# Image Search by Text Prompt using CLIP
![screenshot_1741423960690.png](<https://media-hosting.imagekit.io//e4818e4f2ef64718/screenshot_1741423960690.png?Expires=1836031962&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=QOFHcLkRFC1XsnZ5rPNwbRvdbuFa5rlrHhp5H4Pxpk4jWUigvvJK1VieKRZpN-lXwiqqxgFGgwKvBxZAtm3~B8i3fCoX2K5lHQUhsOXNSPjKw2h0LmEXV57RSC-pcQMmAz43sIxQKIUSDjwx3AV4355ewXC1Ye1bmA7B01Kgp-KmssLIs0MTwJSL7-HC1Uff0XIQHrgY6y9xwK6-6BRoEaUKM9vH-0r93rFcUlFuhL89NPx0gU-pZFjPBT3zH6XsTB6qH~nw3eYcRxlbrWZckvey8m01IfCn0H1yP0OyrkSLFcKawYMqtjvFZqE~EftQdnqVR1bhT9oSAWl84Tk9cQ__>)
## Overview

This project uses OpenAI's CLIP (Contrastive Language-Image Pretraining) model to find images in a specified folder that are most similar to a given text prompt. The text prompt is encoded using CLIP's text encoder, and images are compared to the prompt using cosine similarity to rank the images based on their relevance to the prompt.

## Features

- Load images from a folder (supports `.png`, `.jpg`, `.jpeg`, `.webp`).
- Use CLIP to encode and compare text and image features.
- Display the top `k` most similar images based on cosine similarity between the prompt and image features.

## Prerequisites

### System Requirements

- **Operating System**: Cross-platform (Linux, MacOS, Windows)
- **Hardware**: Preferably a machine with a GPU (especially for better performance on large datasets)

### Software Requirements

- Python 3.7 or higher
- PyTorch (supports MPS for ARM-based Macs like M1/M2/M3)
- OpenAI CLIP model
- Other dependencies for image loading and visualization

## Initial Environment Setup

1. **Clone the Repository**:

   Clone the repository containing the project files.

   ```bash
   git clone https://github.com/MDSAMIULSAMI/Image-Search-by-Text-Prompt-using-CLIP.git
   cd Image-Search-by-Text-Prompt-using-CLIP
   
2.	**Create a Virtual Environment (optional but recommended)**:
   
    It’s recommended to create a virtual environment to avoid conflicts with other projects.
  	
  	```python copy
     python3 -m venv venv
     source venv/bin/activate  # For Mac/Linux
     venv\Scripts\activate  # For Windows

3. **Install Dependencies**:
   Install the required libraries using pip. You can use the requirements.txt file for this.
   ```python copy
    pip install -r requirements.txt

## Usage

1.	**Prepare Image Folder**:
   Place the images you want to search through in a folder.
  	
2.	**Run the Script**:
   Execute the script to input a text prompt and find the top matching images.

3.	**Input a Prompt**:
  The script will prompt you to enter a text query. Based on your query, it will show you the top images that match the query.

## Code Explanation

	•	load_images(folder_path): Loads all the images from a given folder. It checks for supported image file formats (.png, .jpg, .jpeg, .webp) and returns a list of image paths and the corresponding PIL image objects.
	•	find_images_by_prompt(folder_path, prompt, top_k=5): Takes a folder path and a text prompt, encodes the prompt using the CLIP model, compares the similarity of each image in the folder to the prompt, and returns the top k matching images.
	•	show_images(results): Displays the top matching images using matplotlib.

 ## Notes
 	•	This implementation uses the MPS backend for macOS (Apple Silicon) when available, or defaults to CPU for other systems.
	•	The find_images_by_prompt function returns the top k results, where k can be adjusted as needed.
	•	You can modify the folder_path in the code to point to any directory containing your images.
 ## License
 This project is licensed under the MIT License.
 This should cover the full setup process, including creating a virtual environment and installing the dependencies.
