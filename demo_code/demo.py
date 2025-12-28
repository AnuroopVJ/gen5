from main import Gen5FileHandler
import torch
import json
import numpy as np
import io
from PIL import Image
gen5 = Gen5FileHandler()
#filename (str): Output GEN5 filename. [REQUIRED]
# latent (Dict[str, np.ndarray]): Dictionary of latent arrays.
# chunk_records (list): List to append chunk records to.
# model_name (str): Name of the model.
# model_version (str): Version of the model.
# prompt (str): Prompt used for generation.
# tags (list): List of tags.
# img_binary (bytes): Binary image data.
# Returns:
# dict: Dictionary with header, latent chunks, metadata, and image chunk.

batch_size = 1
channels = 3  # For RGB images
height = 64
width = 64

# Generate the initial noise tensor (often called z_T or x_T)
initial_noise_tensor = torch.randn(batch_size, channels, height, width)
binary_img_data = gen5.png_to_bytes(r"C:\Users\neela\Desktop\Miscellaneous\image file format - .gen5\gen5\src\gen5\example.png")
latent = {
    "initial_noise": initial_noise_tensor.detach().cpu().numpy()
}
gen5.file_encoder(
    filename="converted_img.gen5",
    latent=latent,
    chunk_records=[],
    model_name="Stable Diffusion 3",
    model_version="3",
    prompt="A puppy smiling, cinematic",
    tags=["puppy","dog","smile"],
    img_binary=binary_img_data,
)
print("Image Encoded Successfully...")
decoded = gen5.file_decoder(
    r"C:\Users\neela\Desktop\Miscellaneous\image file format - .gen5\gen5\src\gen5\converted_img.gen5"
)


with open("decoded_metadata.json", "w") as f:
    json.dump(decoded["metadata"], f, indent=2)

image_bytes = decoded["chunks"].get("image")
if image_bytes is not None:
    img = Image.open(io.BytesIO(image_bytes))
    img.save("decoded_image.png")

for i, latent_array in enumerate(decoded["chunks"]["latent"]):
    latent_data = decoded["chunks"].get("latent", [])
    for i, latent_array in enumerate(latent_data):
        np.save(f"latent_{i}.npy", latent_array)
    
print("Decoded metadata saved to decoded_metadata.json")
if image_bytes is not None:
    print("Decoded image saved to decoded_image.png")