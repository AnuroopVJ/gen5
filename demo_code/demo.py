from gen5 import Gen5FileHandler
import torch
import json
import numpy as np
import io
from PIL import Image

gen5 = Gen5FileHandler()

batch_size = 1
channels = 3  # For RGB images
height = 64
width = 64

# Generate the initial noise tensor (often called z_T or x_T)
initial_noise_tensor = torch.randn(batch_size, channels, height, width)
binary_img_data = gen5.png_to_bytes(r"/workspaces/File_format_structure/example.png")
latent = {
    "initial_noise": initial_noise_tensor.detach().cpu().numpy()
}
gen5.file_encoder(
    should_compress=False,
    filename="converted_img.gen5",
    latent=latent,
    chunk_records=[],
    model_name="Stable Diffusion 3",
    model_version="3",
    prompt="A puppy smiling, cinematic",
    tags=["puppy","dog","smile"],
    img_binary=binary_img_data,
    convert_float16=False,
    generation_settings={
        "seed": 42,
        "steps": 20,
        "sampler": "ddim",
        "cfg_scale": 7.5,
        "scheduler": "pndm",
        "eta": 0.0,
        "guidance": "classifier-free",
        "precision": "fp16",
        "deterministic": True
    },
    hardware_info={
        "machine_name": "test_machine",
        "os": "linux",
        "cpu": "Intel",
        "cpu_cores": 8,
        "gpu": [{"name": "RTX 3090", "memory_gb": 24, "driver": "nvidia", "cuda_version": "12.1"}],
        "ram_gb": 64.0,
        "framework": "torch",
        "compute_lib": "cuda"
    }
)
print("Image Encoded Successfully...")
decoded = gen5.file_decoder(
    r"/workspaces/File_format_structure/converted_img.gen5"
)


with open("decoded_metadata.json", "w") as f:
    json.dump(decoded["metadata"], f, indent=2)

image_bytes = decoded["chunks"].get("image")
if image_bytes is not None:
    img = Image.open(io.BytesIO(image_bytes))
    img.save("decoded_image.png")

latent_data = decoded["chunks"].get("latent", [])
for i, latent_array in enumerate(latent_data):
    np.save(f"latent_{i}.npy", latent_array)
print("Decoded metadata saved to decoded_metadata.json")
if image_bytes is not None:
    print("Decoded image saved to decoded_image.png")