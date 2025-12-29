## Overview

Gen5 is a binary container format aimed at increased reproducibility for AI-generated images. It enables the storage of several key pieces of information, such as :

- The initial noise tensor (which usually changes every run)
- Model name and version
- Prompt  
- Tags  
- Hardware information  
- Generation settings
(may include sampler-specific parameters)

The Initial noise tensor can be fed back in while using a model (local ones) to obtain similar results.
This has proven to be capable of producing extremely similar images. Although we use a random seed integer value, the usage of the real tensor provides increased reproducibility.


## Installation
Just pip install the package!
```bash
pip install gen5
```
## Usage
import the classes
```python
from gen5.main import Gen5FileHandler
```
First you need to instantiate the Gen5FileHandler class.
```python
gen5 = Gen5FileHandler()
```

# Encoding

    DISCLAIMER:
    The encoder expects NumPy arrays.  
    If you use PyTorch tensors, convert them with `.detach().cpu().numpy()`.

```python
from gen5.main import Gen5FileHandler

gen5 = Gen5FileHandler()
initial_noise_tensor = torch.randn(batch_size, channels, height, width)
latent = {
    "initial_noise": initial_noise_tensor.detach().cpu().numpy() #The encoder expects numpy array not a torch tensor object
}
binary_img_data = gen5.png_to_bytes(r'path/to/image.png') # use the helper function to convert image to bytes

gen5.file_encoder(
    filename="encoded_img.gen5", # The .gen5 extension is required!
    latent=latent,# initial latent noise
    chunk_records=[],
    model_name="Stable Diffusion 3",
    model_version="3", # Model Version
    prompt="A puppy smiling, cinematic",
    tags=["puppy","dog","smile"],
    img_binary=binary_img_data,
    convert_float16=False, # whether to convert to float16 (enable if input tensors is in float32)
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
        "cpu_cores": 8, # minimum 1
        "gpu": [{"name": "RTX 3090", "memory_gb": 24, "driver": "nvidia", "cuda_version": "12.1"}],
        "ram_gb": 64.0,
        "framework": "torch",
        "compute_lib": "cuda"
    }
)
```

# Decoding
```python
decoded = gen5.file_decoder(filename)
# Now to save the metadata
metadata = decoded["metadata"]["gen5_metadata"]

# to just get specific metadata blocks
model_info = decoded["metadata"]["gen5_metadata"]["model_info"]

# to save decoded metadata to a json file
with open("decoded_metadata.json", "w") as f:
    json.dump(decoded["metadata"], f, indent=2)

# to save just the image_binary as png
image_bytes = decoded["chunks"].get("image")
if image_bytes is not None:
    img = Image.open(io.BytesIO(image_bytes))
    img.save("decoded_image.png")
```
