Gen5 is a binary container format aimed at increased reproducibility for AI-generated images. It enables the storage of several key pieces of information, such as :

## 1. Environment & Provenance Tracking
The GEN5 file format has a dedicated chunk that captures Runtime Environment Details at the time of generation.
This includes:
  - Operating System and Machine Identifiers
  - CPU/GPU models, core counts, and memory
  - Deep learning framework (e.g., PyTorch) and compute backend (e.g., CUDA)
  - Driver and library versions (e.g., CUDA version, NVIDIA driver)
This is further utilised to warn the user of environment mismatches.
  ### Environment Chunk (ENVC)
  GEN5 includes an optional but strongly recommended environment chunk (ENVC) that captures a verifiable snapshot of the software stack used during generation. Unlike generic metadata, this chunk:

  Is structured as a list of canonicalized components (e.g., torch, python, cuda, gpu)
  Records each component’s name, version, and a SHA-256 digest of its canonical string
  Is compressed and stored as a standalone binary chunk with its own offset and hash in the chunk table
  Enables integrity verification and drift detection across environments
  This chunk is not embedded in the metadata JSON, but referenced via the chunks array in the decoded output (type "ENVC"), ensuring it remains tamper-evident and tooling-friendly.

  HOW TO USE?
  It is automatically populated and stored for you!
  When environment mismatch is detected, a warning is issued.
  Example Warning for gpu mismatch:
  ```
  UserWarning: Environment component 'gpu' differs:
  File: name=gpu;model=NVIDIA A100;driver=535.129.01
  Current: name=gpu;model=NVIDIA GeForce RTX 4090;driver=550.40.07
  ```

## 2. Latent Tensor Storage
GEN5 natively supports the storing of latent representations (ie, diffusion model latents, VAE encodings) alongside the generated images. These are serialized as one or more 'LATN' chunks.
These are stored in their native memory layout (as provided by the user). For PyTorch-generated latents, this is typically NCHW. And for TensorFlow, it is going to be NHWC. The format does not enforce or convert layout, insted it preserves the exact shape and byte representation as provided by the user.

  HOW TO STORE?
  Example:
  ```python
  gen5.file_encoder(
    filename="my_ai_art.gen5",
    latent={
        "initial_noise": initial_noise.numpy(),
        "final_latent": final_latent.numpy()
    },
    chunk_records=[],
    model_name="Stable Diffusion 3",
    model_version="3.0",
    prompt="A cyberpunk cat wearing neon goggles, cinematic lighting",
    tags=["cat", "cyberpunk", "neon"],
    img_binary=img_bytes,
    convert_float16=True,       # store latents in float16 to save space (optional)
    should_compress=True,       #SEE THE CRITICAL WARNING BELOW
    generation_settings={
        "seed": 1337,
        "steps": 30,
        "sampler": "euler_ancestral",
        "cfg_scale": 7.0,
        "scheduler": "karras",
        "precision": "fp16"
    },
    hardware_info={
        "machine_name": "desktop-alpha",
        "os": "linux",
        "cpu": "AMD Ryzen 9 7950X",
        "cpu_cores": 16,
        "gpu": [
            {
                "name": "NVIDIA RTX 4090",
                "memory_gb": 24,
                "driver": "550.54.14",
                "cuda_version": "12.4"
            }
        ],
        "ram_gb": 128.0,
        "framework": "torch",
        "compute_lib": "cuda"
    }
)
```
!!! danger
    **CRITICAL WARNING**:
    The ```should _compress``` is for compressinng the latent chunks. For High Entropy Tensors: FALSE, Low Entropy Tensors: TRUE

Extra Metadata:
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
!!! danger
    **DISCLAIMER**:
    The encoder expects **NumPy arrays**.  
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
    convert_float16=False, # whether to convert input tensors to float16
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
# now to save the metadata
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
