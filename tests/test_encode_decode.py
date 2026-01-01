from pathlib import Path
import numpy as np
import tempfile
import os
import torch
import pytest
import json.decoder
import zstandard as zstd
import copy
from gen5 import Gen5FileHandler
from gen5.core.exceptions import Gen5CorruptHeader, Gen5MetadataError, Gen5ImageError, Gen5ChunkError
from gen5.chunks.metadata import Gen5Metadata
from gen5.core.header import header_parse
from PIL import Image
import io
gen5 = Gen5FileHandler()

def create_test_image():
    # Create a small deterministic RGB or RGBA image
    img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 255))  # red square
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_file_encode_decode():
    batch_size = 1
    channels = 4
    height = 64
    width = 64
    initial_noise_latent = {
        "latent_1": torch.randn(batch_size, channels, height, width, dtype=torch.float32).numpy()
    }
    chunk_records = []
    img_bytes = create_test_image()

    with tempfile.NamedTemporaryFile(suffix=".gen5", delete=False) as tmp_file:
        filename = tmp_file.name

    try:
        gen5.file_encoder(
    filename=filename,
    latent=initial_noise_latent,
    chunk_records=chunk_records,
    model_name="TestModel",
    model_version="1.0",
    prompt="Test prompt",
    tags=["test"],
    img_binary=img_bytes,
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
        decoded = gen5.file_decoder(filename)

        
        header = decoded["header"]
        assert header["magic"] == b"GEN5"
        assert header["version_major"] == 1


        decoded_latent = decoded["chunks"]["latent"][0]
        np.testing.assert_array_equal(decoded_latent, initial_noise_latent["latent_1"])
        assert decoded["chunks"]["image"] == img_bytes
        metadata = decoded["metadata"]["gen5_metadata"]["model_info"]
        assert metadata["model_name"] == "TestModel"
        assert metadata["prompt"] == "Test prompt"

    finally:
        os.remove(filename)