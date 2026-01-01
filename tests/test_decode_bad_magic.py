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
    img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 255))  #justa  red square
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_decoder_bad_magic(tmp_path: Path):

    filename = tmp_path / "test.gen5"
    batch_size = 1
    channels = 4
    height = 64
    width = 64
    initial_noise_latent = {
        "latent_1": torch.randn(batch_size, channels, height, width, dtype=torch.float32).numpy()
    }
    chunk_records = []
    img_bytes = create_test_image()
    gen5.file_encoder(
        filename=str(filename),
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


    # Corrupt header
    with open(filename, "r+b") as f:
        f.write(b"XXXX")

    with pytest.raises(Gen5CorruptHeader):
        gen5.file_decoder(str(filename))