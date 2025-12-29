from pathlib import Path
from ..gen5.main import Gen5FileHandler, Gen5ChunkError, Gen5MetadataError, Gen5ImageError, Gen5CorruptHeader
import numpy as np
import tempfile
import os
import torch
import pytest
import json.decoder
import zstandard as zstd
import copy

gen5 = Gen5FileHandler()
def test_file_encode_decode():
    # Initialize handler


    batch_size = 1
    channels = 4
    height = 64
    width = 64
    initial_noise_latent = {
        "latent_1": torch.randn(batch_size, channels, height, width, dtype=torch.float32).numpy()
    }
    chunk_records = []
    with open('example.png', 'rb') as f:
        img_bytes = f.read()




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
    with open('example.png', 'rb') as f:
        img_bytes = f.read()
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

def test_corrupt_metadata():
    valid_manifest = gen5.build_manifest(
        version_major=1,
        version_minor=0,
        model_name="TestModel",
        model_version="v1",
        prompt="Hello world",
        tags=["test"],
        chunk_records=[],
        generation_settings={
            "seed": 0,
            "steps": 0,
            "sampler": "",
            "cfg_scale": 0.0,
            "scheduler": "",
            "eta": 0.0,
            "guidance": "",
            "precision": "fp16",
            "deterministic": True
        },
        hardware_info={
            "machine_name": "",
            "os": "",
            "cpu": "",
            "cpu_cores": 16,
            "gpu": [],
            "ram_gb": 32.0,
            "framework": "torch",
            "compute_lib": ""
        }
    )

    corrupt_manifest = copy.deepcopy(valid_manifest)
    del corrupt_manifest['gen5_metadata']['file_info']['magic']
    corrupt_manifest['gen5_metadata']['file_info']['version_major'] = "not an int"
    corrupt_bytes = gen5.metadata_compressor(corrupt_manifest)

    with pytest.raises((Gen5MetadataError, zstd.ZstdError, json.JSONDecodeError)):
        gen5.metadata_validator(corrupt_bytes)


def test_corrupt_chunk():
    with open('example.png', 'rb') as f:
        img_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".gen5", delete=False) as tmp_file:
        filename = tmp_file.name
        latent = {
            "latent_1": torch.randn(1, 4, 64, 64, dtype=torch.float32).numpy()
        }
        chunk_records = []
        gen5.file_encoder(
    filename=filename,
    latent=latent,
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

        # Corrupt chunk
        with open(filename, "r+b") as f:
            f.seek(gen5.HEADER_SIZE)
            f.write(b"\x00\x01\x02")

        with pytest.raises(Gen5ChunkError):
            gen5.file_decoder(filename)

def test_corrupt_image():
    with open('example.png', 'rb') as f:
        img_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".gen5", delete=False) as tmp_file:
        filename = tmp_file.name
        latent = {
            "latent_1": torch.randn(1, 4, 64, 64, dtype=torch.float32).numpy()
        }
        chunk_records = []
        img_bytes = img_bytes[:10]

        with pytest.raises(Gen5ImageError):
            gen5.file_encoder(
    filename=filename,
    latent=latent,
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

        
def test_corrupt_image_decoder():
    with open('example.png', 'rb') as f:
        img_bytes = f.read()
    latent = {
        "latent_1": torch.randn(1, 4, 64, 64, dtype=torch.float32).numpy()
    }
    chunk_records = []

    with tempfile.NamedTemporaryFile(suffix=".gen5", delete=False) as tmp_file:
        filename = tmp_file.name

    result = gen5.file_encoder(
    filename=filename,
    latent=latent,
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
    metadata = gen5.metadata_parser(result["metadata_chunk"])
    with open(filename, "r+b") as f:
        for rec in metadata["gen5_metadata"]["chunks"]:
            if rec["type"] == "DATA":          #image chunk
                f.seek(rec["offset"])
                f.write(b"\xFF" * rec["compressed_size"])
                break

    with pytest.raises(Gen5ImageError):
        gen5.file_decoder(filename)
