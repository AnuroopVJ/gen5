from .main import Gen5FileHandler
import numpy as np
import tempfile
import os
import torch
import pytest
def test_file_encode_decode():
    # Initialize handler
    gen5 = Gen5FileHandler()

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



def test_decoder_rejects_bad_magic(tmp_path):
    gen5 = Gen5FileHandler()
    filename = tmp_path / "test.gen5"

    gen5.file_encoder(
        filename=str(filename),
        latent={"l": np.zeros((1,4,64,64), dtype=np.float32)},
        chunk_records=[],
        model_name="X",
        model_version="1",
        prompt="p",
        tags=[],
        img_binary=b"img"
    )

    # Corrupt header
    with open(filename, "r+b") as f:
        f.write(b"XXXX")

    with pytest.raises(Exception):
        gen5.file_decoder(str(filename))


