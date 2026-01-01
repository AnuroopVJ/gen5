
from dataclasses import dataclass, asdict
from typing import List

class Gen5DecodeError(Exception):
    def __init__(self, message: str):
        super().__init__(f"Gen5 Decode Error: {message}")

class Gen5CorruptHeader(Gen5DecodeError):
    def __init__(self, message: str):
        super().__init__(f"Corrupt header: {message}")

class Gen5MetadataError(Gen5DecodeError):
    def __init__(self, message: str):
        super().__init__(f"Corrupt Metadata: {message}")

class Gen5ChunkError(Gen5DecodeError):
    def __init__(self, message: str):
        super().__init__(f"Corrupt Chunk: {message}")

class Gen5LatentError(Gen5ChunkError):
    def __init__(self, message: str):
        super().__init__(f"Corrupt Latent: {message}")

class Gen5ImageError(Gen5ChunkError):
    def __init__(self, message: str):
        super().__init__(f"Corrupt Image: {message}")

class Gen5EnvChunkError(Gen5ChunkError):
    def __init__(self, message: str):
        super().__init__(f"Corrupt Environment Chunk: {message}")
