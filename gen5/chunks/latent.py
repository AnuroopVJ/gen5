import struct
import numpy as np
from zstandard import ZstdCompressor, ZstdDecompressor
from gen5.core.exceptions import Gen5LatentError, Gen5DecodeError
import hashlib
from typing import Dict, Iterable, List


class Gen5Latent:
    def __init__(self):
        pass
    def dtype_from_flags(self, flags: bytes) -> np.dtype:
            """
            Convert 4-byte chunk flags to NumPy dtype.
            Handles null-padding and validates known types.
            """
            flag_str = flags.rstrip(b'\x00').decode('ascii', errors='ignore').strip()
            if flag_str == "F16":
                return np.dtype('float16')
            if flag_str == "F32":
                return np.dtype('float32')
            raise ValueError(f"Unknown dtype flag: {flag_str!r} (raw: {flags!r})")

    def make_lazy_latent_loader(self, filename: str, chunk_record: dict):
        """Return a callable that loads the latent array on demand."""
        loaded_array = None

        def load():
            nonlocal loaded_array
            if loaded_array is None:
                with open(filename, "rb") as f:
                    try:
                        offset = chunk_record['offset']
                        size = chunk_record['compressed_size']
                        
                        shape = tuple(chunk_record['extra']['shape'])
                        compressed = chunk_record.get("compressed", True)
                        f.seek(0, 2)
                        file_size = f.tell()
                        if offset + size > file_size:
                            raise Gen5LatentError(
                                f"Chunk at offset {offset} with size {size} exceeds file bounds ({file_size})."
                            )
                        f.seek(offset)
                        compressed_chunk = f.read(size)
                        loaded_array = self.latent_parser(compressed_chunk, shape, compressed)
                    except Exception as e:
                        raise Gen5LatentError(f"Failed to load the latent chunk : {chunk_record} | {e}")

            return loaded_array
        return load

    def iter_lazy_latents(self, filename: str, chunk_records: list):
        """Yield callables for each latent chunk for lazy loading."""
        for record in chunk_records:
            if record['type'] == "LATN":
                yield self.make_lazy_latent_loader(filename, record)

    def chunk_packer_binary(self, chunk_type:bytes, chunk_flags: bytes, data:bytes) -> bytes:
            """
            Pack a binary chunk with header and compress it using Zstandard.
            Args:
                chunk_type (bytes): 4-byte chunk type identifier.
                chunk_flags (bytes): 4-byte chunk flags.
                data (bytes): Payload data.
            Returns:
                bytes: Compressed chunk data.
            """
            chunk_size = len(data)
            #chunk maker
            chunk_header = struct.pack('<4s 4s I', chunk_type, chunk_flags, chunk_size)
            #compress chunk
            chunk = chunk_header + data
            compressed_chunk = ZstdCompressor().compress(chunk)
            return compressed_chunk

    def latent_shape_validator(self, latent_array: np.ndarray, expected_dims: int = 4, 
                            max_dimension_size: int = 8192) -> bool:
        """Validate latent array shape and dimensions.
        
        Args:
            latent_array (np.ndarray): Latent array to validate.
            expected_dims (int): Expected number of dimensions (default: 4 for NCHW).
            max_dimension_size (int): Maximum size for any single dimension.
            
        Returns:
            bool: True if valid.
            
        Raises:
            Gen5LatentError: If validation fails.
        """
        if not isinstance(latent_array, np.ndarray):
            raise Gen5LatentError(f"Latent must be numpy array, got {type(latent_array)}")
        
        if latent_array.ndim != expected_dims:
            raise Gen5LatentError(
                f"Expected {expected_dims}D latent, got {latent_array.ndim}D shape {latent_array.shape}"
            )
        
        #check for reasonable dimension sizes
        for i, dim in enumerate(latent_array.shape):
            if dim <= 0:
                raise Gen5LatentError(f"Invalid dimension size at axis {i}: {dim}")
            if dim > max_dimension_size:
                raise Gen5LatentError(
                    f"Dimension {i} size {dim} exceeds maximum {max_dimension_size}"
                )
        
        # Check for reasonable total size (prevent memory bombs)
        total_elements = np.prod(latent_array.shape)
        max_elements = 100_000_000  # 100M elements (~200MB for float16)
        if total_elements > max_elements:
            raise Gen5LatentError(
                f"Latent too large: {total_elements} elements exceeds maximum {max_elements}"
            )
        
        return True


    def latent_dtype_validator(self, latent_array: np.ndarray) -> bool:
        """Validate latent data type.
        
        Args:
            latent_array (np.ndarray): Latent array to validate.
            
        Returns:
            bool: True if valid.
            
        Raises:
            Gen5LatentError: If dtype is invalid.
        """
        allowed_dtypes = [np.float16, np.float32]
        if latent_array.dtype not in allowed_dtypes:
            raise Gen5LatentError(
                f"Latent dtype must be float16 or float32, got {latent_array.dtype}"
            )
        
        # Check for NaN or Inf values
        if np.isnan(latent_array).any():
            raise Gen5LatentError("Latent contains NaN values")
        
        if np.isinf(latent_array).any():
            raise Gen5LatentError("Latent contains Inf values")
        
        return True

    def latent_packer(self, latent: Dict[str, np.ndarray], file_offset: int = 0, chunk_records=None, should_compress: bool = True, convert_float16: bool = True) -> list:
            """Pack latent arrays into compressed chunk.
            Args:
                latent (Dict[str, np.ndarray]): Dictionary of latent arrays.
                file_offset (int): Current file offset for chunk placement.
                chunk_records (list): List to append chunk records to.
            Returns:
                list: List of compressed latent chunks."""
            latents=[]
            if chunk_records is None:
                chunk_records = []
            if not latent:
                raise Gen5LatentError("Latent dictionary cannot be empty")
            for key, latent_array in latent.items():
                self.latent_shape_validator(latent_array)
                self.latent_dtype_validator(latent_array=latent_array)
                
                if convert_float16:
                    if latent_array.dtype != np.float16:
                        latent_array = latent_array.astype(np.float16)
                    chunk_flags = b"F16\x00"
                else:
                    if latent_array.dtype == np.float32:
                        chunk_flags = b'F32\x00'
                    elif latent_array.dtype == np.float16:
                        chunk_flags = b'F16\x00'
                    else:
                        raise Gen5LatentError(f"Unsupported dtype when convert_float16=False: {latent_array.dtype}")
                chunk_type = b'LATN'
                data_bytes = latent_array.tobytes()
                uncompressed_size = len(data_bytes)
                if should_compress:
                    compressed = self.chunk_packer_binary(chunk_type, chunk_flags, data_bytes)
                    compressed_size = len(compressed)
                else:

                    header = struct.pack('<4s 4s I', chunk_type, chunk_flags, uncompressed_size)
                    compressed = header + data_bytes 
                    compressed_size = len(compressed)

                #create manifest
                chunk_records.append({
                    "type": "LATN",
                    "flags": chunk_flags.decode('ascii').strip('\x00'),
                    "offset": file_offset,
                    "compressed_size": compressed_size,
                    "compressed": should_compress,
                    "uncompressed_size": uncompressed_size,
                    "hash": hashlib.sha256(data_bytes).hexdigest(),
                    "extra": {
                        "shape": list(latent_array.shape),
                        "dtype": str(latent_array.dtype),
                        "key": key
                }
                })
                file_offset += compressed_size
                latents.append(compressed)
                
            return latents

    def latent_parser(self, chunk: bytes, shape: tuple, compressed: bool = True):
        """Parse a given single latent chunk and return the latent array."""
        if compressed:
            decompressor = ZstdDecompressor()
            try:
                decompressed = decompressor.decompress(chunk)
            except Exception as e:
                raise Gen5LatentError(f"Failed to decompress latent chunk: {e}") from e

            if len(decompressed) < 12:
                raise Gen5LatentError("Truncated latent chunk after decompression")

            chunk_type, chunk_flags, chunk_size = struct.unpack('<4s 4s I', decompressed[:12])
            data_bytes = decompressed[12:12 + chunk_size]

            if len(data_bytes) != chunk_size:
                raise Gen5LatentError("Truncated latent chunk after decompression")

            dtype = self.dtype_from_flags(chunk_flags)
            expected = np.prod(shape) * dtype.itemsize
            if len(data_bytes) != expected:
                raise Gen5LatentError(
                    f"Size mismatch: expected {expected} bytes, got {len(data_bytes)}"
                )
            return np.frombuffer(data_bytes, dtype=dtype).copy().reshape(shape)

        else:
            if len(chunk) < 12:
                raise Gen5LatentError("Truncated latent chunk header")
            chunk_type, chunk_flags, data_size = struct.unpack('<4s 4s I', chunk[:12])
            dtype = self.dtype_from_flags(chunk_flags)
            expected_data_size = np.prod(shape) * dtype.itemsize

            if data_size != expected_data_size:
                raise Gen5LatentError(
                    f"Size mismatch! Header says {data_size} bytes, "
                    f"but shape {shape} requires {expected_data_size} bytes"
                )

            data_bytes = chunk[12:12 + data_size]
            if len(data_bytes) != data_size:
                raise Gen5LatentError(f"Truncated latent: expected {data_size} bytes, got {len(data_bytes)}")

            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
