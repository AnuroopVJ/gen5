import io
from PIL import Image
from zstandard import ZstdCompressor, ZstdDecompressor
from ..core.exceptions import Gen5ImageError
import struct

class Gen5Image:
    def __init__(self):
        pass
    def image_bytes_validator(self, image_bytes: bytes):
        """Validate image bytes.
        Args:
            image_bytes (bytes): Image bytes to validate.
        """
        try:
            img = io.BytesIO(image_bytes)
            with Image.open(img) as img:
                img.verify()
            return True
        except Exception as e:
            raise Gen5ImageError(f"Invalid image bytes: {e}")


    def image_data_chunk_builder(self, image_binary: bytes):
        self.image_bytes_validator(image_binary)
        try:
            chunk_type = b'DATA'
            chunk_flags = b'0000'
            chunk_size = len(image_binary)
            chunk_header = struct.pack('<4s 4s I', chunk_type, chunk_flags, chunk_size)
            chunk = chunk_header + image_binary
            compressed_chunk = ZstdCompressor().compress(chunk)
            return compressed_chunk         
        except Exception as e:
            raise Gen5ImageError(f"Failed to build image data chunk: {e}") from e

    def image_data_chunk_parser(self, compressed_chunk):
        """Parse image data chunk.
        Args:
            compressed_chunk (bytes): Compressed image data chunk.
        Returns:
            dict: Parsed image data chunk information.
        """
        try:
            decompressor = ZstdDecompressor()
            chunk = decompressor.decompress(compressed_chunk)
            chunk_type, chunk_flags, chunk_size = struct.unpack('<4s 4s I', chunk[:12])
            image_data = chunk[12:12+chunk_size]
            return {
                "chunk_type": chunk_type,
                "chunk_flags": chunk_flags,
                "chunk_size": chunk_size,
                "image_data": image_data
            }
        except Exception as e:
            raise Gen5ImageError(f"Failed to parse image data chunk: {e}") from e
