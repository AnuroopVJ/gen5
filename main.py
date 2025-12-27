import hashlib
import struct
from PIL import Image
from typing import Optional, Dict, Iterable
from zstandard import ZstdCompressor, ZstdDecompressor
import json
import numpy as np
import jsonschema
from datetime import datetime
import io


#HEADER
HEADER_FORMAT = '<4s B B H I I I I Q'  #uses little endian
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def header_init(version_major = 1,
    version_minor = 0,
    flags = 0,
    chunk_table_offset = 0,
    chunk_table_size = 0,
    chunk_count = 0,
    file_size = 0,
    reserved = 0):
    """Initialize GEN5 file header.
    Returns the packed header bytes.
    Args:
        version_major (int): Major version number.
        version_minor (int): Minor version number.
        flags (int): Header flags.
        chunk_table_offset (int): Offset to the chunk table.
        chunk_table_size (int): Size of the chunk table.
        chunk_count (int): Number of chunks.
        file_size (int): Total file size.
        reserved (int): Reserved for future use.
    """
    # header init
    return struct.pack(
        HEADER_FORMAT,
        b'GEN5',
        version_major,
        version_minor,
        flags,
        chunk_table_offset,
        chunk_table_size,
        chunk_count,
        file_size,
        reserved
    )

def header_parse(data: bytes):
    """Parse GEN5 file header.
    Args:
        data (bytes): The header bytes to parse.
        Returns:
        dict: Parsed header fields.
        
    Returns:
        dict: Parsed header fields.
        """
    unpacked = struct.unpack(HEADER_FORMAT, data)
    return {
        'magic': unpacked[0],
        'version_major': unpacked[1],
        'version_minor': unpacked[2],
        'flags': unpacked[3],
        'chunk_table_offset': unpacked[4],
        'chunk_table_size': unpacked[5],
        'chunk_count': unpacked[6],
        'file_size': unpacked[7],
        'reserved': unpacked[8]
    }

def header_validate(header: dict) -> bool:
    """Validate GEN5 file header.
    Args:
        header (dict): Parsed header fields.
        Returns:
        bool: True if it is valid or False otherwise."""
    if header['magic'] != b'GEN5':
        return False
    if header['version_major'] < 1:
        return False
    if header['chunk_count'] < 0:
        return False
    return True

def png_to_bytes(png_path: str) -> bytes:
    """Convert PNG image to bytes, preserving transparency.
    Args:
        png_path (str): Path to the PNG image.
    Returns:
        bytes: PNG image data in bytes.

    """
    with Image.open(png_path) as img:
        img = img.convert("RGBA") 
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

def bytes_to_png(img_bytes: bytes) -> Image.Image:
    """Convert bytes back to PNG image."""
    buffer = io.BytesIO(img_bytes)
    img = Image.open(buffer)
    return img

def chunk_packer_binary(chunk_type:bytes, chunk_flags: bytes, data:bytes) -> bytes:
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
    
   #METADATA
def metadata_validator(manifest) -> bytes:
    """Validate and compress metadata manifest using JSON Schema and zstd."""
    json_schema = """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "GEN5 Metadata Schema",
  "type": "object",
  
  "properties": {
    "gen5_metadata": {
      "type": "object",

      "properties": {

        "file_info": {
          "type": "object",
          "properties": {
            "magic":          { "type": "string", "const": "GEN5" },
            "version_major":  { "type": "integer", "minimum": 1 },
            "version_minor":  { "type": "integer", "minimum": 0 },
            "file_size":      { "type": "integer", "minimum": 0 },
            "chunk_count":    { "type": "integer", "minimum": 0 }
          },
          "required": ["magic", "version_major", "version_minor", "file_size", "chunk_count"]
        },

        "model_info": {
          "type": "object",
          "properties": {
            "model_name": { "type": "string" },
            "version":    { "type": "string" },
            "date":       { "type": "string" },
            "prompt":     { "type": "string" },
            "tags": {
              "type": "array",
              "items": { "type": "string" }
            }
          },
          "required": ["model_name", "version", "date", "prompt", "tags"]
        },

        "chunks": {
          "type": "array",
          "items": {
            "type": "object",

            "properties": {
              "index":             { "type": "integer", "minimum": 0 },
              "type":              { "type": "string" },
              "flags":             { "type": "string" },
              "offset":            { "type": "integer", "minimum": 0 },
              "compressed_size":   { "type": "integer", "minimum": 0 },
              "uncompressed_size": { "type": "integer", "minimum": 0 },
              "hash":              { "type": "string" },
              "extra":             { "type": "object" }
            },

            "required": [
              "index",
              "type",
              "flags",
              "offset",
              "compressed_size",
              "uncompressed_size",
              "hash",
              "extra"
            ]
          }
        }
      },

      "required": ["file_info", "model_info", "chunks"]
    }
  },

  "required": ["gen5_metadata"]
}
"""
    schema = json.loads(json_schema)
    jsonschema.validate(instance=manifest, schema=schema)
    json_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    chunk_type = b"META"
    chunk_flags = b"0000"
    chunk_size = len(json_bytes)
    header = struct.pack("<4s 4s I", chunk_type, chunk_flags, chunk_size)
    compressed = ZstdCompressor().compress(header + json_bytes)
    return compressed

def build_manifest(
    version_major: int,
    version_minor: int,
    model_name: str,
    model_version: str,
    prompt: str,
    tags: list,
    chunk_records: list
):
    """
    chunk_records must be a list of dicts, with each having:
        {
            "type": str,
            "flags": str,
            "offset": int,
            "compressed_size": int,
            "uncompressed_size": int,
            "hash": str,
            "extra": dict
        }
    Args:
        version_major (int): Major version number.
        version_minor (int): Minor version number.
        model_name (str): Name of the model.
        model_version (str): Version of the model.
        prompt (str): Prompt used for generation.
        tags (list): List of tags.
        chunk_records (list): List of chunk record dictionaries.
    Returns:
        dict: Manifest dictionary.
    """

    manifest = {
        "gen5_metadata": {
            "file_info": {
                "magic": "GEN5",
                "version_major": version_major,
                "version_minor": version_minor,
                "file_size": 0,
                "chunk_count": len(chunk_records)
            },

            "model_info": {
                "model_name": model_name,
                "version": model_version,
                "date": datetime.utcnow().isoformat() + "Z",
                "prompt": prompt,
                "tags": tags
            },

            "chunks": []
        }
    }

    #build chunk list with indexes
    for idx, rec in enumerate(chunk_records):
        manifest["gen5_metadata"]["chunks"].append({
            "index": idx,
            "type": rec["type"],
            "flags": rec["flags"],
            "offset": rec["offset"],
            "compressed_size": rec["compressed_size"],
            "uncompressed_size": rec["uncompressed_size"],
            "hash": rec["hash"],
            "extra": rec.get("extra", {})
        })
    
    return manifest
    

def metadata_parser(compressed_chunk: bytes) -> dict:
    """Parse and decompress metadata chunk.
    Args:
        compressed_chunk (bytes): Compressed metadata chunk.
    Returns:
        dict: Parsed metadata manifest."""
    decompressor = ZstdDecompressor()
    chunk = decompressor.decompress(compressed_chunk)
    chunk_type, chunk_flags, chunk_size = struct.unpack('<4s 4s I', chunk[:12])
    json_bytes = chunk[12:12+chunk_size]
    manifest = json.loads(json_bytes.decode("utf-8"))
    return manifest

#IMAGE-DATA

def image_data_chunk_builder(image_binary):
    """Build image data chunk from binary image data.
    Args:
        image_binary (bytes): Binary image data.
    Returns:
        bytes: Compressed image data chunk.
    """
    chunk_type = b'DATA'
    chunk_flags = b'0000'
    chunk_size = len(image_binary)
    chunk_header = struct.pack('<4s 4s I', chunk_type, chunk_flags, chunk_size)
    chunk = chunk_header + image_binary
    compressed_chunk = ZstdCompressor().compress(chunk)
    return compressed_chunk

def image_data_chunk_parser(compressed_chunk):
    """Parse image data chunk.
    Args:
        compressed_chunk (bytes): Compressed image data chunk.
    Returns:
        dict: Parsed image data chunk information.
    """
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

#LATENT CHUNK

def latent_packer(latent: Dict[str, np.ndarray], file_offset: int = 0, chunk_records=None) -> list:
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
    for latent_array in latent.values():
        if latent_array.dtype not in (np.float16, np.float32):
            raise ValueError("Latent must be float16 or float32")

        # Convert to float16 if necessary
        if latent_array.dtype != np.float16:
            latent_array = latent_array.astype(np.float16)

        chunk_type = b'LATN'
        chunk_flags = b'F16'
        data_bytes = latent_array.tobytes()
        uncompressed_size = len(data_bytes)

        # Compress the chunk
        compressed = chunk_packer_binary(chunk_type, chunk_flags, data_bytes)
        compressed_size = len(compressed)

        # Create manifest record

        chunk_records.append({
            "type": "LATN",
            "flags": "F16" if latent_array.dtype == np.float16 else "F32",
            "offset": file_offset,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "hash": hashlib.sha256(data_bytes).hexdigest(),
            "extra": {
                "shape": list(latent_array.shape),
                "dtype": str(latent_array.dtype)
            }
        })
        file_offset += compressed_size
        latents.append(compressed)
        
    return latents

def latent_parser(compressed_chunk: bytes, shape: tuple):
    """Parse a single latent chunk and return the latent array.
    
    Args:
        compressed_chunk (bytes): Compressed latent chunk.
        shape (tuple): Shape to reshape the latent array to.
        
    Returns:
        np.ndarray: Parsed latent array.
    """
    decompressor = ZstdDecompressor()
    decompressed = decompressor.decompress(compressed_chunk)

    if len(decompressed) < 12:
        raise ValueError("Truncated latent chunk header")

    chunk_type, chunk_flags, chunk_size = struct.unpack('<4s 4s I', decompressed[:12])
    data_bytes = decompressed[12:12 + chunk_size]

    if len(data_bytes) != chunk_size:
        raise ValueError("Truncated latent chunk payload")

    flag_str = chunk_flags.decode('utf-8', errors='ignore').strip()
    dtype = np.float16 if flag_str == "F16" else np.float32

    arr = np.frombuffer(data_bytes, dtype=dtype).copy()
    arr = arr.reshape(shape)
    return arr


#RUNNING
def file_encoder(filename: str, latent: Dict[str, np.ndarray], chunk_records: list,
                 model_name: str, model_version: str, prompt: str, tags: list, img_binary: bytes):
    """ Orchestrator function to encode GEN5 file.
    Args:
        filename (str): Output GEN5 filename. [REQUIRED]
        latent (Dict[str, np.ndarray]): Dictionary of latent arrays.
        chunk_records (list): List to append chunk records to.
        model_name (str): Name of the model.
        model_version (str): Version of the model.
        prompt (str): Prompt used for generation.
        tags (list): List of tags.
        img_binary (bytes): Binary image data.
        Returns:
        dict: Dictionary with header, latent chunks, metadata, and image chunk.
        """
    # Pack the latent chunks first and update chunk_records with correct offsets
    if not filename.endswith('.gen5'):
        raise ValueError("Filename must have a .gen5 extension")

    current_offset = HEADER_SIZE
    latent_chunks = latent_packer(latent, file_offset=current_offset, chunk_records=chunk_records)
    latent_chunk = b"".join(latent_chunks)
    current_offset += len(latent_chunk)

    #Pack image chunk if provided
    image_chunk = None
    if img_binary is not None:
        image_chunk = image_data_chunk_builder(img_binary)
        image_chunk_record = {
            "type": "DATA",
            "flags": "0000",
            "offset": current_offset,
            "compressed_size": len(image_chunk),
            "uncompressed_size": len(img_binary),
            "hash": hashlib.sha256(img_binary).hexdigest(),
            "extra": {}
        }
        chunk_records.append(image_chunk_record)
        current_offset += len(image_chunk)

    #build manifest with all chunks
    manifest = build_manifest(
        version_major=1,
        version_minor=0,
        model_name=model_name,
        model_version=model_version,
        prompt=prompt,
        tags=tags,
        chunk_records=chunk_records
    )

    # compress metadata
    compressed_metadata = metadata_validator(manifest)
    metadata_size = len(compressed_metadata)
    
    # calculate total file size
    total_file_size = HEADER_SIZE + len(latent_chunk) + (len(image_chunk) if image_chunk else 0) + metadata_size

    # update file_size in manifest
    manifest["gen5_metadata"]["file_info"]["file_size"] = total_file_size
    
    # recompress metadata with the updated file_size
    compressed_metadata = metadata_validator(manifest)
    metadata_size = len(compressed_metadata)

    #final header
    header = header_init(
        version_major=1,
        version_minor=0,
        flags=0,
        chunk_table_offset=HEADER_SIZE + len(latent_chunk) + (len(image_chunk) if image_chunk else 0),
        chunk_table_size=metadata_size,
        chunk_count=len(chunk_records),
        file_size=total_file_size
    )

    
    with open(filename, "wb") as f:
        f.write(header)
        f.write(latent_chunk)
        if image_chunk is not None:
            f.write(image_chunk)
        f.write(compressed_metadata)

    return {
        "header": header,
        "latent_chunks": latent_chunk,
        "metadata_chunk": compressed_metadata,
        "image_chunk": image_chunk
    }


def file_decoder(filename: str):
    """ Orchestrator function to decode GEN5 file.
    Args:
        filename (str): Input GEN5 filename.
    Returns:
        dict: Dictionary with header, chunks, and metadata.
    """
    with open(filename, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        header = header_parse(header_bytes)
        if not header_validate(header):
            raise ValueError("Invalid or corrupted GEN5 header")
        f.seek(header['chunk_table_offset'])
        metadata_compressed = f.read(header['chunk_table_size'])
        metadata = metadata_parser(metadata_compressed)

        chunk_records = metadata['gen5_metadata']['chunks']
        chunks = {}
        chunks['latent'] = []
        
        for record in chunk_records:
            chunk_type = record['type']
            
            # seek to the exact position
            f.seek(record['offset'])
            compressed_chunk = f.read(record['compressed_size'])
            if len(compressed_chunk) != record['compressed_size']:
                raise IOError(f"Truncated chunk {chunk_type} at offset {record['offset']}")
            
            if chunk_type == "LATN":
                shape = tuple(record['extra']['shape'])
                latent_array = latent_parser(compressed_chunk, shape)
                chunks['latent'].append(latent_array)
                
            elif chunk_type == "DATA":
                parsed = image_data_chunk_parser(compressed_chunk)
                chunks['image'] = parsed['image_data']
                
            else:
                #unknown chunks are skipped but still accessible if required
                raise ValueError(f"Unknown chunk type: {chunk_type}. "
                                 f"Supported: LATN, DATA")
        
        return {
            "header": header,
            "chunks": chunks,
            "metadata": metadata
        }
    

def make_lazy_latent_loader(filename: str, chunk_record: dict):
    """Return a callable that loads the latent array on demand."""
    loaded_array = None

    def load():
        nonlocal loaded_array
        if loaded_array is None:
            with open(filename, "rb") as f:
                offset = chunk_record['offset']
                size = chunk_record['compressed_size']
                f.seek(offset)
                compressed_chunk = f.read(size)
                shape = tuple(chunk_record['extra']['shape'])
                loaded_array = latent_parser(compressed_chunk, shape)
        return loaded_array

    return load

def iter_lazy_latents(filename: str, chunk_records: list):
    """Yield callables for each latent chunk for lazy loading."""
    for record in chunk_records:
        if record['type'] == "LATN":
            yield make_lazy_latent_loader(filename, record)

