import hashlib
from ..core.constants import MAX_FILE_SIZE, MAX_CHUNK_SIZE, HEADER_SIZE, MAX_CHUNKS
from ..core.exceptions import Gen5DecodeError, Gen5ChunkError, Gen5CorruptHeader, Gen5ImageError, Gen5EnvChunkError
from ..core.header import header_init, header_parse, header_validate
from ..chunks.latent import Gen5Latent
from ..chunks.image import Gen5Image  
from ..chunks.env import Gen5Env
from ..chunks.metadata import Gen5Metadata
from typing import Optional, Dict
from PIL import Image
import numpy as np
import struct
import io
import warnings
import json
class Gen5FileHandler():
    def __init__(self, max_file_size: Optional[int] = None, 
             max_chunk_size: Optional[int] = None):
        """Initialize with optional custom limits."""
        self.latent = Gen5Latent()
        self.image = Gen5Image()
        self.metadata = Gen5Metadata()
        self.env = Gen5Env()
        self.max_file_size = max_file_size or MAX_FILE_SIZE
        self.max_chunk_size = max_chunk_size or MAX_CHUNK_SIZE
        self.HEADER_SIZE = HEADER_SIZE  
    def validate_file_size(self, size: int, context: str = "file") -> bool:
        """Validate file or chunk size.
        
        Args:
            size (int): Size in bytes to validate.
            context (str): Context for error message.
            
        Returns:
            bool: True if valid.
            
        Raises:
            Gen5DecodeError: If size exceeds limits.
        """
        if size < 0:
            raise Gen5DecodeError(f"Invalid {context} size: {size} (negative)")
        
        if context == "file" and size > self.max_file_size:
            raise Gen5DecodeError(
                f"File size {size:,} bytes exceeds maximum {self.max_file_size:,} bytes "
                f"({size / (1024**3):.2f} GB)"
            )
        
        if context == "chunk" and size > self.max_chunk_size:
            raise Gen5ChunkError(
                f"Chunk size {size:,} bytes exceeds maximum {self.max_chunk_size:,} bytes"
            )
        
        return True
    
    def validate_chunk_count(self, count: int) -> bool:
        """Validate number of chunks.
        
        Args:
            count (int): Number of chunks.
            
        Returns:
            bool: True if valid.
            
        Raises:
            Gen5DecodeError: If count exceeds limits.
        """
        if count < 0:
            raise Gen5DecodeError(f"Invalid chunk count: {count}")
        
        if count > MAX_CHUNKS:
            raise Gen5DecodeError(
                f"Chunk count {count} exceeds maximum {MAX_CHUNKS}"
            )
        
        return True
        

    def file_encoder(self, filename: str, latent: Dict[str, np.ndarray], chunk_records: list,
                    model_name: str, model_version: str, prompt: str, tags: list, img_binary: bytes, should_compress: bool = True, convert_float16: bool = True, generation_settings: Optional[dict] = None, hardware_info: Optional[dict] = None):
        """ Orchestrator function to encode GEN5 file.
        Args:
            filename (str): Output GEN5 filename. [.gen5 extension is REQUIRED]
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

        current_offset = self.HEADER_SIZE
        latent_chunks = self.latent.latent_packer(latent, file_offset=current_offset, chunk_records=chunk_records, should_compress=should_compress, convert_float16=convert_float16)
        latent_chunk = b"".join(latent_chunks)
        self.validate_file_size(len(latent_chunk), "chunk")
        current_offset += len(latent_chunk)
        

        #Pack image chunk if provided
        image_chunk = None
        if img_binary is not None:
            image_chunk = self.image.image_data_chunk_builder(img_binary)
            image_chunk_record = {
                "type": "DATA",
                "flags": "0000",
                "offset": current_offset,
                "compressed_size": len(image_chunk),
                "uncompressed_size": len(img_binary),
                "hash": hashlib.sha256(img_binary).hexdigest(),
                "extra": {},
                "compressed": True
            }
            chunk_records.append(image_chunk_record)
            current_offset += len(image_chunk)
            print("ENCODER STORED FLAG:", chunk_records[-1]["flags"])

        #env chunk
        env_chunk, env_raw = self.env.env_chunk_builder(self.env.env_chunk_populator())
        if len(env_chunk) > 0:
            env_record = {
                "type": "ENVC",
                "flags": "0000",
                "offset": current_offset,
                "compressed_size": len(env_chunk),   # bytes written
                "uncompressed_size": len(env_raw),   # JSON bytes
                "hash": hashlib.sha256(env_raw).hexdigest(),   # hash original
                "extra": {},
                "compressed": True
            }

            chunk_records.append(env_record)
            current_offset += len(env_chunk)
            



        #build manifest with all chunks
        manifest = self.metadata.build_manifest(
            version_major=1,
            version_minor=0,
            model_name=model_name,
            model_version=model_version,
            prompt=prompt,
            tags=tags,
            chunk_records=chunk_records,
            generation_settings=generation_settings,
            hardware_info=hardware_info
        )
        #ensure that manifest/metadata is valid
        self.metadata.metadata_validator(manifest)
        #compress metadata
        compressed_metadata = self.metadata.metadata_compressor(manifest)
        metadata_size = len(compressed_metadata)
        

        #calculate total file size
        total_file_size = (self.HEADER_SIZE + len(latent_chunk) + (len(image_chunk) if image_chunk else 0) + len(env_chunk) + metadata_size)
        #update file_size in manifest
        manifest["gen5_metadata"]["file_info"]["file_size"] = total_file_size

        #recompress metadata with the updated file_size
        compressed_metadata = self.metadata.metadata_compressor(manifest)
        metadata_size = len(compressed_metadata)
        total_file_size = self.HEADER_SIZE + len(latent_chunk) + (len(image_chunk) if image_chunk else 0) + len(env_chunk) + len(compressed_metadata)
        chunk_table_offset = (
    self.HEADER_SIZE
    + len(latent_chunk)
    + (len(image_chunk) if image_chunk else 0)
    + len(env_chunk)
)

        #final header
        header = header_init(
            version_major=1,
            version_minor=0,
            flags=0,
            chunk_table_offset=chunk_table_offset,
            chunk_table_size=metadata_size,
            chunk_count=len(chunk_records),
            file_size=total_file_size
        )



        
        with open(filename, "wb") as f:
            f.write(header)
            f.write(latent_chunk)
            if image_chunk is not None:
                f.write(image_chunk)
            f.write(env_chunk)
            f.write(compressed_metadata)


        return {
            "header": header,
            "latent_chunks": latent_chunk,
            "metadata_chunk": compressed_metadata,
            "image_chunk": image_chunk
        }


    def file_decoder(self, filename: str):
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
                raise Gen5CorruptHeader(message=f"Invalid header: {header}")
            f.seek(header['chunk_table_offset'])
            metadata_compressed = f.read(header['chunk_table_size'])
            metadata = self.metadata.metadata_parser(metadata_compressed)
            chunk_records = metadata['gen5_metadata']['chunks']
            chunks = {}
            chunks['latent'] = []
            
            for record in chunk_records:
                chunk_type = record['type']
                compressed = record.get("compressed", True)

                f.seek(record['offset'])
                raw_chunk = f.read(record['compressed_size'])

                if len(raw_chunk) != record['compressed_size']:
                    raise Gen5ChunkError(f"Truncated chunk {chunk_type} at offset {record['offset']}")

                if chunk_type == "LATN":
                    shape = tuple(record['extra']['shape'])

                    if compressed:
                        print("decoded raw_chunk len:", len(raw_chunk))
                        print("expected total bytes:", np.prod(shape))
                        latent_array = self.latent.latent_parser(raw_chunk, shape, True)
                    else:
                        latent_array = self.latent.latent_parser(raw_chunk, shape, False)

                    chunks['latent'].append(latent_array)
                elif chunk_type == "DATA":
                    if compressed:
                        parsed = self.image.image_data_chunk_parser(raw_chunk)
                        chunks['image'] = parsed['image_data']
                    else:
                        # DATA chunks have the same header layout even when not compressed
                        chunk_type_b, flags_b, size = struct.unpack('<4s 4s I', raw_chunk[:12])
                        chunks['image'] = raw_chunk[12:12+size]
                elif chunk_type == "ENVC":
                    if compressed:
                        parsed = self.env.env_chunk_parser(raw_chunk)
                        chunks['env'] = parsed['env_chunk']
                    else:
                        if len(raw_chunk) < 12:
                            raise Gen5EnvChunkError("Truncated ENVC chunk header")
                        chunk_type_b, flags_b, size = struct.unpack('<4s 4s I', raw_chunk[:12])
                        env_json_bytes = raw_chunk[12:12 + size]
                        env_dict = json.loads(env_json_bytes.decode('utf-8'))
                        chunks['env'] = env_dict

                    try:
                        # --- Normalize CURRENT environment (from populator) ---
                        current_env_obj = self.env.env_chunk_populator()
                        current_env = {}
                        for comp in current_env_obj.components:
                            if isinstance(comp, dict):
                                comp_id = comp["component_id"]
                                cononical_str = comp["cononical_str"]
                                digest = comp["component_sha256_digest"]
                            else:
                                comp_id = comp.component_id
                                cononical_str = comp.cononical_str          # ✅ matches your class
                                digest = comp.component_sha256_digest

                            sha256 = digest.hex() if isinstance(digest, bytes) else digest
                            current_env[comp_id] = {
                                "cononical_str": cononical_str,
                                "sha256": sha256
                            }

                        # --- Normalize STORED environment (from file) ---
                        stored_raw = chunks['env']
                        stored_env = {}

                        # Extract components regardless of format
                        if hasattr(stored_raw, 'components'):
                            components_list = stored_raw.components
                        elif isinstance(stored_raw, dict) and 'components' in stored_raw:
                            components_list = stored_raw['components']
                        else:
                            components_list = []

                        for comp in components_list:
                            if isinstance(comp, dict):
                                comp_id = comp["component_id"]
                                cononical_str = comp["cononical_str"]
                                digest = comp["component_sha256_digest"]
                            else:
                                comp_id = comp.component_id
                                cononical_str = comp.cononical_str          # ✅
                                digest = comp.component_sha256_digest

                            sha256 = digest.hex() if isinstance(digest, bytes) else digest
                            stored_env[comp_id] = {
                                "cononical_str": cononical_str,
                                "sha256": sha256
                            }

                        # --- Compare environments ---
                        all_ids = set(stored_env.keys()) | set(current_env.keys())
                        for comp_id in all_ids:
                            stored = stored_env.get(comp_id)
                            current = current_env.get(comp_id)
                            if stored and current:
                                if stored["sha256"] != current["sha256"]:
                                    warnings.warn(
                                        f"Environment component '{comp_id}' differs:\n"
                                        f"  File: {stored['cononical_str']}\n"
                                        f"  Current: {current['cononical_str']}",
                                        UserWarning
                                    )
                            elif stored and not current:
                                warnings.warn(f"Environment component '{comp_id}' missing in current system", UserWarning)
                            elif not stored and current:
                                warnings.warn(f"Environment component '{comp_id}' missing in file", UserWarning)

                    except Exception as e:
                        warnings.warn(f"Failed to compare environment chunks: {e}", UserWarning)                
                else:
                    raise ValueError(f"Unknown chunk type: {chunk_type}. Supported: LATN, DATA")

            return {
                "header": header,
                "chunks": chunks,
                "metadata": metadata
            }

    @staticmethod
    def png_to_bytes(png_path: str) -> bytes:
        """Convert PNG image to bytes, preserving transparency.
        Args:
            png_path (str): Path to the PNG image.
        Returns:
            bytes: PNG image data in bytes.

        """
        try:
            with Image.open(png_path) as img:
                img = img.convert("RGBA") 
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            raise Gen5ImageError(f"Failed to convert PNG to bytes: {e}") from e
    @staticmethod
    def bytes_to_png(img_bytes: bytes) -> Image.Image:
        """Convert bytes back to PNG image."""
        try:
            buffer = io.BytesIO(img_bytes)
            img = Image.open(buffer)
            return img
        except Exception as e:
            raise Gen5ImageError(f"Failed to convert bytes to PNG: {e}") from e

