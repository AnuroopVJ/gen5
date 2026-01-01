import jsonschema
from ..core.constants import JSON_SCHEMA
import json
from ..core.exceptions import Gen5MetadataError
import struct
import zstandard as zstd
from typing import Optional
from datetime import datetime, UTC

class Gen5Metadata:
    def __init__(self):
        pass
    def metadata_validator(self, manifest) -> bool:
        """Validate metadata manifest using JSON Schema.
        Args:
            manifest (dict): Metadata manifest.
        Returns:
            bool: True if it is valid or False otherwise.
        Raises:
            Gen5MetadataError: If the metadata is invalid.
        """
        json_schema = JSON_SCHEMA
        schema = json.loads(json_schema)
        try:
            jsonschema.validate(instance=manifest, schema=schema)
            return True
        except Exception as e:
            raise Gen5MetadataError(f"Invalid metadata: {e}")
            
                    
    def metadata_compressor(self, manifest):
        json_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        chunk_type = b"META"
        chunk_flags = b"0000"
        chunk_size = len(json_bytes)
        header = struct.pack("<4s 4s I", chunk_type, chunk_flags, chunk_size)
        compressed = zstd.ZstdCompressor().compress(header + json_bytes)
        return compressed
    def metadata_parser(self, compressed_chunk: bytes) -> dict:
            """Parse and decompress metadata chunk.
            Args:
                compressed_chunk (bytes): Compressed metadata chunk.
            Returns:
                dict: Parsed metadata manifest."""
            decompressor = zstd.ZstdDecompressor()
            chunk = decompressor.decompress(compressed_chunk)
            chunk_type, chunk_flags, chunk_size = struct.unpack('<4s 4s I', chunk[:12])
            json_bytes = chunk[12:12+chunk_size]
            manifest = json.loads(json_bytes.decode("utf-8"))
            return manifest

    def build_manifest(
            self,
            version_major: int,
            version_minor: int,
            model_name: str,
            model_version: str,
            prompt: str,
            tags: list,
            chunk_records: list,
            generation_settings: Optional[dict] = None,
            hardware_info: Optional[dict] = None
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
                        "date": datetime.now(UTC).isoformat(),
                        "prompt": prompt,
                        "tags": tags,

                        "generation_settings": generation_settings or {
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

                        "hardware_info": hardware_info or {
                            "machine_name": "",
                            "os": "",
                            "cpu": "",
                            "cpu_cores": 0,
                            "gpu": [],
                            "ram_gb": 0.0,
                            "framework": "",
                            "compute_lib": ""
                        }
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
                    "extra": rec.get("extra", {}),
                    "compressed": rec.get("compressed", True)
                })

            
            return manifest