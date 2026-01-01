import struct

MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 gb default
MAX_CHUNK_SIZE = 2 * 1024 * 1024 * 1024  #2 gb per chunk
MAX_CHUNKS = 1000
HEADER_FORMAT = '<4s B B H I I I I Q'  #uses little endian
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
JSON_SCHEMA = """{
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
        "required": [
        "magic",
        "version_major",
        "version_minor",
        "file_size",
        "chunk_count"
        ]
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
        },

        "generation_settings": {
            "type": "object",
            "properties": {
            "seed":         { "type": "integer", "minimum": 0 },
            "steps":        { "type": "integer", "minimum": 1 },
            "sampler":      { "type": "string" },
            "cfg_scale":    { "type": "number", "minimum": 0 },
            "scheduler":    { "type": "string" },
            "eta":          { "type": "number", "minimum": 0 },
            "guidance":     { "type": "string" },
            "precision":    { "type": "string" },
            "deterministic":{ "type": "boolean" }
            },
            "required": ["seed", "steps", "sampler"]
        },

        "hardware_info": {
            "type": "object",
            "properties": {
            "machine_name": { "type": "string" },
            "os":           { "type": "string" },
            "cpu":          { "type": "string" },
            "cpu_cores":    { "type": "integer", "minimum": 1 },

            "gpu": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                    "name":         { "type": "string" },
                    "memory_gb":    { "type": "number", "minimum": 0 },
                    "driver":       { "type": "string" },
                    "cuda_version": { "type": "string" }
                },
                "required": ["name"]
                }
            },

            "ram_gb":      { "type": "number", "minimum": 0 },
            "framework":   { "type": "string" },
            "compute_lib": { "type": "string" }
            },
            "required": ["os"]
        }
        },

        "required": [
        "model_name",
        "version",
        "date",
        "prompt",
        "tags"
        ]
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
            "extra":             { "type": "object" },
            "compressed":        { "type": "boolean" }
        },
        "required": [
            "index",
            "type",
            "flags",
            "offset",
            "compressed_size",
            "uncompressed_size",
            "hash",
            "extra",
            "compressed"
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