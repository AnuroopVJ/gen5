import struct
from .constants import HEADER_SIZE, HEADER_FORMAT
from .exceptions import Gen5CorruptHeader

def header_init(
    version_major = 1,
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