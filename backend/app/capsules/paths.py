"""Native Windows long-path I/O without changing capsule manifest paths."""
import os
from pathlib import Path


def windows_extended_path(value: str) -> str:
    if value.startswith('\\\\?\\'):
        return value
    if value.startswith('\\\\'):
        return '\\\\?\\UNC\\'+value[2:]
    return '\\\\?\\'+value


def capsule_storage_path(path: Path) -> Path:
    resolved=path.resolve()
    if os.name=='nt':
        return Path(windows_extended_path(str(resolved)))
    return resolved
