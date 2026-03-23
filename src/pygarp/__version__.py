import importlib.metadata

try:
    __version__ = importlib.metadata.version("pygarp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = []
