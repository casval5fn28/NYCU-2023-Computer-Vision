"""OpenFace for PyTorch using the PyTorch weights `net.pth`."""
from . import net

import pathlib

openface_model_path = str(pathlib.Path(__file__).resolve().parent / "net.pth")
