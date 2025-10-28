"""localMode package"""

__all__ = [
    "cellMode",
    "faceMode"
]

from .cellmode import cellMode
from .facemode import faceMode

import subprocess
import sys

def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name, github_url):
    if not is_package_installed(package_name):
        print(f"{package_name} not found. Installing from {github_url}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name} @ {github_url}"])

# Define your dependencies
dependencies = {
    "foamToPython": "git+https://github.com/Ruansh233/foamToPython.git",
    "PODImodels": "git+https://github.com/Ruansh233/PODImodels.git"
}

# Install dependencies
for package, url in dependencies.items():
    install_package(package, url)
