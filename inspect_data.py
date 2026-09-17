import os
import sys
import platform
import json
import importlib

packages_to_check = ['cv2', 'numpy', 'scipy', 'skimage', 'sklearn', 'pandas', 'matplotlib', 'streamlit']
installed_packages = {}

for pkg in packages_to_check:
    try:
        module = importlib.import_module(pkg)
        installed_packages[pkg] = getattr(module, '__version__', 'unknown')
    except ImportError:
        installed_packages[pkg] = 'Not installed'

env_info = {
    'os': f"{platform.system()} {platform.release()}",
    'python_version': sys.version,
    'python_executable': sys.executable,
    'in_venv': sys.prefix != sys.base_prefix,
    'packages': installed_packages
}

print("--- ENV INFO ---")
print(json.dumps(env_info, indent=2))
