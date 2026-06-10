"""Constants and small config dataclass.

This file contains only data/constants and is safe to import at build time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

NON_MACHINE_DEPS: List[str] = [
    "ultralytics>=8.0.0",
    "numpy>=1.26.0",
    "av>=10.0.0",
    "Pillow>=10.0.0",
    "aiohttp>=3.8.0",
    "aiortc>=1.14.0",
    "boto3>=1.28.0",
    "flask>=3.0.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    "rich>=13.6.0",
    "filterpy>=1.4.5",
    "scipy>=1.11.0",
    "lap>=0.4.0",
    "tqdm>=4.66.1",
    # Dev/testing (optional)
    "pytest>=7.4.0",
    "black>=23.9.1",
    "isort>=5.12.0",
    "flake8>=6.1.0",
]

# BUG-39 fix: PYTHON_SUPPORT_MAP was defined here, imported in core.py, but
# never actually read by any code path in the installer — it was dead data.
# Rather than delete it outright (it may be useful for future validation logic),
# we:
#   1. Document its intended purpose clearly.
#   2. Update the min-Python values to 3.10 to match the project requirement
#      (aligns with BUG-40 fix in validate_python_version).
#   3. Add a helper so callers can use the map without duplicating the lookup.
#
# Format: package_name -> (min_major, min_minor, max_major, max_minor)
PYTHON_SUPPORT_MAP: Dict[str, tuple] = {
    "numpy":        (3, 10, 3, 12),
    "scipy":        (3, 10, 3, 12),
    "torch":        (3, 10, 3, 12),
    "torchvision":  (3, 10, 3, 12),
    "torchaudio":   (3, 10, 3, 12),
    "opencv":       (3, 10, 3, 12),
    "av":           (3, 10, 3, 12),
    "aiortc":       (3, 10, 3, 12),
    "aiohttp":      (3, 10, 3, 12),
    "Pillow":       (3, 10, 3, 12),
    "boto3":        (3, 10, 3, 12),
    "flask":        (3, 10, 3, 12),
    "fastapi":      (3, 10, 3, 12),
    "uvicorn":      (3, 10, 3, 12),
    "pyyaml":       (3, 10, 3, 12),
    "rich":         (3, 10, 3, 12),
    "tqdm":         (3, 10, 3, 12),
    "cuda-python":  (3, 10, 3, 12),
    "nvidia-ml-py": (3, 10, 3, 12),
}


def check_python_support(package_name: str) -> bool:
    """
    Return True if the running Python version is within the supported range
    for the given package name as listed in PYTHON_SUPPORT_MAP.

    Returns True when the package is not in the map (unknown = assume OK).
    This helper is the intended consumption point for PYTHON_SUPPORT_MAP.
    """
    import sys  # local import keeps this module safe at build time
    entry = PYTHON_SUPPORT_MAP.get(package_name)
    if entry is None:
        return True  # not in map → assume supported
    min_maj, min_min, max_maj, max_min = entry
    cur = (sys.version_info.major, sys.version_info.minor)
    return (min_maj, min_min) <= cur <= (max_maj, max_min)


@dataclass
class Config:
    """Basic defaults used by the installer (kept minimal here)."""

    # UI / metrics defaults
    metrics_file: str = "install_metrics.json"
    metrics_dir: str = "./logs"

    # UI tuning
    heartbeat: int = 4
    always_progress: bool = False

    # other defaults can be referenced in code, but keep config small here
