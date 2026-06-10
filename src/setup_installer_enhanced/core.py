"""Core installer: PackageTask dataclass + Installer class.

This module contains the heavy logic. It imports only from .utils and .constants
so importing it will not invoke argparse; however do avoid importing core from
build-time tooling unless you intend to run the installer.
"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import queue
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from .utils import (
    log,
    run,
    pip_show,
    importable,
    package_already_present,
    ensure_rich,
    RICH,
    CONSOLE,
)
from .constants import NON_MACHINE_DEPS, PYTHON_SUPPORT_MAP, Config


# Keep PackageTask unchanged
@dataclass
class PackageTask:
    name: str
    wheel_spec: Optional[str] = None
    install_args: Optional[List[str]] = None
    optional: bool = False
    reason: Optional[str] = None


class Installer:
    """The big Installer class — logic preserved from original installer.py.

    NOTE: This class expects an argparse.Namespace 'args' matching the CLI flags.
    The rest of the code is intentionally unchanged except for using log/run/etc
    from utils rather than module-level definitions.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # BUG-36 fix: tempdir was created in __init__ but never used for any
        # intermediate work — it existed as unused scaffolding.  The directory is
        # retained so cleanup() can still call shutil.rmtree() without error, but
        # actual temp work (e.g. torch wheel validation) now uses function-local
        # tempfile.mkdtemp() calls that are cleaned up in their own finally-blocks.
        self.tempdir = tempfile.mkdtemp(prefix="setup_installer_")
        self.start = time.time()
        self.summary = {"installed": [], "skipped": [], "failed": [], "already": []}
        self.metrics: Dict[str, dict] = {}
        # logfile (append, line-buffered)
        self.logfile = open(
            self.args.metrics_file.replace(".json", "_full.log"), "a", buffering=1
        )
        self.heartbeat = max(2, getattr(self.args, "heartbeat", 4))

    # --------------------- system detection ---------------------
    def validate_python_version(self):
        """
        BUG-40 fix: was checking >= 3.8 but the project requires Python 3.10+
        (documented in README.md and enforced by type-annotation syntax used
        throughout the codebase, e.g. `dict | None` union syntax requires 3.10+).
        Updated minimum to (3, 10).
        """
        major, minor = sys.version_info[:2]
        if (major, minor) < (3, 10):
            log(
                "error",
                f"Unsupported Python {major}.{minor}. Python >= 3.10 is required.",
            )
            sys.exit(1)
        else:
            log("ok", f"Python version {major}.{minor} is supported.")

    def detect_system(self) -> dict:
        cur_py = (sys.version_info.major, sys.version_info.minor)
        log("info", f"Current Python: {cur_py[0]}.{cur_py[1]}")
        n = self._detect_nvidia_smi_basic()
        if n:
            log(
                "ok",
                f"nvidia-smi found: GPUs={n.get('gpus')} driver={n.get('driver')} names={n.get('names')}",
            )
        else:
            log("warn", "nvidia-smi not found or returned non-zero; GPU info unknown")
        ff = self.detect_ffmpeg()
        if ff:
            log("ok", "ffmpeg present on PATH")
        else:
            log("warn", "ffmpeg not detected; some packages may require it installed")
        head = self.is_headless()
        if head:
            log(
                "warn",
                "Headless environment detected; will prefer opencv-python-headless",
            )
        else:
            log("info", "Display server looks available; GPU-capable OpenCV wheel ok")
        return {"nvidia": n, "ffmpeg": ff, "headless": head}

    def _detect_nvidia_smi_basic(self) -> Optional[dict]:
        # BUG-51 fix: also query compute_cap (e.g. "12.0" for Blackwell /
        # RTX 50-series, "9.0" for Hopper, "8.9" for Ada/RTX 40-series).
        # This tells us the minimum torch version that ships kernels for
        # this GPU's architecture, independent of the driver/CUDA-runtime
        # check done elsewhere. Older nvidia-smi builds may not support the
        # "compute_cap" field, so fall back to the basic query if it errors.
        code, out, err = run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,count,compute_cap",
                "--format=csv,noheader",
            ]
        )
        has_compute_cap = True
        if code != 0 or not out.strip():
            has_compute_cap = False
            log(
                "info",
                "nvidia-smi --query-gpu=...,compute_cap failed "
                f"(exit {code}); retrying without compute_cap field. "
                f"stderr={err!r}",
            )
            code, out, err = run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,count",
                    "--format=csv,noheader",
                ]
            )
        if code != 0 or not out.strip():
            log("warn", f"nvidia-smi query failed entirely (exit {code}): {err}")
            return None
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        names = []
        driver = None
        total = 0
        compute_caps = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            min_parts = 4 if has_compute_cap else 3
            if len(parts) >= min_parts:
                names.append(parts[0])
                driver = parts[1]
                try:
                    total += int(parts[2])
                except Exception:
                    total += 1
                if has_compute_cap:
                    try:
                        compute_caps.append(float(parts[3]))
                    except Exception:
                        log(
                            "info",
                            f"Could not parse compute_cap from nvidia-smi line: {ln!r}",
                        )
        compute_cap = max(compute_caps) if compute_caps else None
        result = {
            "gpus": total,
            "names": names,
            "driver": driver,
            "compute_cap": compute_cap,
        }
        log(
            "info",
            f"nvidia-smi GPU detection: gpus={total} names={names} "
            f"driver={driver} compute_cap={compute_cap}",
        )
        return result

    def detect_ffmpeg(self) -> bool:
        code, out, err = run(["ffmpeg", "-version"])
        return code == 0

    def is_headless(self) -> bool:
        """
        Detect whether the current environment lacks a display server.

        BUG-33 fix: the original logic checked DISPLAY/WAYLAND_DISPLAY on both
        Linux AND macOS.  On macOS, neither env var is set by default (the display
        is managed by the WindowServer, not via X11/Wayland), so the function
        incorrectly returned True — causing the installer to always prefer
        opencv-python-headless on macOS even when a full GUI is available.

        Fix: macOS always has a display server (WindowServer) and is never
        considered headless.  Only apply the DISPLAY/WAYLAND_DISPLAY check on
        Linux.  Windows always returns False (handled implicitly by the else).
        """
        if sys.platform == "darwin":
            # macOS: WindowServer is always present; never headless.
            return False
        if sys.platform.startswith("linux"):
            # Linux: headless if neither X11 nor Wayland display is available.
            return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        # Windows and other platforms: assume a display is present.
        return False

    # --------------------- queue planning ---------------------
    def plan_queue(self, env: dict) -> List[PackageTask]:
        q: List[PackageTask] = []
        if getattr(self.args, "torch_version", None):
            torch_spec = (
                f"torch=={self.args.torch_version}+{self.args.cuda_tag}"
                if getattr(self.args, "cuda_tag", None)
                else f"torch=={self.args.torch_version}"
            )
            q.append(
                PackageTask(
                    name="torch",
                    wheel_spec=torch_spec,
                    install_args=[
                        "-i",
                        self.args.index_url,
                        "--extra-index-url",
                        self.args.extra_index_url,
                    ],
                    optional=False,
                    reason="PyTorch (user-pinned)",
                )
            )
            if getattr(self.args, "torchvision_version", None):
                q.append(
                    PackageTask(
                        name="torchvision",
                        wheel_spec=f"torchvision=={self.args.torchvision_version}+{self.args.cuda_tag}",
                        install_args=[
                            "-i",
                            self.args.index_url,
                            "--extra-index-url",
                            self.args.extra_index_url,
                        ],
                        optional=True,
                    )
                )
            if getattr(self.args, "torchaudio_version", None):
                q.append(
                    PackageTask(
                        name="torchaudio",
                        wheel_spec=f"torchaudio=={self.args.torchaudio_version}+{self.args.cuda_tag}",
                        install_args=[
                            "-i",
                            self.args.index_url,
                            "--extra-index-url",
                            self.args.extra_index_url,
                        ],
                        optional=True,
                    )
                )
        elif not getattr(self.args, "auto_detect_torch", False):
            log(
                "warn",
                "PyTorch auto-install skipped (no --auto-detect-torch and no --torch-version provided)",
            )

        if getattr(self.args, "install_cuda_python", False):
            q.append(
                PackageTask(
                    name="cuda-python",
                    optional=True,
                    reason="cuda-python: python bindings for CUDA runtime",
                )
            )
        if getattr(self.args, "install_nvidia_ml", False):
            q.append(
                PackageTask(
                    name="nvidia-ml-py", optional=True, reason="pynvml / nvidia-ml-py"
                )
            )

        opencv_pkg = (
            "opencv-python-headless" if env.get("headless") else "opencv-python"
        )
        q.append(
            PackageTask(
                name=opencv_pkg,
                wheel_spec=f"{opencv_pkg}>={getattr(self.args,'opencv_min','4.8.0')}",
                optional=False,
            )
        )
        q.append(
            PackageTask(
                name="ffmpeg", optional=True, reason="system binary; validated only"
            )
        )

        for spec in NON_MACHINE_DEPS:
            token = spec.split()[0].split("=")[0].split(">")[0]
            q.append(
                PackageTask(
                    name=token, wheel_spec=spec, optional=True, reason="non-machine dep"
                )
            )

        return q

    # (All the remaining methods — detect_cuda_runtime, cuda_runtime_to_tags,
    # try_install_torch_trio, get_installed_torch_info, trio_needs_install,
    # uninstall_trio, run_stream, install_task, execute, _print_summary, cleanup)
    #
    # For brevity here I include them unchanged from your original file.
    # Paste the body of each method from your installer.py into this class
    # (they use run/log/package_already_present/ensure_rich which are imported above).
    #
    # Because you requested NO logic changes, keep implementations identical.
    #

    # --- To avoid truncation in this message, we include the remainder of the
    # methods below exactly as they appeared in your installer.py. ---
    # (Start of large unchanged code)
    def detect_cuda_runtime(self) -> Optional[str]:
        queries = [
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            [
                "nvidia-smi",
                "--query-gpu=driver_version,cuda_version",
                "--format=csv,noheader",
            ],
        ]
        for q in queries:
            code, out, err = run(q)
            if code == 0 and out.strip():
                first = out.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",") if p.strip()]
                for p in reversed(parts):
                    if p and any(ch.isdigit() for ch in p):
                        if "." in p or p.isdigit():
                            return p
        code, out, err = run(["nvidia-smi", "-q"])
        if code == 0 and out:
            for line in out.splitlines():
                if "CUDA Version" in line:
                    try:
                        return line.split(":")[-1].strip()
                    except Exception:
                        pass
        code, out, err = run(["nvcc", "--version"])
        if code == 0 and out:
            for line in out.splitlines():
                if "release" in line:
                    try:
                        seg = line.split("release")[-1].split(",")[0].strip()
                        if seg:
                            return seg
                    except Exception:
                        pass
        return None

    def cuda_runtime_to_tags(self, cuda_runtime: Optional[str]) -> List[str]:
        if not cuda_runtime:
            candidates = ["cu130", "cu128", "cu121", "cu118"]
        else:
            seg = cuda_runtime.split(".")
            try:
                major = int(seg[0])
                minor = int(seg[1]) if len(seg) > 1 else 0
                candidates = [f"cu{major}{minor}"]
                if minor >= 1:
                    candidates.append(f"cu{major}{minor-1}")
                if major >= 13:
                    candidates += ["cu130", "cu128", "cu121"]
                elif major == 12:
                    candidates += ["cu128", "cu121", "cu118"]
                elif major == 11:
                    candidates += ["cu118", "cu117", "cu116"]
                else:
                    candidates += ["cu118", "cu121"]
            except Exception:
                candidates = ["cu130", "cu128", "cu121", "cu118"]
        seen = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        seen.append("cpu")
        return seen

    # BUG-51: known-good pinned torch/torchvision/torchaudio combos, used as
    # a last-resort fallback in try_install_torch_trio() when "latest torch"
    # fails torch.cuda.is_available() on every driver-compatible CUDA tag.
    # Each entry is (min_compute_capability, cuda_tag, torch_ver,
    # torchvision_ver, torchaudio_ver). The first entry whose
    # min_compute_capability the detected GPU meets/exceeds is used,
    # provided its cuda_tag is in the driver-compatible candidate list.
    #   - 12.0 = Blackwell (RTX 50-series / B100 etc.) -> needs torch >= 2.7
    #   -  9.0 = Hopper / Ada / Ampere and newer
    #   -  0.0 = catch-all for older GPUs
    _PINNED_TORCH_FALLBACKS = [
        (12.0, "cu128", "2.7.1", "0.22.1", "2.7.1"),
        (9.0, "cu121", "2.5.1", "0.20.1", "2.5.1"),
        (0.0, "cu118", "2.5.1", "0.20.1", "2.5.1"),
    ]

    def _try_pinned_fallback(
        self,
        gpu_info: dict,
        candidates: List[str],
        no_deps: bool,
        tv_ver: Optional[str],
        ta_ver: Optional[str],
    ) -> bool:
        """BUG-51 fix: install one known-good pinned torch/vision/audio combo.

        Used when "latest torch" failed torch.cuda.is_available() on every
        driver-compatible CUDA tag (a real-world case: RTX 5090/Blackwell +
        driver supporting up to CUDA 12.8, where "latest" torch is built
        against CUDA 13 and refuses to initialize on this driver, while older
        cuXXX-pinned torch builds predate Blackwell kernel support).
        """
        cc = gpu_info.get("compute_cap")
        log(
            "info",
            f"BUG-51 pinned-fallback: GPU compute capability detected as "
            f"{cc!r}; selecting a pinned torch build known to support both "
            "this architecture and this driver's CUDA ceiling.",
        )
        if cc is None:
            log(
                "warn",
                "BUG-51 pinned-fallback: nvidia-smi did not report a compute "
                "capability (older nvidia-smi?) — cannot safely pick a "
                "pinned build. Skipping pinned fallback.",
            )
            return False

        chosen = None
        for min_cc, cuda_tag, t_ver, tv_pin, ta_pin in self._PINNED_TORCH_FALLBACKS:
            if cc >= min_cc:
                chosen = (min_cc, cuda_tag, t_ver, tv_pin, ta_pin)
                break

        if chosen is None:
            log(
                "warn",
                f"BUG-51 pinned-fallback: no pinned combo matches compute "
                f"capability {cc}. Skipping pinned fallback.",
            )
            return False

        min_cc, cuda_tag, t_ver, tv_pin, ta_pin = chosen
        if tv_ver:
            tv_pin = tv_ver
        if ta_ver:
            ta_pin = ta_ver

        if cuda_tag not in candidates:
            log(
                "warn",
                f"BUG-51 pinned-fallback: pinned combo for compute capability "
                f">= {min_cc} needs CUDA tag '{cuda_tag}', but that tag is "
                f"not in this driver's compatible tag list {candidates}. "
                "The driver may be too old even for the pinned build. "
                "Skipping pinned fallback.",
            )
            return False

        index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
        pkg_specs = [
            f"torch=={t_ver}+{cuda_tag}",
            f"torchvision=={tv_pin}+{cuda_tag}",
            f"torchaudio=={ta_pin}+{cuda_tag}",
        ]
        log(
            "info",
            f"BUG-51 pinned-fallback: installing pinned trio "
            f"{pkg_specs} from {index_url} "
            f"(matched compute capability {cc} >= {min_cc}).",
        )
        install_cmd = (
            [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
            + pkg_specs
            + ["-i", index_url, "--extra-index-url", self.args.extra_index_url]
        )
        if no_deps:
            install_cmd.append("--no-deps")
        if "--progress-bar=on" not in install_cmd:
            install_cmd += ["--progress-bar=on", "-v"]

        log(
            "info",
            "BUG-51 pinned-fallback: running pip install: "
            + " ".join(install_cmd),
        )
        code, out, err = self.run_stream(
            install_cmd, task_name="torch.trio.pinned-fallback", show_stdout=True
        )
        if code != 0:
            log(
                "warn",
                f"BUG-51 pinned-fallback: pip install of pinned trio failed "
                f"(exit {code}). stdout={out} stderr={err}",
            )
            return False

        log(
            "ok",
            f"BUG-51 pinned-fallback: installed pinned torch=={t_ver}+{cuda_tag} "
            "trio successfully. Verifying torch.cuda.is_available()...",
        )
        if self._verify_torch_cuda_usable():
            log(
                "ok",
                f"BUG-51 pinned-fallback: torch=={t_ver}+{cuda_tag} verified "
                "working (torch.cuda.is_available() == True).",
            )
            return True

        log(
            "warn",
            f"BUG-51 pinned-fallback: pinned torch=={t_ver}+{cuda_tag} also "
            "failed torch.cuda.is_available(). Uninstalling before "
            "continuing to CPU-only fallback.",
        )
        self.uninstall_trio()
        return False

    def try_install_torch_trio(
        self, candidates: List[str], no_deps: bool = False
    ) -> bool:
        log(
            "info",
            f"Attempting PyTorch trio auto-install with candidates: {candidates}",
        )
        tv_ver = getattr(self.args, "torchvision_version", None)
        ta_ver = getattr(self.args, "torchaudio_version", None)
        pinned_fallback_tried = False

        # BUG-51 fix (reorder for Blackwell+): for GPUs with compute
        # capability >= 12.0 (RTX 50-series / Blackwell and newer), we
        # already KNOW from experience that "latest torch" under every
        # cuXXX index resolves to the same CUDA-13-based build that fails
        # torch.cuda.is_available() on current drivers — so walking the
        # full cu128 -> cu127 -> cu121 -> cu118 ladder first just wastes
        # several ~530MB downloads/uninstalls before reaching the pinned
        # fallback anyway. For these GPUs ONLY, try the pinned fallback
        # FIRST. If it fails for any reason, fall through to the normal
        # ladder below unchanged (so the existing safety net still applies).
        # GPUs below compute_cap 12.0 (4090, 3090, etc.) are completely
        # unaffected and keep the original ladder-first order.
        gpu_info_early = self._detect_nvidia_smi_basic()
        if gpu_info_early and gpu_info_early.get("gpus", 0) > 0:
            cc_early = gpu_info_early.get("compute_cap")
            if cc_early is not None and cc_early >= 12.0:
                log(
                    "info",
                    f"BUG-51: detected Blackwell-class GPU (compute_cap="
                    f"{cc_early}, {gpu_info_early.get('names')}) before "
                    "trying the normal CUDA-tag ladder. 'Latest' torch is "
                    "known to be incompatible with this driver/architecture "
                    "combo across all cuXXX tags, so trying the pinned "
                    "fallback build FIRST to avoid wasted downloads.",
                )
                pinned_fallback_tried = True
                if self._try_pinned_fallback(
                    gpu_info_early, candidates, no_deps, tv_ver, ta_ver
                ):
                    return True
                log(
                    "warn",
                    "BUG-51: pinned-fallback-first attempt did not succeed; "
                    "falling back to the normal CUDA-tag ladder "
                    f"({candidates}) as a safety net.",
                )

        for cand in candidates:
            if cand == "cpu":
                # BUG-51 fix: before giving up and installing CPU-only torch,
                # check whether every GPU candidate failed verification not
                # because the GPU is unsupported, but because "latest torch"
                # under every cuXXX index resolves to the SAME build (e.g.
                # torch 2.12.0/CUDA13) which this driver can't run, AND/OR
                # this GPU's architecture (e.g. Blackwell/RTX 50-series,
                # compute_cap 12.0) needs torch >= 2.7 for kernel support.
                # In that case, try one known-good PINNED torch/vision/audio
                # combo before falling back to CPU-only.
                if not pinned_fallback_tried:
                    pinned_fallback_tried = True
                    gpu_info = self._detect_nvidia_smi_basic()
                    if gpu_info and gpu_info.get("gpus", 0) > 0:
                        log(
                            "warn",
                            "All driver-detected CUDA tags failed verification "
                            f"({[c for c in candidates if c != 'cpu']}). "
                            "An NVIDIA GPU is present "
                            f"({gpu_info.get('names')}, compute_cap="
                            f"{gpu_info.get('compute_cap')}) — trying a known-good "
                            "pinned torch build before falling back to CPU-only "
                            "(BUG-51).",
                        )
                        if self._try_pinned_fallback(
                            gpu_info, candidates, no_deps, tv_ver, ta_ver
                        ):
                            return True
                        log(
                            "warn",
                            "Pinned-fallback (BUG-51) also failed verification; "
                            "proceeding with CPU-only torch as last resort.",
                        )
                    else:
                        log(
                            "info",
                            "No NVIDIA GPU detected via nvidia-smi; proceeding "
                            "directly with CPU-only torch.",
                        )
                index_args = ["-i", self.args.extra_index_url]
                index_label = self.args.extra_index_url
            else:
                index_args = ["-i", f"https://download.pytorch.org/whl/{cand}"]
                index_label = index_args[1]

            check_spec = "torch"
            log(
                "info",
                f"Validating torch availability on index {index_label} (candidate {cand})...",
            )
            tmpd = tempfile.mkdtemp(prefix="torch_validate_")
            try:
                cmd = (
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        "--no-deps",
                        "--only-binary=:all:",
                        check_spec,
                        "-d",
                        tmpd,
                    ]
                    + index_args
                    + ["--extra-index-url", self.args.extra_index_url]
                )
                code, out, err = self.run_stream(
                    cmd, task_name="torch.validate", show_stdout=True
                )
                files = os.listdir(tmpd) if os.path.isdir(tmpd) else []
                if code == 0 and any(f.endswith(".whl") for f in files):
                    log(
                        "ok",
                        f"Found torch wheel(s) on {index_label}. Proceeding to install trio using that index.",
                    )
                    pkg_names = ["torch"]
                    if tv_ver:
                        pkg_names.append(f"torchvision=={tv_ver}")
                    else:
                        pkg_names.append("torchvision")
                    if ta_ver:
                        pkg_names.append(f"torchaudio=={ta_ver}")
                    else:
                        pkg_names.append("torchaudio")

                    install_cmd = (
                        [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
                        + pkg_names
                        + index_args
                        + ["--extra-index-url", self.args.extra_index_url]
                    )
                    if no_deps:
                        install_cmd.append("--no-deps")
                    if "--progress-bar=on" not in install_cmd:
                        install_cmd += ["--progress-bar=on", "-v"]
                    code2, out2, err2 = self.run_stream(
                        install_cmd, task_name="torch.trio", show_stdout=True
                    )
                    if code2 == 0:
                        log(
                            "ok",
                            f"Installed torch trio from {index_label} (candidate {cand}) successfully.",
                        )
                        # BUG-50 fix: a successful pip install only proves the
                        # wheel exists and installs cleanly — it does NOT
                        # prove the GPU build actually works on this
                        # machine's CUDA driver (e.g. a cu130 build can
                        # install fine on a driver that only supports CUDA
                        # 12.7, but torch.cuda.is_available() then returns
                        # False). When this is a GPU candidate and an NVIDIA
                        # GPU is present, verify CUDA actually initializes;
                        # if not, uninstall and fall back to the next
                        # (older) candidate tag instead of reporting success.
                        if cand != "cpu" and self._detect_nvidia_smi_basic():
                            if self._verify_torch_cuda_usable():
                                return True
                            log(
                                "warn",
                                f"torch installed from {index_label} (candidate {cand}) "
                                "but torch.cuda.is_available() is False on this machine "
                                "(likely a CUDA build too new for the installed driver). "
                                "Uninstalling and trying an older CUDA tag...",
                            )
                            self.uninstall_trio()
                            continue
                        return True
                    else:
                        log(
                            "warn", f"Install failed on candidate {cand}: {out2} {err2}"
                        )
                else:
                    log(
                        "warn",
                        f"No wheel for torch on {index_label}. pip output: {out} {err}",
                    )
            finally:
                try:
                    shutil.rmtree(tmpd)
                except Exception:
                    pass

        log("error", "Could not install torch trio across all candidates.")
        return False

    def _verify_torch_cuda_usable(self) -> bool:
        """BUG-50 fix: run torch.cuda.is_available() in a *fresh* subprocess.

        We must not just `importlib.import_module("torch")` here: if torch
        was already imported earlier in this process (e.g. by a previous
        candidate attempt or by get_installed_torch_info()), Python would
        return the cached module from sys.modules instead of the
        freshly-installed one, giving a stale/incorrect result. A subprocess
        always sees the package as pip just installed it.
        """
        code, out, err = run(
            [
                sys.executable,
                "-c",
                "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)",
            ]
        )
        if code == 0:
            log("ok", "Verified torch.cuda.is_available() == True for this build.")
            return True
        log(
            "warn",
            f"torch.cuda.is_available() check failed (exit {code}). "
            f"stdout={out!r} stderr={err!r}",
        )
        return False

    def get_installed_torch_info(self) -> dict:
        info = {
            "torch": None,
            "torchvision": None,
            "torchaudio": None,
            "torch_cuda": None,
            "cuda_available": False,
        }
        try:
            tmod = importlib.import_module("torch")
            info["torch"] = getattr(tmod, "__version__", None)
            try:
                info["torch_cuda"] = getattr(tmod, "version").cuda
            except Exception:
                info["torch_cuda"] = None
            try:
                info["cuda_available"] = tmod.cuda.is_available()
            except Exception:
                info["cuda_available"] = False
        except Exception:
            pass
        try:
            tv = importlib.import_module("torchvision")
            info["torchvision"] = getattr(tv, "__version__", None)
        except Exception:
            pass
        try:
            ta = importlib.import_module("torchaudio")
            info["torchaudio"] = getattr(ta, "__version__", None)
        except Exception:
            pass
        return info

    def trio_needs_install(
        self, candidate_tag: Optional[str]
    ) -> Tuple[bool, List[str]]:
        info = self.get_installed_torch_info()
        reasons: List[str] = []
        if not info.get("torch"):
            reasons.append("torch not installed")
            return True, reasons
        if candidate_tag and candidate_tag != "cpu":
            tv = info.get("torch") or ""
            if candidate_tag not in tv:
                tcuda = info.get("torch_cuda")
                if tcuda:
                    # BUG-34 fix: torch.version.cuda can be a 3-part string like
                    # "12.4.1".  str(tcuda).replace('.','') would give "cu1241"
                    # instead of the correct PyPI tag "cu124".  Only take
                    # major.minor (first two dot-separated components).
                    _tcuda_parts = str(tcuda).split(".")
                    _major = _tcuda_parts[0] if len(_tcuda_parts) >= 1 else "0"
                    _minor = _tcuda_parts[1] if len(_tcuda_parts) >= 2 else "0"
                    tag_from_tcuda = f"cu{_major}{_minor}"
                    if tag_from_tcuda != candidate_tag:
                        reasons.append(
                            f"installed torch CUDA tag {tag_from_tcuda} != desired {candidate_tag}"
                        )
                else:
                    reasons.append("installed torch has no cuda metadata to verify tag")
        if not info.get("torchvision"):
            reasons.append("torchvision missing")
        if not info.get("torchaudio"):
            reasons.append("torchaudio missing")
        critical = [r for r in reasons if r.startswith("torch not") or "mismatch" in r]
        if critical or reasons:
            return True, reasons
        return False, []

    def uninstall_trio(self):
        log("info", "Uninstalling torch/torchvision/torchaudio (best-effort)...")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )

    def run_stream(
        self,
        cmd: List[str],
        task_name: Optional[str] = None,
        show_stdout: bool = True,
        show_stderr: bool = True,
        capture: bool = True,
        heartbeat: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        if heartbeat is None:
            heartbeat = self.heartbeat

        if (
            "pip" in " ".join(cmd)
            and "--progress-bar=on" not in cmd
            and getattr(self.args, "always_progress", False)
        ):
            cmd = cmd + ["--progress-bar=on", "-v"]

        if task_name:
            if task_name not in self.metrics:
                self.metrics[task_name] = {
                    "name": task_name,
                    "status": "running",
                    "start_ts": time.time(),
                    "end_ts": None,
                    "downloaded_bytes": 0,
                    "total_bytes": None,
                    "speed_bps": 0.0,
                    "attempts": 1,
                    "logs": [],
                }

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_q = queue.Queue()
        stderr_q = queue.Queue()

        def _reader(pipe, q):
            try:
                for line in iter(pipe.readline, ""):
                    q.put(line)
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t_out = threading.Thread(
            target=_reader, args=(proc.stdout, stdout_q), daemon=True
        )
        t_err = threading.Thread(
            target=_reader, args=(proc.stderr, stderr_q), daemon=True
        )
        t_out.start()
        t_err.start()

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        last_activity = time.time()
        sliding: List[Tuple[float, int]] = []

        re_downloading = re.compile(
            r"Downloading.*\((?P<size>[\d\.]+)\s*(?P<unit>kB|KB|MB|GB)\)", re.IGNORECASE
        )
        re_saved = re.compile(
            r"Saved\s+.*\((?P<size_bytes>\d+)\s*bytes\)", re.IGNORECASE
        )
        re_saved_alt = re.compile(
            r"Downloaded\s+(?P<size>[\d\.]+)\s*(?P<unit>kB|KB|MB|GB)", re.IGNORECASE
        )
        re_using_cached = re.compile(r"Using cached (?P<fname>.*\.whl)", re.IGNORECASE)
        re_success = re.compile(r"Successfully installed (?P<pkgs>.*)", re.IGNORECASE)
        re_collecting = re.compile(
            r"Collecting\s+(?P<name>[\w\-\._\[\]=]+)", re.IGNORECASE
        )

        while True:
            try:
                try:
                    line = stdout_q.get_nowait()
                except queue.Empty:
                    line = None

                if line is None:
                    try:
                        err_line = stderr_q.get_nowait()
                    except queue.Empty:
                        err_line = None
                else:
                    err_line = None

                if line:
                    last_activity = time.time()
                    self.logfile.write(f"[OUT {time.time()}] {' '.join(cmd)} | {line}")
                    if show_stdout:
                        if RICH and CONSOLE:
                            CONSOLE.print(line.rstrip())
                        else:
                            print(line.rstrip())
                    if capture:
                        stdout_lines.append(line)
                    if task_name:
                        m = re_downloading.search(line)
                        if m:
                            size = float(m.group("size"))
                            unit = m.group("unit").lower()
                            if unit.startswith("k"):
                                total = int(size * 1024)
                            elif unit.startswith("m"):
                                total = int(size * 1024 * 1024)
                            elif unit.startswith("g"):
                                total = int(size * 1024 * 1024 * 1024)
                            else:
                                total = int(size)
                            self.metrics[task_name]["total_bytes"] = total
                        m2 = re_saved.search(line)
                        if m2:
                            try:
                                b = int(m2.group("size_bytes"))
                                self.metrics[task_name]["downloaded_bytes"] = max(
                                    self.metrics[task_name].get("downloaded_bytes", 0),
                                    b,
                                )
                                sliding.append(
                                    (
                                        time.time(),
                                        self.metrics[task_name]["downloaded_bytes"],
                                    )
                                )
                            except Exception:
                                pass
                        m3 = re_saved_alt.search(line)
                        if m3:
                            try:
                                size = float(m3.group("size"))
                                unit = m3.group("unit").lower()
                                if unit.startswith("k"):
                                    b = int(size * 1024)
                                elif unit.startswith("m"):
                                    b = int(size * 1024 * 1024)
                                elif unit.startswith("g"):
                                    b = int(size * 1024 * 1024 * 1024)
                                else:
                                    b = int(size)
                                self.metrics[task_name]["downloaded_bytes"] = max(
                                    self.metrics[task_name].get("downloaded_bytes", 0),
                                    b,
                                )
                                sliding.append(
                                    (
                                        time.time(),
                                        self.metrics[task_name]["downloaded_bytes"],
                                    )
                                )
                            except Exception:
                                pass
                        m4 = re_using_cached.search(line)
                        if m4:
                            self.metrics[task_name]["status"] = "using_cached"
                        m5 = re_success.search(line)
                        if m5:
                            self.metrics[task_name]["status"] = "installed"
                            self.metrics[task_name]["end_ts"] = time.time()
                    continue

                if err_line:
                    last_activity = time.time()
                    self.logfile.write(
                        f"[ERR {time.time()}] {' '.join(cmd)} | {err_line}"
                    )
                    if show_stderr:
                        if RICH and CONSOLE:
                            CONSOLE.print(f"[red]{err_line.rstrip()}[/red]")
                        else:
                            print(err_line.rstrip(), file=sys.stderr)
                    if capture:
                        stderr_lines.append(err_line)
                    if task_name:
                        m2 = re_saved.search(err_line)
                        if m2:
                            try:
                                b = int(m2.group("size_bytes"))
                                self.metrics[task_name]["downloaded_bytes"] = max(
                                    self.metrics[task_name].get("downloaded_bytes", 0),
                                    b,
                                )
                                sliding.append(
                                    (
                                        time.time(),
                                        self.metrics[task_name]["downloaded_bytes"],
                                    )
                                )
                            except Exception:
                                pass
                        m5 = re_success.search(err_line)
                        if m5:
                            self.metrics[task_name]["status"] = "installed"
                            self.metrics[task_name]["end_ts"] = time.time()
                    continue

                if proc.poll() is not None:
                    while not stdout_q.empty():
                        l = stdout_q.get_nowait()
                        self.logfile.write(f"[OUT {time.time()}] {' '.join(cmd)} | {l}")
                        if show_stdout:
                            if RICH and CONSOLE:
                                CONSOLE.print(l.rstrip())
                            else:
                                print(l.rstrip())
                        if capture:
                            stdout_lines.append(l)
                    while not stderr_q.empty():
                        l = stderr_q.get_nowait()
                        self.logfile.write(f"[ERR {time.time()}] {' '.join(cmd)} | {l}")
                        if show_stderr:
                            if RICH and CONSOLE:
                                CONSOLE.print(f"[red]{l.rstrip()}[/red]")
                            else:
                                print(l.rstrip(), file=sys.stderr)
                        if capture:
                            stderr_lines.append(l)
                    break

                if time.time() - last_activity > heartbeat:
                    if task_name and self.metrics.get(task_name):
                        if len(sliding) >= 2:
                            t0, b0 = sliding[0]
                            t1, b1 = sliding[-1]
                            dt = max(0.001, t1 - t0)
                            db = max(0, b1 - b0)
                            speed = db / dt
                            self.metrics[task_name]["speed_bps"] = (
                                0.8 * self.metrics[task_name].get("speed_bps", 0)
                                + 0.2 * speed
                            )
                        db = self.metrics[task_name].get("downloaded_bytes", 0)
                        tb = self.metrics[task_name].get("total_bytes")
                        if RICH and CONSOLE:
                            if tb:
                                CONSOLE.print(
                                    f"[cyan][{time.strftime('%Y-%m-%d %H:%M:%S')}] {task_name}: {db/1024/1024:.2f} MB / {tb/1024/1024:.2f} MB • {self.metrics[task_name].get('speed_bps',0)/1024/1024:.2f} MB/s[/cyan]"
                                )
                            else:
                                CONSOLE.print(
                                    f"[cyan][{time.strftime('%Y-%m-%d %H:%M:%S')}] {task_name}: {db/1024/1024:.2f} MB downloaded • {self.metrics[task_name].get('speed_bps',0)/1024/1024:.2f} MB/s[/cyan]"
                                )
                        else:
                            print(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {task_name}: {db/1024/1024:.2f} MB downloaded"
                            )
                    else:
                        if RICH and CONSOLE:
                            CONSOLE.print(
                                f"[cyan][{time.strftime('%Y-%m-%d %H:%M:%S')}] running: {' '.join(cmd)}[/cyan]"
                            )
                        else:
                            print(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] running: {' '.join(cmd)}"
                            )
                    last_activity = time.time()
                time.sleep(0.1)
            except KeyboardInterrupt:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise

        code = proc.returncode
        if task_name and self.metrics.get(task_name):
            if self.metrics[task_name].get("end_ts") is None:
                self.metrics[task_name]["end_ts"] = time.time()
            if code == 0 and self.metrics[task_name].get("status") != "installed":
                self.metrics[task_name]["status"] = "installed"
        return (
            code,
            "".join(stdout_lines) if capture else "",
            "".join(stderr_lines) if capture else "",
        )

    def install_task(self, task: PackageTask) -> bool:
        if task.name == "ffmpeg":
            ok = self.detect_ffmpeg()
            if ok:
                self.summary["already"].append("ffmpeg")
                log("ok", "ffmpeg available on PATH")
                return True
            else:
                # Note: ffmpeg is optional (validated only, not pip-installable),
                # so it is intentionally NOT added to summary["failed"] — that
                # list is used by the CLI to decide whether to abort and skip
                # --run-after, and a missing ffmpeg shouldn't block that.
                log(
                    "warn",
                    "ffmpeg not found on PATH. Install it manually: "
                    "Windows (winget install Gyan.FFmpeg / choco install ffmpeg), "
                    "macOS (brew install ffmpeg), Linux (apt/dnf/pacman install ffmpeg).",
                )
                return False

        if task.reason == "non-machine dep" or task.name in [
            t.split("==")[0] for t in NON_MACHINE_DEPS
        ]:
            if package_already_present(task.wheel_spec or task.name):
                self.summary["already"].append(task.name)
                log("info", f"{task.name} already present; skipping install")
                return True

        # OpenCV variant handling: this machine needs exactly ONE of
        # opencv-python / opencv-python-headless (they both ship the `cv2`
        # module and silently fight over the same namespace if both are
        # present). Rather than blindly installing the variant the queue
        # planned (based on headless detection), check what's *actually*
        # on this machine first:
        #   - desired variant already present, no conflict -> skip (already)
        #   - conflicting variant present -> uninstall it (it's wrong for
        #     this machine's display capability), then install/keep desired
        #   - neither present -> fall through to normal pip install below
        if task.name in ("opencv-python", "opencv-python-headless"):
            conflict = (
                "opencv-python-headless"
                if task.name == "opencv-python"
                else "opencv-python"
            )
            desired_info = pip_show(task.name)
            conflict_info = pip_show(conflict)

            if desired_info and not conflict_info:
                self.summary["already"].append(task.name)
                log(
                    "info",
                    f"{task.name} (v{desired_info.get('Version','?')}) already installed "
                    f"and matches this machine; skipping.",
                )
                return True

            if conflict_info:
                log(
                    "warn",
                    f"Conflicting OpenCV variant '{conflict}' (v{conflict_info.get('Version','?')}) "
                    f"is installed, but this machine needs '{task.name}'. Both variants provide "
                    f"the 'cv2' module and conflict — uninstalling '{conflict}' first.",
                )
                run([sys.executable, "-m", "pip", "uninstall", "-y", conflict])
                if desired_info:
                    self.summary["already"].append(task.name)
                    log(
                        "ok",
                        f"{task.name} remains installed after removing the conflicting variant.",
                    )
                    return True
                # desired variant not installed yet -> fall through to install it

        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if task.wheel_spec:
            pip_cmd.append(task.wheel_spec)
        else:
            pip_cmd.append(task.name)
        if task.install_args:
            pip_cmd.extend(task.install_args)
        if getattr(self.args, "no_deps", False):
            pip_cmd.append("--no-deps")

        if "--progress-bar=on" not in pip_cmd and getattr(
            self.args, "always_progress", False
        ):
            pip_cmd += ["--progress-bar=on", "-v"]

        log("info", f"Installing {task.name}... (this may take a few minutes)")

        # BUG-31 fix: --retries CLI flag was parsed but never applied in install_task().
        # Use it to retry failed installs (network hiccups, transient index errors).
        max_retries = max(1, int(getattr(self.args, "retries", 1)))
        code = -1
        out = err = ""
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                log(
                    "warn", f"Retrying {task.name} (attempt {attempt}/{max_retries})..."
                )
            code, out, err = self.run_stream(
                pip_cmd,
                task_name=f"{task.name}{'_retry' + str(attempt - 1) if attempt > 1 else ''}",
                show_stdout=(getattr(self.args, "verbose", False) or True),
                show_stderr=(getattr(self.args, "verbose", False) or True),
                capture=True,
            )
            if code == 0:
                break

        if code == 0:
            self.summary["installed"].append(task.name)
            log("ok", f"Installed {task.name}")
            return True
        else:
            self.summary["failed"].append(task.name)
            log(
                "error",
                f"Failed to install {task.name} after {max_retries} attempt(s): returncode={code}",
            )
            tail = err.splitlines()[-10:] if err else []
            if tail:
                log("error", "Last pip stderr lines:\n" + "\n".join(tail))
            return False

    def execute(self):
        env = self.detect_system()
        ensure_rich(getattr(self.args, "no_deps", False))
        self.validate_python_version()
        try:
            # rebind rich objects if available
            importlib.invalidate_caches()
            from rich.console import Console as _RichConsole  # type: ignore
            from rich.table import Table as _RichTable  # type: ignore
            from rich.panel import Panel as _RichPanel  # type: ignore

            globals()["CONSOLE"] = _RichConsole()
            globals()["Table"] = _RichTable
            globals()["Panel"] = _RichPanel
            log("ok", "Rich UI bound for phase-2 (Table/Panel available).")
        except Exception as _e:
            log(
                "warn",
                f"Rich UI not bound after install: {_e} — falling back to ANSI logging.",
            )

        missing = [s for s in NON_MACHINE_DEPS if not package_already_present(s)]
        if missing:
            log("info", f"Non-machine packages missing: {missing}")
            pip_cmd = [sys.executable, "-m", "pip", "install"] + missing
            if getattr(self.args, "no_deps", False):
                pip_cmd.append("--no-deps")
            if "--progress-bar=on" not in pip_cmd and getattr(
                self.args, "always_progress", False
            ):
                pip_cmd += ["--progress-bar=on", "-v"]
            code, out, err = self.run_stream(
                pip_cmd, task_name="non-machine", show_stdout=True
            )
            if code == 0:
                log("ok", "Installed missing non-machine packages")
                for spec in missing:
                    token = spec.split()[0].split("=")[0].split(">")[0].strip()
                    if token and token not in self.summary["installed"]:
                        self.summary["installed"].append(token)
                log(
                    "info",
                    f"Marked non-machine packages as installed in summary: {self.summary['installed']}",
                )
            else:
                log("warn", f"Some non-machine installs failed: {err}")
        else:
            log("ok", "All non-machine dependencies already present")

        installer = self
        # BUG-35 fix: local variable `queue` shadowed the stdlib `import queue`
        # module imported at the top of this file.  Renamed to `task_queue` to
        # eliminate the shadowing so queue.Queue/queue.Empty still work correctly
        # in run_stream() within the same method scope.
        task_queue = self.plan_queue(env)

        if getattr(self.args, "auto_detect_torch", False):
            cuda_rt = self.detect_cuda_runtime()
            tags = self.cuda_runtime_to_tags(cuda_rt)
            log("info", f"Auto-detected CUDA runtime: {cuda_rt} -> trying tags {tags}")
            candidate_tag = tags[0] if tags else None

            need_install, reasons = self.trio_needs_install(candidate_tag)
            if not need_install:
                info = self.get_installed_torch_info()
                self.summary["already"].append("torch")
                if info.get("torchvision"):
                    self.summary["already"].append("torchvision")
                if info.get("torchaudio"):
                    self.summary["already"].append("torchaudio")
                log("ok", f"Existing torch trio seems OK: {info}")
                task_queue = [
                    q
                    for q in task_queue
                    if q.name not in ("torch", "torchvision", "torchaudio")
                ]
            else:
                log("warn", f"Torch trio needs install or fix: {reasons}")
                if getattr(self.args, "force_reinstall", False):
                    self.uninstall_trio()
                    installed = self.try_install_torch_trio(
                        tags, no_deps=getattr(self.args, "no_deps", False)
                    )
                    if installed:
                        self.summary["installed"].extend(
                            ["torch", "torchvision", "torchaudio"]
                        )
                        task_queue = [
                            q
                            for q in task_queue
                            if q.name not in ("torch", "torchvision", "torchaudio")
                        ]
                    else:
                        log(
                            "error",
                            "Auto reinstall of trio failed; will fall back to normal queue",
                        )
                else:
                    info = self.get_installed_torch_info()
                    missing = []
                    if not info.get("torch"):
                        missing.append("torch")
                    if not info.get("torchvision"):
                        missing.append("torchvision")
                    if not info.get("torchaudio"):
                        missing.append("torchaudio")
                    if missing:
                        installed = self.try_install_torch_trio(
                            tags, no_deps=getattr(self.args, "no_deps", False)
                        )
                        if installed:
                            self.summary["installed"].extend(missing)
                            task_queue = [
                                q
                                for q in task_queue
                                if q.name not in ("torch", "torchvision", "torchaudio")
                            ]
                        else:
                            log(
                                "warn",
                                "Auto-install attempt for missing trio parts failed; continuing with regular queue",
                            )

        visible_queue = [
            q
            for q in task_queue
            if q.name not in self.summary["already"]
            and q.name not in self.summary["installed"]
        ]

        if RICH and CONSOLE:
            try:
                from rich.table import Table as _Table  # type: ignore

                t = _Table(title="Planned install queue", show_lines=True)
                t.add_column("#", style="bold")
                t.add_column("Package")
                t.add_column("Spec")
                t.add_column("Optional")
                t.add_column("Notes")
                for i, tt in enumerate(visible_queue, 1):
                    t.add_row(
                        str(i),
                        tt.name,
                        tt.wheel_spec or "-",
                        str(tt.optional),
                        tt.reason or "-",
                    )
                CONSOLE.print(t)
            except Exception:
                log("info", "Planned install queue:")
                for i, tt in enumerate(visible_queue, 1):
                    log(
                        "info",
                        f"{i}. {tt.name} spec={tt.wheel_spec or '-'} optional={tt.optional} notes={tt.reason or '-'}",
                    )
        else:
            log("info", "Planned install queue:")
            for i, tt in enumerate(visible_queue, 1):
                log(
                    "info",
                    f"{i}. {tt.name} spec={tt.wheel_spec or '-'} optional={tt.optional} notes={tt.reason or '-'}",
                )

        if getattr(self.args, "dry_run", False):
            log("warn", "Dry-run: not performing any installs. Exiting.")
            return

        for t in visible_queue:
            ok = self.install_task(t)
            if not ok and not t.optional:
                log(
                    "error",
                    f"Required task {t.name} failed — aborting further non-optional installs",
                )
                break

        self._print_summary()

    def _print_summary(self):
        total_time = time.time() - self.start
        try:
            metrics_out = {
                "summary": self.summary,
                "start_ts": self.start,
                "end_ts": time.time(),
                "elapsed_s": total_time,
                "packages": self.metrics,
            }
            with open(self.args.metrics_file, "w") as f:
                json.dump(metrics_out, f, indent=2)
            log("ok", f"Wrote install metrics to {self.args.metrics_file}")
        except Exception as e:
            log("warn", f"Failed to write metrics file: {e}")

        if RICH and CONSOLE:
            try:
                from rich.table import Table as _Table  # type: ignore

                t = _Table(title="Installation summary", show_lines=True)
                t.add_column("Status")
                t.add_column("Packages")
                t.add_row("Installed", ", ".join(self.summary["installed"]) or "-")
                t.add_row(
                    "Already present / Skipped",
                    ", ".join(self.summary["already"]) or "-",
                )
                t.add_row("Failed", ", ".join(self.summary["failed"]) or "-")
                t.add_row("Time(s)", f"{total_time:.1f}s")
                CONSOLE.print(t)
            except Exception:
                log("info", f"Summary Installed: {self.summary['installed']}")
                log("info", f"Already/Skipped: {self.summary['already']}")
                log("info", f"Failed: {self.summary['failed']}")
                log("info", f"Total time: {total_time:.1f}s")
        else:
            log("info", f"Summary Installed: {self.summary['installed']}")
            log("info", f"Already/Skipped: {self.summary['already']}")
            log("info", f"Failed: {self.summary['failed']}")
            log("info", f"Total time: {total_time:.1f}s")

    def cleanup(self):
        try:
            self.logfile.close()
        except Exception:
            pass
        try:
            shutil.rmtree(self.tempdir)
        except Exception:
            pass
            log("ok", f"Wrote install metrics to {self.args.metrics_file}")
        except Exception as e:
            log("warn", f"Failed to write metrics file: {e}")

        if RICH and CONSOLE:
            try:
                from rich.table import Table as _Table  # type: ignore

                t = _Table(title="Installation summary", show_lines=True)
                t.add_column("Status")
                t.add_column("Packages")
                t.add_row("Installed", ", ".join(self.summary["installed"]) or "-")
                t.add_row(
                    "Already present / Skipped",
                    ", ".join(self.summary["already"]) or "-",
                )
                t.add_row("Failed", ", ".join(self.summary["failed"]) or "-")
                t.add_row("Time(s)", f"{self.metrics['elapsed_s']:.1f}s")
                CONSOLE.print(t)
            except Exception:
                log("info", f"Summary Installed: {self.summary['installed']}")
                log("info", f"Already/Skipped: {self.summary['already']}")
                log("info", f"Failed: {self.summary['failed']}")
                log("info", f"Total time: {self.metrics['elapsed_s']:.1f}s")
        else:
            log("info", f"Summary Installed: {self.summary['installed']}")
            log("info", f"Already/Skipped: {self.summary['already']}")
            log("info", f"Failed: {self.summary['failed']}")
            log("info", f"Total time: {self.metrics['elapsed_s']:.1f}s")

    def cleanup(self):
        try:
            self.logfile.close()
        except Exception:
            pass
        try:
            shutil.rmtree(self.tempdir)
        except Exception:
            pass
        except Exception:
            pass
