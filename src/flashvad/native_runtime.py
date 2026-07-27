from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DetectorConfig
from .detector import HysteresisDetector, VadEvent

_HOP_SAMPLES = 160
_SAMPLE_RATE = 16_000
_INSUFFICIENT_OUTPUT = ctypes.c_size_t(-1).value


class MacosNativeVadModel:
    """Process-level owner for an embedded Apple Accelerate FlashVAD library."""

    def __init__(
        self,
        library_path: str | Path,
        detector_config: DetectorConfig | None = None,
    ) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("the embedded Accelerate runtime requires macOS")
        self.path = Path(library_path)
        self.detector_config = detector_config or DetectorConfig()
        self.detector_config.validate()
        self.library = ctypes.CDLL(str(self.path))
        self.library.flashvad_state_size.argtypes = []
        self.library.flashvad_state_size.restype = ctypes.c_size_t
        self.library.flashvad_init.argtypes = [ctypes.c_void_p]
        self.library.flashvad_init.restype = ctypes.c_int
        self.library.flashvad_reset.argtypes = [ctypes.c_void_p]
        self.library.flashvad_destroy.argtypes = [ctypes.c_void_p]
        self.library.flashvad_push.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.library.flashvad_push.restype = ctypes.c_size_t
        self.state_size = int(self.library.flashvad_state_size())
        if self.state_size <= 0:
            raise RuntimeError("FlashVAD native library returned an invalid state size")

    def new_stream(
        self,
        detector_config: DetectorConfig | None = None,
    ) -> MacosNativeVadStream:
        return MacosNativeVadStream(
            self,
            detector_config or self.detector_config,
        )

    @classmethod
    def load_bundled(
        cls,
        detector_config: DetectorConfig | None = None,
    ) -> MacosNativeVadModel:
        """Build the packaged Accelerate runtime once, then reuse its owner."""

        if cls is MacosNativeVadModel and detector_config is None:
            return _cached_bundled_model()
        return cls(_bundled_native_library(), detector_config)


class MacosNativeVadStream:
    def __init__(
        self,
        owner: MacosNativeVadModel,
        detector_config: DetectorConfig,
    ) -> None:
        self.owner = owner
        self.detector = HysteresisDetector(detector_config, _HOP_SAMPLES / _SAMPLE_RATE)
        self._state = ctypes.create_string_buffer(owner.state_size)
        self._closed = False
        if owner.library.flashvad_init(self._state) != 0:
            raise RuntimeError("FlashVAD native state initialization failed")

    def reset(self) -> None:
        self._ensure_open()
        self.owner.library.flashvad_reset(self._state)
        self.detector.reset()

    def close(self) -> None:
        if not self._closed:
            self.owner.library.flashvad_destroy(self._state)
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("FlashVAD native stream is closed")

    def push(self, audio: object) -> tuple[np.ndarray, list[VadEvent]]:
        self._ensure_open()
        try:
            samples = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError("audio must be a one-dimensional numeric sequence") from exc
        if samples.ndim != 1:
            raise ValueError("audio must be one-dimensional")
        if samples.size and not np.isfinite(samples).all():
            raise ValueError("audio samples must be finite")
        samples = np.ascontiguousarray(samples)
        capacity = (samples.size + _HOP_SAMPLES - 1) // _HOP_SAMPLES + 1
        output = np.empty(capacity, dtype=np.float32)
        sample_pointer = (
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if samples.size
            else None
        )
        output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        emitted = int(
            self.owner.library.flashvad_push(
                self._state,
                sample_pointer,
                samples.size,
                output_pointer,
                capacity,
            )
        )
        if emitted == _INSUFFICIENT_OUTPUT:
            raise RuntimeError("FlashVAD native output capacity calculation failed")
        probabilities = output[:emitted].copy()
        events = [
            event
            for probability in probabilities
            for event in self.detector.update(float(probability))
        ]
        return probabilities, events

    def __enter__(self) -> MacosNativeVadStream:
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _bundled_native_source_dir() -> Path:
    packaged = Path(__file__).resolve().parent / "_native" / "macos"
    if packaged.is_dir():
        return packaged
    repository = Path(__file__).resolve().parents[2] / "native" / "macos"
    if repository.is_dir():
        return repository
    raise RuntimeError("FlashVAD macOS native sources are missing from this installation")


def _native_source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "flashvad_native.c",
        "flashvad_native.h",
        "flashvad_weights.c",
        "flashvad_weights.h",
    ):
        path = source / name
        if not path.is_file():
            raise RuntimeError(f"FlashVAD native source is incomplete: missing {name}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _bundled_native_library() -> Path:
    if platform.system() != "Darwin":
        raise RuntimeError("the embedded Accelerate runtime requires macOS")
    compiler = shutil.which("clang")
    if compiler is None:
        raise RuntimeError("clang is required to build FlashVAD's bundled runtime")

    source = _bundled_native_source_dir()
    compiler_identity = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    build_identity = json.dumps(
        {
            "architecture": platform.machine(),
            "compiler": str(Path(compiler).resolve()),
            "compiler_version": compiler_identity,
            "macos": platform.mac_ver()[0],
            "source": _native_source_digest(source),
        },
        sort_keys=True,
    )
    revision = hashlib.sha256(build_identity.encode()).hexdigest()[:16]
    configured_cache = os.environ.get("FLASHVAD_CACHE_DIR")
    cache_root = (
        Path(configured_cache).expanduser()
        if configured_cache
        else Path.home() / "Library" / "Caches" / "flashvad"
    )
    output_dir = cache_root / "native" / revision
    library = output_dir / "libflashvad_native.dylib"
    if library.is_file():
        return library

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".build.lock"
    with lock_path.open("a+b") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if library.is_file():
            return library

        descriptor, temporary_name = tempfile.mkstemp(
            prefix="libflashvad-native-",
            suffix=".dylib",
            dir=output_dir,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-O3",
                    "-mcpu=native",
                    "-DNDEBUG",
                    "-DACCELERATE_NEW_LAPACK",
                    f"-I{source}",
                    str(source / "flashvad_native.c"),
                    str(source / "flashvad_weights.c"),
                    "-framework",
                    "Accelerate",
                    "-dynamiclib",
                    "-o",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(temporary, library)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or "unknown compiler error"
            raise RuntimeError(f"FlashVAD native build failed: {message}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return library


@lru_cache(maxsize=1)
def _cached_bundled_model() -> MacosNativeVadModel:
    return MacosNativeVadModel(_bundled_native_library())


__all__ = ["MacosNativeVadModel", "MacosNativeVadStream"]
