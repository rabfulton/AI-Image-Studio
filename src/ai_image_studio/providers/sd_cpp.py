"""
SDCpp Provider - Local Stable Diffusion inference via stable-diffusion.cpp.

This provider uses the stable-diffusion-cpp-python library bindings to
run local inference on CPU, CUDA, or Vulkan devices.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
import re
import multiprocessing as mp
from io import BytesIO
from typing import Any, TYPE_CHECKING

from ai_image_studio.providers.base import (
    ImageProvider,
    ProviderConfig,
    GenerationRequest,
    GenerationResult,
    GenerationError,
)

if TYPE_CHECKING:
    from ai_image_studio.core.data_types import ImageData

logger = logging.getLogger(__name__)


# Check if stable-diffusion-cpp-python is available
try:
    import stable_diffusion_cpp as sd_cpp
    HAS_SD_CPP = True
except ImportError:
    sd_cpp = None  # type: ignore
    HAS_SD_CPP = False


# ============================================================================
# Device Configuration
# ============================================================================

@dataclass
class SDCppDeviceConfig:
    """Device configuration for sd.cpp."""
    
    device: str = "auto"  # "auto", "cpu", "cuda", "vulkan"
    n_threads: int = -1  # -1 = auto-detect
    
    @property
    def use_cuda(self) -> bool:
        return self.device in ("auto", "cuda")
    
    @property
    def use_vulkan(self) -> bool:
        return self.device == "vulkan"
    
    @property
    def effective_threads(self) -> int:
        if self.n_threads > 0:
            return self.n_threads
        # Auto-detect: use physical cores
        if HAS_SD_CPP:
            return sd_cpp.sd_get_num_physical_cores()
        import os
        return os.cpu_count() or 4


# ============================================================================
# Sampler/Scheduler names (string-based for library)
# ============================================================================

# Sample methods from SAMPLE_METHOD_MAP
SAMPLE_METHODS = [
    'default', 'euler', 'euler_a', 'heun', 'dpm2', 'dpm++2s_a',
    'dpm++2m', 'dpm++2mv2', 'ipndm', 'ipndm_v', 'lcm', 'ddim_trailing', 'tcd',
]

# Schedulers from SCHEDULER_MAP
SCHEDULERS = [
    'default', 'discrete', 'karras', 'exponential', 'ays',
    'gits', 'sgm_uniform', 'simple', 'smoothstep', 'lcm',
]


# ============================================================================
# Provider Implementation
# ============================================================================

def _sd_cpp_worker(
    conn,
    *,
    model_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    sample_steps: int,
    cfg_scale: float,
    sample_method: str,
    scheduler: str,
    seed: int,
    preview_method: str,
    preview_interval: int,
    stream_previews: bool,
    keep_vae_on_cpu: bool,
    keep_clip_on_cpu: bool,
    vae_tiling: bool,
    vae_tile_overlap: float,
    vae_tile_size: Any,
    vae_relative_tile_size: Any,
    extra_constructor_kwargs: dict[str, Any] | None = None,
):
    """
    Run stable-diffusion-cpp-python in a separate process.

    This is primarily to guarantee cancellation: native code can block the main
    process and ignore SIGINT/Qt events. A subprocess can always be terminated.
    """
    try:
        import stable_diffusion_cpp as sd_cpp  # type: ignore

        def progress_callback(step: int, steps: int, elapsed: float):
            try:
                conn.send({"type": "progress", "step": int(step), "steps": int(steps), "elapsed": float(elapsed)})
            except Exception:
                pass

        def preview_callback(step: int, images: list[Any], is_noisy: bool):
            if not stream_previews or not images:
                return
            try:
                img0 = images[0]
                buf = BytesIO()
                img0.save(buf, format="PNG")
                conn.send(
                    {
                        "type": "preview",
                        "step": int(step),
                        "is_noisy": bool(is_noisy),
                        "png": buf.getvalue(),
                    }
                )
            except Exception:
                return

        constructor_kwargs = {
            "model_path": model_path,
            "verbose": False,
            "keep_vae_on_cpu": keep_vae_on_cpu,
            "keep_clip_on_cpu": keep_clip_on_cpu,
        }
        if extra_constructor_kwargs:
            constructor_kwargs.update(extra_constructor_kwargs)

        sd = sd_cpp.StableDiffusion(**constructor_kwargs)

        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "sample_steps": int(sample_steps),
            "cfg_scale": float(cfg_scale),
            "sample_method": sample_method,
            "scheduler": scheduler,
            "seed": int(seed),
            "batch_count": 1,
            "progress_callback": progress_callback,
            "vae_tiling": bool(vae_tiling),
            "vae_tile_overlap": float(vae_tile_overlap),
            "vae_tile_size": vae_tile_size,
            "vae_relative_tile_size": vae_relative_tile_size,
        }

        if stream_previews:
            gen_kwargs.update(
                {
                    "preview_method": preview_method,
                    "preview_interval": int(preview_interval),
                    "preview_callback": preview_callback,
                }
            )

        output = sd.generate_image(**gen_kwargs)
        if isinstance(output, list):
            output = output[0] if output else None
        if output is None:
            raise RuntimeError("Generation returned no output")

        buf = BytesIO()
        output.save(buf, format="PNG")
        conn.send({"type": "result", "png": buf.getvalue()})
    except Exception as e:
        try:
            conn.send({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class SDCppProvider(ImageProvider):
    """
    Local Stable Diffusion provider using stable-diffusion.cpp.
    
    This provider enables local inference without API dependencies,
    supporting CPU, CUDA, and Vulkan acceleration.
    """
    
    id = "sd-cpp"
    name = "Local (sd.cpp)"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._ctx = None  # StableDiffusion context
        self._loaded_model_path: str | None = None
        self._device_config = self._parse_device_config()
    
    def _parse_device_config(self) -> SDCppDeviceConfig:
        """Parse device config from provider config extra."""
        extra = self.config.extra
        return SDCppDeviceConfig(
            device=extra.get("device", "auto"),
            n_threads=extra.get("n_threads", -1),
        )

    def _get_runtime_flags(self) -> dict[str, Any]:
        """
        Runtime flags for stable-diffusion-cpp-python.

        These are *runtime* knobs that can reduce VRAM usage and avoid failures
        like "Device memory allocation failed" during VAE decode.
        """
        extra = self.config.extra
        # Keep VAE on CPU is recommended on <= 8GB GPUs and often helps Vulkan backends.
        keep_vae_on_cpu = bool(extra.get("keep_vae_on_cpu", True))
        keep_clip_on_cpu = bool(extra.get("keep_clip_on_cpu", False))
        return {
            "keep_vae_on_cpu": keep_vae_on_cpu,
            "keep_clip_on_cpu": keep_clip_on_cpu,
        }

    def _get_component_paths(self, model_path: Path) -> dict[str, str]:
        """
        Get optional component paths (VAE, CLIP, etc).

        stable-diffusion.cpp often requires separate CLIP and VAE files.
        If the user didn't configure them explicitly, try to auto-detect
        common filenames in the model's folder.
        """
        extra = self.config.extra
        component_keys = ["vae_path", "taesd_path", "clip_l_path", "clip_g_path", "t5xxl_path"]

        resolved: dict[str, str] = {}
        for key in component_keys:
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                p = Path(value).expanduser()
                if p.exists():
                    resolved[key] = str(p)

        # Auto-detect in model directory for missing components.
        model_dir = model_path.parent
        files = {p.name.lower(): p for p in model_dir.iterdir() if p.is_file()}

        def pick(candidates: list[str]) -> Path | None:
            for name in candidates:
                p = files.get(name.lower())
                if p:
                    return p
            # Try "contains" matches for common variants like "vae-ft-mse-840000-ema-pruned.safetensors"
            for candidate in candidates:
                needle = re.sub(r"\\.(gguf|safetensors)$", "", candidate, flags=re.IGNORECASE).lower()
                for fname, p in files.items():
                    if needle and needle in fname:
                        return p
            return None

        # Prefer GGUF when present, then safetensors.
        if "vae_path" not in resolved:
            p = pick(["vae.gguf", "vae.safetensors", "vae-ft-mse-840000-ema-pruned.safetensors"])
            if p:
                resolved["vae_path"] = str(p)

        if "clip_l_path" not in resolved:
            p = pick(["clip_l.gguf", "clip_l.safetensors", "clip-vit-large-patch14.safetensors"])
            if p:
                resolved["clip_l_path"] = str(p)

        if "clip_g_path" not in resolved:
            p = pick(["clip_g.gguf", "clip_g.safetensors"])
            if p:
                resolved["clip_g_path"] = str(p)

        if "t5xxl_path" not in resolved:
            p = pick(["t5xxl.gguf", "t5xxl.safetensors"])
            if p:
                resolved["t5xxl_path"] = str(p)

        if "taesd_path" not in resolved:
            p = pick(["taesd.gguf", "taesd.safetensors"])
            if p:
                resolved["taesd_path"] = str(p)

        return resolved

    @property
    def is_configured(self) -> bool:
        """Check if provider can generate images."""
        if not HAS_SD_CPP:
            return False
        # Check if any model folders are configured
        folders = self.config.extra.get("model_folders", [])
        return bool(folders)
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate images using stable-diffusion.cpp.
        
        Args:
            request: Generation parameters including model, prompt, dimensions
            
        Returns:
            GenerationResult with generated images
            
        Raises:
            GenerationError: If generation fails
        """
        if not HAS_SD_CPP:
            raise GenerationError("stable-diffusion-cpp-python is not installed")
        
        # Get model path from request
        model_id = request.model.id
        if not model_id.startswith("local/"):
            raise GenerationError(f"Invalid local model ID: {model_id}")
        
        model_filename = model_id[len("local/"):]
        model_path = self._find_model_path(model_filename)
        
        if not model_path:
            raise GenerationError(f"Model file not found: {model_filename}")
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._generate_sync,
            request,
            model_path,
        )
        
        return result
    
    def _find_model_path(self, filename: str) -> Path | None:
        """Find a model file in configured folders."""
        folders = self.config.extra.get("model_folders", [])
        
        for folder_str in folders:
            folder = Path(folder_str)
            if not folder.exists():
                continue
            
            # Check direct path
            path = folder / filename
            if path.exists():
                return path
            
            # Search recursively
            for found in folder.rglob(filename):
                return found
        
        return None
    
    def _generate_sync(
        self,
        request: GenerationRequest,
        model_path: Path,
    ) -> GenerationResult:
        """Synchronous generation (runs in thread pool)."""
        import time
        from ai_image_studio.core.data_types import ImageData
        from PIL import Image

        start_time = time.time()

        # Extract parameters from request
        params = request.extra_params or {}
        
        # Get generation parameters with defaults
        steps = params.get("steps", 20)
        cfg_scale = params.get("cfg_scale", 7.0)
        sample_method = params.get("sampler", "euler_a")
        scheduler = params.get("scheduler", "default")
        seed = request.seed if request.seed is not None else -1

        # Optional callbacks (used for intermediate previews/progress)
        progress_callback = params.get("_progress_callback")
        preview_callback = params.get("_preview_callback")
        cancel_check = params.get("_cancel_check")
        preview_method = params.get("preview_method", "proj")
        preview_interval = int(params.get("preview_interval", 2) or 2)

        # VAE tiling (reduces VRAM usage during decode; helpful on Vulkan backends)
        vae_tiling = bool(params.get("vae_tiling", False))
        vae_tile_overlap = float(params.get("vae_tile_overlap", 0.5))
        vae_tile_size = params.get("vae_tile_size", "0x0")
        vae_relative_tile_size = params.get("vae_relative_tile_size", "0x0")

        stream_previews = callable(preview_callback) and bool(params.get("stream_previews", True))

        # IMPORTANT: Run sd.cpp in a subprocess so cancellation always works,
        # even if native code blocks inside VAE decode or GPU allocation.
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        runtime_flags = self._get_runtime_flags()
        component_paths = self._get_component_paths(model_path)

        extra_constructor_kwargs: dict[str, Any] = {
            **component_paths,
            "n_threads": self._device_config.effective_threads,
            "wtype": "default",
        }

        proc = ctx.Process(
            target=_sd_cpp_worker,
            kwargs={
                "conn": child_conn,
                "model_path": str(model_path),
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or "",
                "width": request.width,
                "height": request.height,
                "sample_steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "sample_method": sample_method,
                "scheduler": scheduler,
                "seed": int(seed),
                "preview_method": preview_method,
                "preview_interval": int(preview_interval),
                "stream_previews": bool(stream_previews),
                "keep_vae_on_cpu": bool(runtime_flags.get("keep_vae_on_cpu", False)),
                "keep_clip_on_cpu": bool(runtime_flags.get("keep_clip_on_cpu", False)),
                "vae_tiling": bool(vae_tiling),
                "vae_tile_overlap": float(vae_tile_overlap),
                "vae_tile_size": vae_tile_size,
                "vae_relative_tile_size": vae_relative_tile_size,
                "extra_constructor_kwargs": extra_constructor_kwargs,
            },
            daemon=True,
        )
        proc.start()

        images: list[ImageData] = []
        try:
            while True:
                if callable(cancel_check) and cancel_check():
                    proc.terminate()
                    proc.join(timeout=5)
                    raise asyncio.CancelledError("Execution cancelled")

                if parent_conn.poll(0.1):
                    msg = parent_conn.recv()
                    msg_type = msg.get("type")
                    if msg_type == "progress" and callable(progress_callback):
                        progress_callback(msg.get("step", 0), msg.get("steps", steps), msg.get("elapsed", 0.0))
                    elif msg_type == "preview" and callable(preview_callback):
                        try:
                            img = Image.open(BytesIO(msg["png"]))
                            preview_callback(msg.get("step", 0), [img], bool(msg.get("is_noisy", False)))
                        except Exception:
                            pass
                    elif msg_type == "result":
                        img = Image.open(BytesIO(msg["png"]))
                        images = [self._convert_to_image_data(img)]
                        break
                    elif msg_type == "error":
                        raise GenerationError(f"Generation failed: {msg.get('error', 'unknown error')}")

                if not proc.is_alive():
                    # Child exited without sending a result.
                    raise GenerationError("Generation process exited unexpectedly")
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=1)
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
            seed=seed if seed >= 0 else None,
            generation_time=generation_time,
        )
    
    def _ensure_model_loaded(self, model_path: Path) -> None:
        """Load model if not already loaded."""
        path_str = str(model_path)
        
        if self._loaded_model_path == path_str and self._ctx is not None:
            return  # Already loaded
        
        # Unload previous model
        if self._ctx is not None:
            del self._ctx
            self._ctx = None
            self._loaded_model_path = None
        
        logger.info(f"Loading model: {model_path.name}")

        component_paths = self._get_component_paths(model_path)
        runtime_flags = self._get_runtime_flags()
        if component_paths:
            logger.info(
                "Using sd.cpp component paths: %s",
                {k: Path(v).name for k, v in component_paths.items()},
            )
        else:
            logger.warning(
                "No sd.cpp component paths configured/detected; if you get blank outputs, "
                "set VAE/CLIP paths in provider settings."
            )

        try:
            kwargs: dict[str, Any] = {
                "model_path": path_str,
                "n_threads": self._device_config.effective_threads,
                "wtype": "default",  # Auto-detect weight type
                "verbose": False,
            }
            
            kwargs.update(component_paths)
            kwargs.update(runtime_flags)

            try:
                self._ctx = sd_cpp.StableDiffusion(**kwargs)
            except TypeError as e:
                # Some builds/bindings use a smaller set of keyword args.
                # Retry with only the core args.
                logger.warning("sd.cpp constructor rejected optional args (%s); retrying core args", e)
                core_kwargs = {k: kwargs[k] for k in ("model_path", "n_threads", "wtype", "verbose")}
                self._ctx = sd_cpp.StableDiffusion(**core_kwargs)

            self._loaded_model_path = path_str
        except Exception as e:
            raise GenerationError(f"Failed to load model: {e}") from e
    
    def _convert_to_image_data(self, img) -> "ImageData":
        """Convert sd.cpp output to ImageData."""
        from ai_image_studio.core.data_types import ImageData
        from PIL import Image
        
        # generate_image returns PIL Images
        if isinstance(img, Image.Image):
            # Some sd.cpp bindings return RGBA images with an uninitialized or
            # zero alpha channel, which renders as a blank/grey square in Qt.
            # For generation outputs we don't need alpha, so force RGB.
            if img.mode != "RGB":
                img = img.convert("RGB")

            return ImageData.from_pil(img)
        else:
            raise GenerationError(f"Unexpected output type: {type(img)}")
    
    async def validate_credentials(self) -> bool:
        """
        Validate that the provider is functional.
        
        For local provider, this checks if the library is available
        and at least one model folder exists.
        """
        if not HAS_SD_CPP:
            return False
        
        folders = self.config.extra.get("model_folders", [])
        for folder_str in folders:
            folder = Path(folder_str)
            if folder.exists() and folder.is_dir():
                return True
        
        return False
    
    def list_models(self) -> list:
        """
        List available local models.
        
        Scans configured folders and returns discovered models.
        """
        from ai_image_studio.providers.sd_cpp_models import LocalModelScanner
        
        folders = self.config.extra.get("model_folders", [])
        if not folders:
            return []
        
        scanner = LocalModelScanner()
        models = scanner.scan([Path(f) for f in folders])
        
        return [scanner.to_model_card(info) for info in models]
    
    def unload_model(self) -> None:
        """Explicitly unload the current model to free memory."""
        if self._ctx is not None:
            del self._ctx
            self._ctx = None
            self._loaded_model_path = None
            logger.info("Model unloaded")
    
    def get_system_info(self) -> dict[str, Any]:
        """Get system information about sd.cpp capabilities."""
        if not HAS_SD_CPP:
            return {"available": False, "reason": "Library not installed"}
        
        return {
            "available": True,
            "version": getattr(sd_cpp, "sd_version", lambda: "unknown")(),
            "physical_cores": sd_cpp.sd_get_num_physical_cores(),
            "system_info": sd_cpp.sd_get_system_info(),
        }
