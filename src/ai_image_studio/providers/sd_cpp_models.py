"""
Local Model Scanner - Discovers and catalogs local Stable Diffusion models.

This module scans configured folders for compatible model files (.gguf, .safetensors)
and generates ModelCard entries for integration with the provider system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai_image_studio.providers.base import ModelCard, GenerationMode

if TYPE_CHECKING:
    pass


# ============================================================================
# Model Info
# ============================================================================

@dataclass
class LocalModelInfo:
    """Metadata for a discovered local model."""
    
    path: Path
    filename: str
    format: str  # "gguf", "safetensors"
    size_bytes: int
    architecture: str | None = None  # "sd15", "sdxl", "sd3", "flux"
    quantization: str | None = None  # "q4_0", "q8_0", "f16", etc.
    
    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size_bytes / (1024 ** 3)
    
    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        name = self.path.stem
        # Clean up common suffixes
        name = re.sub(r'[-_]?(ggml|gguf|model|safetensors)[-_]?', '', name, flags=re.IGNORECASE)
        name = name.strip('-_')
        return name or self.filename


# ============================================================================
# Architecture Detection
# ============================================================================

# Patterns to detect model architecture from filename
ARCHITECTURE_PATTERNS = [
    (r'flux', 'flux'),
    (r'sd[-_]?3\.?5', 'sd35'),
    (r'sd[-_]?3', 'sd3'),
    (r'sdxl|stable[-_]?diffusion[-_]?xl', 'sdxl'),
    (r'sd[-_]?2\.?1', 'sd21'),
    (r'sd[-_]?2', 'sd2'),
    (r'sd[-_]?v?1[-_.]?5|stable[-_]?diffusion[-_]?1', 'sd15'),
    (r'sd[-_]?turbo', 'sd-turbo'),
]

# Quantization patterns
QUANTIZATION_PATTERNS = [
    (r'[-_]f32\b', 'f32'),
    (r'[-_]f16\b', 'f16'),
    (r'[-_]bf16\b', 'bf16'),
    (r'[-_]q8[-_]?0\b', 'q8_0'),
    (r'[-_]q8[-_]?1\b', 'q8_1'),
    (r'[-_]q6[-_]?k\b', 'q6_k'),
    (r'[-_]q5[-_]?k\b', 'q5_k'),
    (r'[-_]q5[-_]?1\b', 'q5_1'),
    (r'[-_]q5[-_]?0\b', 'q5_0'),
    (r'[-_]q4[-_]?k\b', 'q4_k'),
    (r'[-_]q4[-_]?1\b', 'q4_1'),
    (r'[-_]q4[-_]?0\b', 'q4_0'),
    (r'[-_]q3[-_]?k\b', 'q3_k'),
    (r'[-_]q2[-_]?k\b', 'q2_k'),
]


def detect_architecture(filename: str) -> str | None:
    """Detect model architecture from filename."""
    name_lower = filename.lower()
    for pattern, arch in ARCHITECTURE_PATTERNS:
        if re.search(pattern, name_lower):
            return arch
    return None


def detect_quantization(filename: str) -> str | None:
    """Detect quantization type from filename."""
    name_lower = filename.lower()
    for pattern, quant in QUANTIZATION_PATTERNS:
        if re.search(pattern, name_lower):
            return quant
    return None


# ============================================================================
# Model Scanner
# ============================================================================

class LocalModelScanner:
    """
    Scans folders for compatible local model files.
    
    Supports:
    - .gguf files (GGML format, primary for sd.cpp)
    - .safetensors files (for future support)
    """
    
    SUPPORTED_EXTENSIONS = {'.gguf', '.safetensors'}
    
    def scan(self, folders: list[Path]) -> list[LocalModelInfo]:
        """
        Scan folders for model files.
        
        Args:
            folders: List of folder paths to scan
            
        Returns:
            List of discovered models with metadata
        """
        models: list[LocalModelInfo] = []
        seen_paths: set[Path] = set()
        
        for folder in folders:
            if not folder.exists() or not folder.is_dir():
                continue
            
            # Scan recursively for model files
            for ext in self.SUPPORTED_EXTENSIONS:
                for path in folder.rglob(f'*{ext}'):
                    # Avoid duplicates
                    resolved = path.resolve()
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    
                    # Skip hidden files and directories (within the scan folder)
                    # Use relative path to avoid filtering out system dirs like .local
                    try:
                        relative = path.relative_to(folder)
                        if any(part.startswith('.') for part in relative.parts):
                            continue
                    except ValueError:
                        pass  # Can't compute relative path, skip check
                    
                    try:
                        info = self._parse_model_file(path)
                        if info:
                            models.append(info)
                    except (OSError, IOError):
                        # Skip files we can't read
                        continue
        
        # Sort by name for consistent ordering
        models.sort(key=lambda m: m.filename.lower())
        return models
    
    def _parse_model_file(self, path: Path) -> LocalModelInfo | None:
        """Parse a model file and extract metadata."""
        if not path.is_file():
            return None
        
        stat = path.stat()
        
        # Skip very small files (likely not models)
        if stat.st_size < 1_000_000:  # < 1MB
            return None
        
        filename = path.name
        ext = path.suffix.lower()
        
        return LocalModelInfo(
            path=path,
            filename=filename,
            format=ext.lstrip('.'),
            size_bytes=stat.st_size,
            architecture=detect_architecture(filename),
            quantization=detect_quantization(filename),
        )
    
    def to_model_card(self, info: LocalModelInfo) -> ModelCard:
        """
        Convert a discovered model to a ModelCard.
        
        The card is configured with conservative defaults that work
        for most Stable Diffusion models.
        """
        # Build description
        parts = []
        if info.architecture:
            arch_names = {
                'sd15': 'SD 1.5',
                'sd2': 'SD 2.0',
                'sd21': 'SD 2.1',
                'sdxl': 'SDXL',
                'sd3': 'SD 3',
                'sd35': 'SD 3.5',
                'flux': 'FLUX',
                'sd-turbo': 'SD Turbo',
            }
            parts.append(arch_names.get(info.architecture, info.architecture.upper()))
        else:
            parts.append('Stable Diffusion')
        
        if info.quantization:
            parts.append(f'({info.quantization.upper()})')
        
        parts.append(f'- {info.size_gb:.1f} GB')
        
        description = ' '.join(parts)
        
        # Determine supported modes based on architecture
        modes = {GenerationMode.TEXT_TO_IMAGE}
        if info.architecture in ('sd15', 'sd2', 'sd21', 'sdxl'):
            modes.add(GenerationMode.IMAGE_TO_IMAGE)
        
        # Build parameter options
        params = {'seed', 'steps', 'cfg_scale', 'sampler', 'scheduler'}
        
        param_options = {
            'sampler': [
                'default', 'euler', 'euler_a', 'heun', 'dpm2', 'dpm++2s_a',
                'dpm++2m', 'dpm++2mv2', 'ipndm', 'ipndm_v', 'lcm', 'tcd',
            ],
            'scheduler': [
                'default', 'discrete', 'karras', 'exponential', 'ays',
                'gits', 'sgm_uniform', 'simple', 'smoothstep', 'lcm',
            ],
        }
        
        param_defaults = {
            'steps': 20,
            'cfg_scale': 7.0,
            'sampler': 'euler_a',
            'scheduler': 'karras',
        }
        
        # Adjust defaults for specific architectures
        if info.architecture == 'flux':
            param_defaults['cfg_scale'] = 1.0
            param_defaults['steps'] = 4
            param_defaults['sampler'] = 'euler'
        elif info.architecture == 'sd-turbo':
            param_defaults['cfg_scale'] = 1.0
            param_defaults['steps'] = 4
        
        return ModelCard(
            id=f'local/{info.filename}',
            provider='sd-cpp',
            name=info.display_name,
            description=description,
            modes=modes,
            params=params,
            param_options=param_options,
            param_defaults=param_defaults,
            pricing_tier='free',
            tags=['local', info.format] + ([info.architecture] if info.architecture else []),
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def scan_and_register(folders: list[Path], registry) -> int:
    """
    Scan folders and register discovered models.
    
    Args:
        folders: Folders to scan for models
        registry: ProviderRegistry instance
        
    Returns:
        Number of models discovered and registered
    """
    scanner = LocalModelScanner()
    models = scanner.scan(folders)
    
    for info in models:
        card = scanner.to_model_card(info)
        registry.register_model(card)
    
    return len(models)
