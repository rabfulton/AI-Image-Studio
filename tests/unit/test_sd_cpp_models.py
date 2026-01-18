"""
Unit tests for the local model scanner.

Tests model discovery and ModelCard generation without requiring
the actual stable-diffusion.cpp library.
"""

import tempfile
from pathlib import Path

import pytest

from ai_image_studio.providers.sd_cpp_models import (
    LocalModelInfo,
    LocalModelScanner,
    detect_architecture,
    detect_quantization,
)
from ai_image_studio.providers.base import GenerationMode


class TestArchitectureDetection:
    """Tests for architecture detection from filenames."""
    
    def test_detect_sd15(self):
        assert detect_architecture('sd-v1-5-ggml-model.gguf') == 'sd15'
        assert detect_architecture('stable-diffusion-1.5.safetensors') == 'sd15'
        assert detect_architecture('sd1.5-pruned.safetensors') == 'sd15'
    
    def test_detect_sdxl(self):
        assert detect_architecture('sdxl-base-1.0.gguf') == 'sdxl'
        assert detect_architecture('stable-diffusion-xl-base.safetensors') == 'sdxl'
    
    def test_detect_sd3(self):
        assert detect_architecture('sd3-medium.gguf') == 'sd3'
        assert detect_architecture('sd-3-large.safetensors') == 'sd3'
    
    def test_detect_sd35(self):
        assert detect_architecture('sd3.5-large.gguf') == 'sd35'
        assert detect_architecture('sd-3.5-medium.safetensors') == 'sd35'
    
    def test_detect_flux(self):
        assert detect_architecture('flux-dev.gguf') == 'flux'
        assert detect_architecture('FLUX-schnell-q4.gguf') == 'flux'
    
    def test_unknown_architecture(self):
        assert detect_architecture('my-custom-model.gguf') is None
        assert detect_architecture('random_weights.safetensors') is None


class TestQuantizationDetection:
    """Tests for quantization detection from filenames."""
    
    def test_detect_q4_0(self):
        assert detect_quantization('model-q4_0.gguf') == 'q4_0'
        assert detect_quantization('model-q4-0.gguf') == 'q4_0'
    
    def test_detect_q8_0(self):
        assert detect_quantization('sdxl-q8_0.gguf') == 'q8_0'
    
    def test_detect_f16(self):
        assert detect_quantization('model-f16.gguf') == 'f16'
    
    def test_detect_q4_k(self):
        assert detect_quantization('model-q4_k.gguf') == 'q4_k'
        assert detect_quantization('model-q4-k.gguf') == 'q4_k'
    
    def test_no_quantization(self):
        assert detect_quantization('sd-v1-5.gguf') is None
        assert detect_quantization('model.safetensors') is None


class TestLocalModelInfo:
    """Tests for LocalModelInfo dataclass."""
    
    def test_size_gb(self):
        info = LocalModelInfo(
            path=Path('/test/model.gguf'),
            filename='model.gguf',
            format='gguf',
            size_bytes=4 * 1024 * 1024 * 1024,  # 4 GB
        )
        assert abs(info.size_gb - 4.0) < 0.01
    
    def test_display_name_clean(self):
        info = LocalModelInfo(
            path=Path('/test/sd-v1-5-ggml-model-q4_0.gguf'),
            filename='sd-v1-5-ggml-model-q4_0.gguf',
            format='gguf',
            size_bytes=1_000_000_000,
        )
        # Should remove ggml/model suffixes
        assert 'ggml' not in info.display_name.lower()
        assert 'model' not in info.display_name.lower()


class TestLocalModelScanner:
    """Tests for the model scanner."""
    
    def test_scan_empty_folder(self, tmp_path):
        """Empty folders return empty list."""
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        assert result == []
    
    def test_scan_nonexistent_folder(self):
        """Nonexistent folders are skipped."""
        scanner = LocalModelScanner()
        result = scanner.scan([Path('/nonexistent/folder')])
        assert result == []
    
    def test_scan_finds_gguf_files(self, tmp_path):
        """Discovers .gguf files."""
        # Create a fake model file (> 1MB)
        model_path = tmp_path / 'test-model.gguf'
        model_path.write_bytes(b'0' * 2_000_000)  # 2MB
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert len(result) == 1
        assert result[0].filename == 'test-model.gguf'
        assert result[0].format == 'gguf'
    
    def test_scan_finds_safetensors_files(self, tmp_path):
        """Discovers .safetensors files."""
        model_path = tmp_path / 'model.safetensors'
        model_path.write_bytes(b'0' * 2_000_000)
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert len(result) == 1
        assert result[0].format == 'safetensors'
    
    def test_scan_ignores_small_files(self, tmp_path):
        """Files under 1MB are skipped."""
        small_file = tmp_path / 'tiny.gguf'
        small_file.write_bytes(b'0' * 100)  # 100 bytes
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert result == []
    
    def test_scan_ignores_other_extensions(self, tmp_path):
        """Non-model files are ignored."""
        txt_file = tmp_path / 'readme.txt'
        txt_file.write_text('Hello')
        
        json_file = tmp_path / 'config.json'
        json_file.write_text('{}')
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert result == []
    
    def test_scan_ignores_hidden_files(self, tmp_path):
        """Hidden files and directories are skipped."""
        hidden_dir = tmp_path / '.hidden'
        hidden_dir.mkdir()
        hidden_model = hidden_dir / 'model.gguf'
        hidden_model.write_bytes(b'0' * 2_000_000)
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert result == []
    
    def test_scan_recursive(self, tmp_path):
        """Scans subdirectories."""
        subdir = tmp_path / 'models' / 'sdxl'
        subdir.mkdir(parents=True)
        model_path = subdir / 'sdxl-base.gguf'
        model_path.write_bytes(b'0' * 2_000_000)
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert len(result) == 1
        assert result[0].filename == 'sdxl-base.gguf'
    
    def test_scan_multiple_folders(self, tmp_path):
        """Scans multiple folders."""
        folder1 = tmp_path / 'folder1'
        folder1.mkdir()
        (folder1 / 'model1.gguf').write_bytes(b'0' * 2_000_000)
        
        folder2 = tmp_path / 'folder2'
        folder2.mkdir()
        (folder2 / 'model2.gguf').write_bytes(b'0' * 2_000_000)
        
        scanner = LocalModelScanner()
        result = scanner.scan([folder1, folder2])
        
        assert len(result) == 2
    
    def test_scan_deduplicates(self, tmp_path):
        """Same folder listed twice doesn't duplicate results."""
        (tmp_path / 'model.gguf').write_bytes(b'0' * 2_000_000)
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path, tmp_path])
        
        assert len(result) == 1
    
    def test_detects_architecture(self, tmp_path):
        """Architecture is detected from filename."""
        (tmp_path / 'sdxl-base-q8.gguf').write_bytes(b'0' * 2_000_000)
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert len(result) == 1
        assert result[0].architecture == 'sdxl'
    
    def test_detects_quantization(self, tmp_path):
        """Quantization is detected from filename."""
        (tmp_path / 'model-q4_0.gguf').write_bytes(b'0' * 2_000_000)
        
        scanner = LocalModelScanner()
        result = scanner.scan([tmp_path])
        
        assert len(result) == 1
        assert result[0].quantization == 'q4_0'


class TestModelCardGeneration:
    """Tests for converting LocalModelInfo to ModelCard."""
    
    def test_model_id_format(self, tmp_path):
        """Model IDs use local/ prefix."""
        info = LocalModelInfo(
            path=tmp_path / 'test-model.gguf',
            filename='test-model.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        assert card.id == 'local/test-model.gguf'
    
    def test_provider_is_sd_cpp(self, tmp_path):
        """Provider is set to sd-cpp."""
        info = LocalModelInfo(
            path=tmp_path / 'model.gguf',
            filename='model.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        assert card.provider == 'sd-cpp'
    
    def test_txt2img_always_supported(self, tmp_path):
        """Text-to-image is always supported."""
        info = LocalModelInfo(
            path=tmp_path / 'model.gguf',
            filename='model.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        assert GenerationMode.TEXT_TO_IMAGE in card.modes
    
    def test_img2img_for_sd_models(self, tmp_path):
        """Image-to-image is supported for SD 1.5/2/XL."""
        info = LocalModelInfo(
            path=tmp_path / 'sdxl.gguf',
            filename='sdxl.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
            architecture='sdxl',
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        assert GenerationMode.IMAGE_TO_IMAGE in card.modes
    
    def test_has_sampler_params(self, tmp_path):
        """Model cards include sampler/scheduler parameters."""
        info = LocalModelInfo(
            path=tmp_path / 'model.gguf',
            filename='model.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        assert 'sampler' in card.params
        assert 'scheduler' in card.params
        assert 'steps' in card.params
        assert 'cfg_scale' in card.params
        assert 'seed' in card.params
    
    def test_flux_adjusted_defaults(self, tmp_path):
        """FLUX models get adjusted CFG and steps defaults."""
        info = LocalModelInfo(
            path=tmp_path / 'flux.gguf',
            filename='flux.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
            architecture='flux',
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        # FLUX uses CFG 1.0 and fewer steps
        assert card.param_defaults['cfg_scale'] == 1.0
        assert card.param_defaults['steps'] == 4
    
    def test_pricing_tier_is_free(self, tmp_path):
        """Local models have 'free' pricing tier."""
        info = LocalModelInfo(
            path=tmp_path / 'model.gguf',
            filename='model.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        assert card.pricing_tier == 'free'
    
    def test_tags_include_format(self, tmp_path):
        """Tags include 'local' and the format."""
        info = LocalModelInfo(
            path=tmp_path / 'model.gguf',
            filename='model.gguf',
            format='gguf',
            size_bytes=4_000_000_000,
        )
        
        scanner = LocalModelScanner()
        card = scanner.to_model_card(info)
        
        assert 'local' in card.tags
        assert 'gguf' in card.tags
