# AI Image Studio

A powerful, Linux-native AI image generation and editing application.

## Features

- **Node-based workflows** - Build flexible, reproducible image processing pipelines
- **Output Studio** - Compare, blend, and refine generated images
- **Universal model support** - Local models (Diffusers, SDXL) and hosted APIs (OpenAI, Stability AI)
- **600+ filters** - G'MIC integration for non-AI image processing
- **Batch processing** - Automate workflows with scripting and batch operations

## Requirements

- Python 3.11+
- Qt6 / PySide6
- CUDA-compatible GPU (recommended for local AI models)
- libgmic (optional, for G'MIC filters)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python -m ai_image_studio
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
ruff check src/
```

## Project Structure

```
ai_image_studio/
├── src/                    # Source code
│   ├── core/              # Core data structures and execution
│   ├── providers/         # AI model providers
│   ├── nodes/             # Node type implementations
│   ├── filters/           # G'MIC integration
│   ├── scripting/         # Scripting engine
│   ├── ui/                # User interface
│   ├── state/             # State management
│   └── utils/             # Utilities
├── resources/             # Icons, themes, templates
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Documentation

See the `docs/` directory for:
- [Design Document](design.md) - Vision and decisions
- [Architecture](architecture.md) - Technical structure
- [Features](features.md) - Detailed specifications
- [Wireframes](wireframes.md) - UI mockups

## License

TBD

## Status

🚧 **In Development** - Phase 0: Project Setup
