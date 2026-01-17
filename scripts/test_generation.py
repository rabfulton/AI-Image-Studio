#!/usr/bin/env python3
"""
Quick test script to verify API generation works.

Usage:
    # First configure your API key:
    python scripts/test_generation.py --configure
    
    # Then run a test generation:
    python scripts/test_generation.py --prompt "A cute cat"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def configure():
    """Interactive configuration of API keys."""
    from ai_image_studio.providers import get_registry, ProviderConfig
    
    registry = get_registry()
    registry.load_config()
    
    print("=== AI Image Studio - Provider Configuration ===\n")
    
    providers = [
        ("openai", "OpenAI (DALL-E)"),
        ("bfl", "Black Forest Labs (FLUX)"),
        ("openrouter", "OpenRouter"),
    ]
    
    for provider_id, name in providers:
        current = registry.get_config(provider_id)
        has_key = "✓" if current.api_key else "✗"
        print(f"{name}: [{has_key}]")
        
        answer = input(f"  Configure {name}? (y/n): ").strip().lower()
        if answer == "y":
            api_key = input(f"  Enter API key: ").strip()
            if api_key:
                registry.set_config(provider_id, ProviderConfig(api_key=api_key))
                print("  ✓ Saved")
    
    registry.save_config()
    print("\nConfiguration saved!")
    
    # Show available models
    print("\nAvailable models:")
    for model in registry.list_available_models()[:5]:
        print(f"  • {model.id}: {model.name}")


async def generate(prompt: str, model_id: str = "dall-e-3"):
    """Generate an image."""
    from ai_image_studio.providers import get_registry, GenerationRequest
    
    registry = get_registry()
    registry.load_config()
    
    # Get model
    model = registry.get_model(model_id)
    if not model:
        print(f"Error: Model '{model_id}' not found")
        print("Available models:")
        for m in registry.list_models():
            print(f"  • {m.id}")
        return
    
    # Get provider
    provider = registry.get_provider(model.provider)
    if not provider:
        print(f"Error: Provider '{model.provider}' not registered")
        return
    
    if not provider.is_configured:
        print(f"Error: Provider '{model.provider}' needs API key")
        print("Run: python scripts/test_generation.py --configure")
        return
    
    print(f"Generating with {model.name}...")
    print(f"Prompt: {prompt}")
    print()
    
    # Build request
    request = GenerationRequest(
        model=model,
        prompt=prompt,
        width=1024,
        height=1024,
    )
    
    try:
        result = await provider.generate(request)
        
        print("✓ Generation successful!")
        print(f"  Images: {len(result.images)}")
        
        if result.revised_prompt:
            print(f"  Revised prompt: {result.revised_prompt}")
        
        # Save first image
        if result.images:
            output_path = Path("output.png")
            pil_img = result.images[0].to_pil()
            pil_img.save(output_path)
            print(f"  Saved to: {output_path.absolute()}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test image generation")
    parser.add_argument("--configure", action="store_true", help="Configure API keys")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over mountains", help="Prompt for generation")
    parser.add_argument("--model", type=str, default="dall-e-3", help="Model ID to use")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.configure:
        configure()
    elif args.list_models:
        from ai_image_studio.providers import get_registry
        registry = get_registry()
        registry.load_config()
        
        print("Available models:")
        for m in registry.list_models():
            configured = m.provider in [p for p in registry.list_configured_providers()]
            status = "✓" if configured else "✗ (needs API key)"
            print(f"  {m.id}: {m.name} [{status}]")
    else:
        asyncio.run(generate(args.prompt, args.model))


if __name__ == "__main__":
    main()
