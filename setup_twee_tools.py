#!/usr/bin/env python3
"""
Setup Script for LLM-Guided Twee3/SugarCube Story Generator
This script helps you set up the environment and download a suitable model.

Usage:
    python setup_twee_tools.py
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required Python packages"""
    print("\n=== Installing Dependencies ===")

    packages = [
        "llama-cpp-python",
        "requests",
        "tqdm"
    ]

    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

    return True

def download_model():
    """Download a suitable model for story generation"""
    print("\n=== Model Download ===")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Suggest some good models for story generation
    suggested_models = [
        {
            "name": "Llama-3.2-3B-Instruct (Q4_K_M)",
            "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "size": "~2.0GB",
            "description": "Good balance of quality and speed, works well on most hardware"
        },
        {
            "name": "Qwen2.5-7B-Instruct (Q4_K_M)",
            "url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", 
            "size": "~4.4GB",
            "description": "Excellent for creative writing, requires 8GB+ RAM"
        },
        {
            "name": "TinyLlama-1.1B-Chat (Q4_K_M)",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size": "~700MB",
            "description": "Fast and lightweight, good for testing and low-end hardware"
        }
    ]

    print("Available models:")
    for i, model in enumerate(suggested_models, 1):
        print(f"  {i}. {model['name']} ({model['size']})")
        print(f"     {model['description']}")
    print(f"  {len(suggested_models) + 1}. Skip download (I'll provide my own model)")

    try:
        choice = int(input("\nSelect a model to download: ")) - 1

        if choice == len(suggested_models):
            print("Skipping model download. You can provide your own GGUF model file.")
            return True

        if 0 <= choice < len(suggested_models):
            model = suggested_models[choice]
            model_path = models_dir / model["filename"]

            if model_path.exists():
                print(f"Model {model['filename']} already exists.")
                return True

            print(f"\nDownloading {model['name']}...")
            print(f"Size: {model['size']}")
            print(f"URL: {model['url']}")

            try:
                urllib.request.urlretrieve(model["url"], model_path)
                print(f"✓ Model downloaded to: {model_path}")
                return True
            except Exception as e:
                print(f"✗ Download failed: {e}")
                print("You can download the model manually from the URL above.")
                return False
        else:
            print("Invalid selection.")
            return False

    except ValueError:
        print("Invalid input.")
        return False

def create_example_config():
    """Create example configuration files"""
    print("\n=== Creating Example Files ===")

    # Create a sample story prompt
    sample_story = '''# Sample Story Ideas

## Fantasy Adventure
Opening: "You wake up in a mysterious forest glade, surrounded by ancient standing stones that pulse with ethereal light."
Choices:
- Approach the glowing stones
- Search for a way out of the forest  
- Call out to see if anyone responds
- Examine your belongings

## Sci-Fi Mystery
Opening: "The space station's emergency lights cast everything in an ominous red glow as you float through the abandoned corridors."
Choices:
- Head to the command center
- Check the escape pods
- Investigate the strange sounds from deck 7
- Access the station's computer logs

## Modern Thriller
Opening: "Your phone buzzes with a text from an unknown number: 'They know what you did. Meet me at the old pier at midnight or everyone will know too.'"
Choices:
- Go to the pier
- Ignore the message
- Try to trace the number
- Call the police
'''

    with open("sample_stories.md", "w") as f:
        f.write(sample_story)
    print("✓ Created sample_stories.md")

    # Create a quick start guide
    quickstart = '''# LLM-Guided Twee3/SugarCube Story Generator Quick Start

## Basic Usage

1. **Simple Story Creation:**
   ```bash
   python twee_story_generator.py --model models/your_model.gguf
   ```

2. **Advanced Story with SugarCube Features:**
   ```bash  
   python advanced_twee_generator.py --model models/your_model.gguf
   ```

## Workflow

1. **Start**: Enter story title and opening scene
2. **Create**: Describe each page and add choices  
3. **Generate**: LLM creates content based on your prompts
4. **Review**: Accept, edit, or regenerate content
5. **Export**: Save as .twee file for use in Twine

## SugarCube Features (Advanced Tool)

- **Variables**: Track player stats, inventory, story state
- **Conditionals**: Show different content based on variables
- **Inventory**: Manage items and equipment
- **Multimedia**: Add images, sounds, styling

## Tips

- Keep content prompts specific but not too restrictive
- Use meaningful choice text that becomes page names
- Review generated content before accepting
- Export frequently to save your progress

## File Outputs

- `.twee` files: Import into Twine 2 or compile with Tweego
- `.html` files: Basic playable version (simple tool only)

## Troubleshooting

- **Out of memory**: Reduce context size with `--context 2048`
- **Slow generation**: Use fewer GPU layers `--gpu-layers 10`
- **Poor quality**: Try a larger/better model

For more advanced usage, see the documentation in each tool file.
'''

    with open("QUICKSTART.md", "w") as f:
        f.write(quickstart)
    print("✓ Created QUICKSTART.md")

def main():
    print("=== LLM-Guided Twee3/SugarCube Story Generator Setup ===")
    print("This will set up everything you need to create interactive fiction\n")

    check_python_version()

    # Install dependencies
    if not install_dependencies():
        print("\n✗ Dependency installation failed. Please install manually:")
        print("  pip install llama-cpp-python requests tqdm")
        return

    # Download model
    download_model()

    # Create example files
    create_example_config()

    print("\n" + "="*50)
    print("✓ Setup complete!")
    print("\nNext steps:")
    print("1. Run 'python twee_story_generator.py --model models/your_model.gguf'")
    print("2. Check QUICKSTART.md for detailed usage instructions")
    print("3. See sample_stories.md for story ideas")
    print("\nHappy story writing!")

if __name__ == "__main__":
    main()
