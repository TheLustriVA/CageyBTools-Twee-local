# LLM-Guided Twee3/SugarCube Story Generator

A battle-tested locally-hosted solution for creating branching interactive fiction using LLMs and the latest SugarCube syntax. This toolkit provides Python tools that work with your local llama.cpp setup to interactively generate Twee3 stories with minimal coding required, scaling up to full-fledged interactive fiction systems.

## 🚀 Quick Start

### 1. Setup
```bash
# Run the setup script to install dependencies and download a model
python setup_twee_tools.py
```

### 2. Basic Story Creation
```bash
# Start with the simple tool
python twee_story_generator.py --model models/your_model.gguf
```

### 3. Advanced Features
```bash
# Use the advanced tool for SugarCube macros, variables, and conditionals
python advanced_twee_generator.py --model models/your_model.gguf
```

## 📁 Files Included

### Core Tools
- **`twee_story_generator.py`** - Basic interactive story creator with simple branching
- **`advanced_twee_generator.py`** - Full-featured tool with SugarCube macro support
- **`setup_twee_tools.py`** - Automated setup and model download script

### Generated Documentation
- **`QUICKSTART.md`** - Detailed usage instructions and workflow
- **`sample_stories.md`** - Example story prompts and structures

## 🎯 Features

### Basic Tool (twee_story_generator.py)
- ✅ Interactive page-by-page story creation
- ✅ LLM-generated content from user prompts
- ✅ Automatic link extraction and page creation
- ✅ Simple branching narrative structure
- ✅ Twee3 format export
- ✅ Page completion tracking
- ✅ Content review and editing

### Advanced Tool (advanced_twee_generator.py)
- ✅ Full SugarCube 2.37+ macro support
- ✅ Variable tracking and management (`<<set>>`, conditionals)
- ✅ Story feature selection (variables, conditionals, inventory)
- ✅ Global variable system with StoryInit passage
- ✅ Advanced page editing interface
- ✅ Feature-aware content generation
- ✅ Professional Twee3 export with proper SugarCube syntax

## 🛠️ System Requirements

### Minimum
- **Hardware**: 8GB RAM, CPU with AVX2 support
- **Software**: Python 3.8+, ~3GB disk space
- **Model**: TinyLlama 1.1B (700MB) for testing

### Recommended
- **Hardware**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Software**: Python 3.9+, CUDA-capable llama-cpp-python build
- **Model**: Llama 3.2 3B or Qwen2.5 7B for quality results

### Your Current Setup (Perfect!)
- **Workstation**: AMD Ryzen 9 9900X3D, 192GB RAM, RTX 5090
- **Server**: Intel i7-11800H, 16GB RAM, RTX 3070 Mobile
- **Software**: Ubuntu 25.04, llama.cpp optimized build

Your RTX 5090 can easily handle the largest available models (70B+) for exceptional story quality.

## 📚 Workflow

### Basic Workflow
1. **Initialize**: Enter story title and opening scene description
2. **Generate Start Page**: LLM creates opening content with choices
3. **Iterative Creation**: Select incomplete pages and provide content prompts
4. **Review & Edit**: Accept, modify, or regenerate content
5. **Export**: Save as .twee file for Twine or Tweego compilation

### Advanced Workflow
1. **Feature Selection**: Choose story mechanics (variables, inventory, etc.)
2. **Global Variables**: Set up persistent story state tracking
3. **Advanced Page Creation**: Generate content with SugarCube macros
4. **Conditional Logic**: Add `<<if>>` statements and variable checks
5. **Professional Export**: Complete SugarCube story with StoryInit

## 🔧 SugarCube Integration

### Supported Macros
- **Variables**: `<<set $variable to value>>`
- **Conditionals**: `<<if condition>>content<</if>>`
- **Links**: `[[Display Text|PassageName]]`
- **Choices**: `<<choice "text" "target">>`
- **Display**: `<<display "PassageName">>`

### Variable Types
- **String**: `<<set $name to "Hero">>`
- **Number**: `<<set $health to 100>>`
- **Boolean**: `<<set $hasKey to true>>`
- **Arrays**: `<<set $inventory to ["sword", "potion"]>>`

### Story Structure
```twee
:: StoryData
{
  "ifid": "generated-uuid",
  "format": "SugarCube",
  "format-version": "2.37.3"
}

:: StoryInit
<<set $health to 100>>
<<set $gold to 50>>

:: Start
You begin your adventure...
[[Continue|FirstChoice]]
```

## 🚀 Getting Started on Your System

### Option 1: Automated Setup
```bash
# Clone or download the tools
# Run the setup script
python setup_twee_tools.py

# Follow the prompts to:
# - Install llama-cpp-python with CUDA support
# - Download a suitable model (recommended: Qwen2.5-7B)
# - Set up example files
```

### Option 2: Manual Setup
```bash
# Install dependencies with CUDA support for your RTX 5090
pip install llama-cpp-python[cuda]

# Download a model (example)
mkdir models
cd models
wget https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# Run the tools
cd ..
python twee_story_generator.py --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

### Optimization for Your Hardware
```bash
# For RTX 5090 - use maximum GPU layers
python twee_story_generator.py --model models/model.gguf --gpu-layers -1 --context 8192

# For the mobile RTX 3070 - conservative settings
python twee_story_generator.py --model models/model.gguf --gpu-layers 20 --context 4096
```

## 📈 Scaling from Simple to Complex

### Phase 1: Simple Stories
Start with `twee_story_generator.py` to create basic choose-your-own-adventure stories with just narrative text and choices.

### Phase 2: Interactive Elements
Move to `advanced_twee_generator.py` and add:
- Health/stats tracking with variables
- Inventory management
- Conditional story branches

### Phase 3: Full Game Systems
Extend with custom SugarCube macros for:
- Combat systems
- Character progression
- Complex inventory
- Save/load functionality
- Multimedia integration

### Phase 4: Professional Publishing
Export to:
- **Twine**: For visual editing and testing
- **Tweego**: For command-line compilation
- **Web**: Host on GitHub Pages or your server
- **Desktop**: Package with Electron
- **Mobile**: Cordova/PhoneGap wrapper

## 🔄 Export Formats

### Twee3 Format
Standard text-based format compatible with:
- Twine 2.x (import directly)
- Tweego compiler
- Custom build systems
- Version control (Git-friendly)

### HTML Output
Self-contained playable files with:
- Embedded story data
- Basic styling
- JavaScript navigation
- No server required

## 🎮 Example Story Types

### Fantasy RPG
- Character stats (strength, magic, health)
- Inventory system (weapons, potions, keys)
- Branching quests with consequences
- Multiple endings based on choices

### Mystery/Thriller
- Clue tracking system
- Character relationship variables
- Time-based events
- Evidence collection mechanics

### Slice of Life
- Relationship meters
- Daily choice consequences
- Character development arcs
- Multiple perspective narratives

## 🚨 Battle-Tested Features

### Robust Error Handling
- Model loading failure recovery
- Invalid input sanitization
- Memory management for long stories
- Graceful degradation on resource limits

### Content Quality Control
- Review/edit system for all generated content
- Manual override capabilities
- Content regeneration options
- Template-based fallbacks

### Professional Output
- Valid Twee3 syntax generation
- SugarCube compatibility testing
- Proper passage linking
- IFID generation for story identification

## 🔗 Integration with Existing Tools

### Twine Ecosystem
- Import generated .twee files directly
- Use Twine's visual editor for refinement
- Leverage existing Twine resources and tutorials
- Compatible with Twine community tools

### Command Line Workflow
- Integrate with Tweego for advanced builds
- Version control with Git
- Automated testing pipelines
- Batch processing capabilities

This toolkit provides everything you need to start creating interactive fiction immediately, then scale up to professional-quality games as your needs grow. The LLM integration reduces the technical barrier while maintaining full access to SugarCube's powerful features.