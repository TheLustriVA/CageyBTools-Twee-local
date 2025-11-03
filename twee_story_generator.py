#!/usr/bin/env python3
"""
Interactive LLM-Guided Twee3/SugarCube Story Generator
A Python tool for creating branching interactive fiction with local LLM support

Requires:
- llama-cpp-python (for local LLM inference)
- A GGUF model file (e.g., from Hugging Face)

Usage:
    python twee_story_generator.py --model path/to/model.gguf
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    sys.exit(1)

@dataclass
class StoryPage:
    """Represents a single page/passage in the story"""
    name: str
    content: str = ""
    links: List[str] = None
    tags: List[str] = None
    completed: bool = False

    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.tags is None:
            self.tags = []

    def to_twee(self) -> str:
        """Convert to Twee3 format"""
        header = f":: {self.name}"
        if self.tags:
            header += f" {' '.join(self.tags)}"

        twee_content = self.content
        # Add link markup for SugarCube
        for link in self.links:
            if f"[[{link}]]" not in twee_content:
                twee_content += f"\n\n[[{link}]]"

        return f"{header}\n{twee_content}\n"

class TweeStoryGenerator:
    """Main class for interactive story generation"""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """Initialize with llama.cpp model"""
        print(f"Loading model: {model_path}")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        self.story_pages: Dict[str, StoryPage] = {}
        self.story_title = ""
        self.story_ifid = str(uuid.uuid4())

    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using the loaded LLM"""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n---", "\n\nUser:", "\n\nHuman:", "\n\n##"],
                echo=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

    def sanitize_page_name(self, name: str) -> str:
        """Convert text to valid Twee passage name"""
        # Remove special characters, keep alphanumeric and spaces
        sanitized = re.sub(r'[^a-zA-Z0-9\s-]', '', name)
        # Replace spaces with underscores, collapse multiple spaces
        sanitized = re.sub(r'\s+', '_', sanitized.strip())
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = "Page_" + sanitized
        return sanitized or "Unnamed_Page"

    def create_story_page(self, page_name: str, content_prompt: str, choice_prompts: List[str]) -> StoryPage:
        """Create a new story page using LLM"""

        # Create the prompt for the LLM
        llm_prompt = f"""You are an expert interactive fiction writer creating content for a Twee3/SugarCube story.

Task: Write engaging content for a passage named "{page_name}".

Content Description: {content_prompt}

Required Choices/Links: {', '.join(choice_prompts)}

Instructions:
- Write 2-4 paragraphs of engaging narrative content
- Use vivid, immersive language
- End with the choices naturally integrated into the story
- Format choices as: "Choice text" or **Choice text**
- Keep it under 300 words
- Do not use any SugarCube macros (<<>>)

Content:"""

        generated_content = self.generate_text(llm_prompt, max_tokens=400)

        if not generated_content:
            generated_content = f"You find yourself in {page_name.lower().replace('_', ' ')}. {content_prompt}"

        # Convert choice prompts to page links
        all_links = []
        for choice in choice_prompts:
            page_link = self.sanitize_page_name(choice)
            if page_link not in all_links:
                all_links.append(page_link)

        return StoryPage(
            name=page_name,
            content=generated_content,
            links=all_links,
            completed=True
        )

    def start_new_story(self) -> None:
        """Interactive story creation workflow"""
        print("\n=== Twee3/SugarCube Story Generator ===")
        print("Create an interactive branching story with LLM assistance\n")

        # Get story title
        self.story_title = input("Enter story title: ").strip()
        if not self.story_title:
            self.story_title = "Untitled Story"

        print(f"\nStarting story: {self.story_title}")
        print("IFID:", self.story_ifid)

        # Create the first page (Start)
        self.create_start_page()

        # Main creation loop
        while True:
            self.show_story_status()
            choice = self.get_user_choice()

            if choice == 'q':
                break
            elif choice == 'c':
                self.create_new_page()
            elif choice == 'e':
                self.export_story()
            elif choice == 'v':
                self.view_page()
            elif choice.isdigit():
                page_num = int(choice) - 1
                incomplete_pages = [name for name, page in self.story_pages.items() if not page.completed]
                if 0 <= page_num < len(incomplete_pages):
                    self.complete_page(incomplete_pages[page_num])

    def create_start_page(self):
        """Create the initial Start page"""
        print("\n--- Creating Start Page ---")
        content_prompt = input("Describe the opening scene: ").strip()

        choices = []
        print("\nEnter the initial choices (press Enter on empty line to finish):")
        while len(choices) < 4:
            choice = input(f"Choice {len(choices) + 1}: ").strip()
            if not choice:
                break
            choices.append(choice)

        if not choices:
            choices = ["Continue", "Look around", "Check inventory"]

        start_page = self.create_story_page("Start", content_prompt, choices)
        self.story_pages["Start"] = start_page

        # Create placeholder pages for the links
        for link in start_page.links:
            if link not in self.story_pages:
                self.story_pages[link] = StoryPage(name=link)

        print(f"\n✓ Created Start page with {len(start_page.links)} choice(s)")

    def complete_page(self, page_name: str):
        """Complete an incomplete page"""
        page = self.story_pages[page_name]
        print(f"\n--- Completing Page: {page_name} ---")

        content_prompt = input("Describe what happens on this page: ").strip()
        if not content_prompt:
            content_prompt = f"Continue the story from {page_name}"

        choices = []
        print("\nEnter the choices for this page (press Enter on empty line to finish):")
        while len(choices) < 4:
            choice = input(f"Choice {len(choices) + 1}: ").strip()
            if not choice:
                break
            choices.append(choice)

        # Generate the page content
        updated_page = self.create_story_page(page_name, content_prompt, choices)
        self.story_pages[page_name] = updated_page

        # Create placeholder pages for new links
        for link in updated_page.links:
            if link not in self.story_pages:
                self.story_pages[link] = StoryPage(name=link)

        print(f"\n✓ Completed page '{page_name}' with {len(updated_page.links)} choice(s)")

        # Show the generated content for review
        print("\n--- Generated Content ---")
        print(updated_page.content)
        print("\n--- Links ---")
        for link in updated_page.links:
            print(f"- {link}")

        # Ask for confirmation
        confirm = input("\nAccept this content? (y/n/e for edit): ").lower()
        if confirm == 'n':
            print("Page creation cancelled.")
            self.story_pages[page_name] = StoryPage(name=page_name)  # Reset to incomplete
        elif confirm == 'e':
            self.edit_page_content(page_name)

    def edit_page_content(self, page_name: str):
        """Allow manual editing of page content"""
        page = self.story_pages[page_name]
        print(f"\n--- Editing Page: {page_name} ---")
        print("Current content:")
        print(page.content)
        print("\nEnter new content (or press Enter to keep current):")

        new_content = input().strip()
        if new_content:
            page.content = new_content
            print("✓ Content updated")

    def show_story_status(self):
        """Display current story status"""
        print("\n" + "="*50)
        print(f"Story: {self.story_title}")
        print(f"Total pages: {len(self.story_pages)}")

        completed = [name for name, page in self.story_pages.items() if page.completed]
        incomplete = [name for name, page in self.story_pages.items() if not page.completed]

        print(f"Completed: {len(completed)}")
        print(f"Incomplete: {len(incomplete)}")

        if incomplete:
            print("\nIncomplete pages:")
            for i, name in enumerate(incomplete, 1):
                print(f"  {i}. {name}")

    def get_user_choice(self):
        """Get user's next action"""
        print("\nOptions:")
        incomplete = [name for name, page in self.story_pages.items() if not page.completed]

        if incomplete:
            print("Select a page number to complete, or:")

        print("  (c) Create new page")
        print("  (v) View existing page")
        print("  (e) Export story")
        print("  (q) Quit")

        return input("\nChoice: ").strip().lower()

    def create_new_page(self):
        """Create a completely new page"""
        print("\n--- Create New Page ---")
        page_name = input("Page name: ").strip()

        if not page_name:
            print("Invalid page name.")
            return

        page_name = self.sanitize_page_name(page_name)

        if page_name in self.story_pages:
            print(f"Page '{page_name}' already exists.")
            return

        content_prompt = input("Describe this page: ").strip()
        choices = []
        print("\nEnter choices (press Enter on empty line to finish):")
        while len(choices) < 4:
            choice = input(f"Choice {len(choices) + 1}: ").strip()
            if not choice:
                break
            choices.append(choice)

        new_page = self.create_story_page(page_name, content_prompt, choices)
        self.story_pages[page_name] = new_page

        # Create placeholder pages for links
        for link in new_page.links:
            if link not in self.story_pages:
                self.story_pages[link] = StoryPage(name=link)

        print(f"✓ Created page '{page_name}'")

    def view_page(self):
        """View an existing page"""
        print("\nExisting pages:")
        page_names = list(self.story_pages.keys())

        for i, name in enumerate(page_names, 1):
            status = "✓" if self.story_pages[name].completed else "○"
            print(f"  {i}. {status} {name}")

        try:
            choice = int(input("\nSelect page number: ")) - 1
            if 0 <= choice < len(page_names):
                page_name = page_names[choice]
                page = self.story_pages[page_name]

                print(f"\n--- Page: {page_name} ---")
                if page.completed:
                    print(page.content)
                    print("\nLinks:")
                    for link in page.links:
                        print(f"- {link}")
                else:
                    print("(Page not completed yet)")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

    def export_story(self):
        """Export the complete story to Twee3 format"""
        if not any(page.completed for page in self.story_pages.values()):
            print("No completed pages to export.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.story_title.replace(' ', '_')}_{timestamp}.twee"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write StoryData
                story_data = {
                    "ifid": self.story_ifid,
                    "format": "SugarCube",
                    "format-version": "2.37.3",
                    "start": "Start"
                }

                f.write(":: StoryData\n")
                f.write(json.dumps(story_data, indent=2))
                f.write("\n\n")

                # Write StoryTitle
                f.write(":: StoryTitle\n")
                f.write(self.story_title)
                f.write("\n\n")

                # Write all completed pages
                for page_name, page in self.story_pages.items():
                    if page.completed:
                        f.write(page.to_twee())
                        f.write("\n")

            print(f"\n✓ Story exported to: {filename}")
            print(f"Pages exported: {sum(1 for p in self.story_pages.values() if p.completed)}")

        except Exception as e:
            print(f"Error exporting story: {e}")

def main():
    parser = argparse.ArgumentParser(description="Interactive LLM-Guided Twee3/SugarCube Story Generator")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model file")
    parser.add_argument("--context", "-c", type=int, default=4096, help="Context size (default: 4096)")
    parser.add_argument("--gpu-layers", "-g", type=int, default=-1, help="GPU layers (-1 for all, default: -1)")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    try:
        generator = TweeStoryGenerator(args.model, args.context, args.gpu_layers)
        generator.start_new_story()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
