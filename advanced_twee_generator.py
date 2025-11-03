#!/usr/bin/env python3
"""
Advanced Interactive LLM-Guided Twee3/SugarCube Story Generator
Battle-tested tool for creating complex branching interactive fiction

Features:
- Local LLM integration with llama.cpp
- SugarCube macro support (variables, conditionals, etc.)
- Advanced story structure management
- Multiple export formats
- Variable tracking and story state management

Requires:
- llama-cpp-python (pip install llama-cpp-python)
- A GGUF model file

Usage:
    python advanced_twee_generator.py --model path/to/model.gguf
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid
from enum import Enum

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    sys.exit(1)

class StoryFeature(Enum):
    VARIABLES = "variables"
    CONDITIONALS = "conditionals"
    INVENTORY = "inventory"
    CHOICES = "choices"
    MULTIMEDIA = "multimedia"

@dataclass
class StoryVariable:
    """Represents a story variable"""
    name: str
    value: str
    var_type: str = "string"  # string, number, boolean, array
    description: str = ""

@dataclass
class StoryPage:
    """Enhanced story page with SugarCube features"""
    name: str
    content: str = ""
    links: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    variables: List[StoryVariable] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    completed: bool = False
    features: Set[StoryFeature] = field(default_factory=set)

    def to_twee(self) -> str:
        """Convert to Twee3 format with SugarCube syntax"""
        header = f":: {self.name}"
        if self.tags:
            header += f" {' '.join(self.tags)}"

        twee_content = self.content

        # Add variable setters if any
        if self.variables:
            for var in self.variables:
                if var.var_type == "number":
                    twee_content = f"<<set ${var.name} to {var.value}>>\n" + twee_content
                elif var.var_type == "boolean":
                    twee_content = f"<<set ${var.name} to {var.value.lower()}>>\n" + twee_content
                else:
                    twee_content = f'<<set ${var.name} to "{var.value}">>\n' + twee_content

        # Add conditional checks if any
        if self.conditions:
            for condition in self.conditions:
                twee_content = f"<<if {condition}>>\n{twee_content}\n<</if>>" 

        # Add links at the end
        if self.links:
            twee_content += "\n\n"
            for link in self.links:
                twee_content += f"[[{link.replace('_', ' ')}|{link}]]\n"

        return f"{header}\n{twee_content}\n"

class AdvancedTweeGenerator:
    """Advanced story generator with full SugarCube support"""

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
        self.global_variables: Dict[str, StoryVariable] = {}
        self.story_features: Set[StoryFeature] = set()

        # SugarCube macro templates
        self.macro_templates = {
            "variable": "<<set ${name} to {value}>>",
            "if": "<<if {condition}>>{content}<</if>>",
            "link": "[[{text}|{target}]]",
            "choice": "<<link '{text}' '{target}'>><</link>>",
            "display": "<<display '{passage}'>>",
            "include": "<<include '{passage}'>>",
        }

    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using the loaded LLM"""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n---", "\n\nUser:", "\n\nHuman:", "\n\n##", "\n\nAssistant:"],
                echo=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

    def sanitize_page_name(self, name: str) -> str:
        """Convert text to valid Twee passage name"""
        sanitized = re.sub(r'[^a-zA-Z0-9\s-]', '', name)
        sanitized = re.sub(r'\s+', '_', sanitized.strip())
        if sanitized and not sanitized[0].isalpha():
            sanitized = "Page_" + sanitized
        return sanitized or "Unnamed_Page"

    def create_story_page_advanced(self, page_name: str, content_prompt: str, 
                                  choice_prompts: List[str], features: Set[StoryFeature]) -> StoryPage:
        """Create a story page with advanced features"""

        # Build feature-specific prompting
        feature_instructions = []
        if StoryFeature.VARIABLES in features:
            feature_instructions.append("- Include variable assignments using SugarCube <<set>> syntax")
        if StoryFeature.CONDITIONALS in features:
            feature_instructions.append("- Add conditional text using <<if>> statements")
        if StoryFeature.INVENTORY in features:
            feature_instructions.append("- Track inventory items and player stats")

        llm_prompt = f"""You are an expert interactive fiction writer creating advanced Twee3/SugarCube content.

Task: Write content for passage "{page_name}"
Description: {content_prompt}
Required choices: {', '.join(choice_prompts)}
Features to include: {', '.join([f.value for f in features])}

Advanced Instructions:
{chr(10).join(feature_instructions)}
- Use SugarCube 2.37+ syntax
- Create immersive, engaging narrative
- Integrate mechanics naturally into the story
- Keep content under 400 words
- End with clear choice options

Example SugarCube syntax:
- Variables: <<set $health to 100>>
- Conditions: <<if $health > 50>>You feel strong.<</if>>
- Links: [[Continue|NextPage]]

Content:"""

        generated_content = self.generate_text(llm_prompt, max_tokens=500, temperature=0.8)

        if not generated_content:
            generated_content = f"You find yourself in {page_name.lower().replace('_', ' ')}. {content_prompt}"

        # Extract variables from generated content
        variables = self.extract_variables_from_content(generated_content)

        # Extract conditions
        conditions = self.extract_conditions_from_content(generated_content)

        # Convert choices to links
        all_links = [self.sanitize_page_name(choice) for choice in choice_prompts]

        return StoryPage(
            name=page_name,
            content=generated_content,
            links=all_links,
            variables=variables,
            conditions=conditions,
            completed=True,
            features=features
        )

    def extract_variables_from_content(self, content: str) -> List[StoryVariable]:
        """Extract variable definitions from generated content"""
        variables = []
        # Look for <<set $variable to value>> patterns
        var_pattern = r'<<set \$(\w+) to ([^>]+)>>'
        matches = re.findall(var_pattern, content)

        for var_name, var_value in matches:
            # Determine variable type
            var_type = "string"
            if var_value.isdigit() or (var_value.startswith('-') and var_value[1:].isdigit()):
                var_type = "number"
            elif var_value.lower() in ['true', 'false']:
                var_type = "boolean"

            variables.append(StoryVariable(
                name=var_name,
                value=var_value,
                var_type=var_type,
                description=f"Variable from {content[:50]}..."
            ))

        return variables

    def extract_conditions_from_content(self, content: str) -> List[str]:
        """Extract conditional statements from content"""
        conditions = []
        # Look for <<if condition>> patterns
        condition_pattern = r'<<if ([^>]+)>>'
        matches = re.findall(condition_pattern, content)
        return matches

    def start_advanced_story(self):
        """Advanced story creation workflow"""
        print("\n=== Advanced Twee3/SugarCube Story Generator ===")
        print("Create complex interactive fiction with full SugarCube support\n")

        # Get story details
        self.story_title = input("Enter story title: ").strip() or "Untitled Story"

        print("\nSelect story features (enter numbers separated by spaces):")
        features_list = list(StoryFeature)
        for i, feature in enumerate(features_list, 1):
            print(f"  {i}. {feature.value.title()} - Advanced {feature.value}")

        feature_input = input("\nFeatures to include: ").strip()
        selected_features = set()

        if feature_input:
            try:
                feature_indices = [int(x) - 1 for x in feature_input.split()]
                selected_features = {features_list[i] for i in feature_indices if 0 <= i < len(features_list)}
            except ValueError:
                print("Invalid input, using basic features")

        self.story_features = selected_features
        print(f"\nStory: {self.story_title}")
        print(f"Features: {', '.join([f.value for f in selected_features])}")
        print(f"IFID: {self.story_ifid}")

        # Create start page with selected features
        self.create_advanced_start_page()

        # Main loop
        while True:
            self.show_advanced_story_status()
            choice = self.get_advanced_user_choice()

            if choice == 'q':
                break
            elif choice == 'c':
                self.create_advanced_page()
            elif choice == 'e':
                self.export_advanced_story()
            elif choice == 'v':
                self.view_advanced_page()
            elif choice == 'g':
                self.manage_global_variables()
            elif choice.isdigit():
                page_num = int(choice) - 1
                incomplete_pages = [name for name, page in self.story_pages.items() if not page.completed]
                if 0 <= page_num < len(incomplete_pages):
                    self.complete_advanced_page(incomplete_pages[page_num])

    def create_advanced_start_page(self):
        """Create the start page with advanced features"""
        print("\n--- Creating Advanced Start Page ---")
        content_prompt = input("Describe the opening scene: ").strip()

        # Get initial choices
        choices = []
        print("\nEnter initial choices (Enter on empty line to finish):")
        while len(choices) < 4:
            choice = input(f"Choice {len(choices) + 1}: ").strip()
            if not choice:
                break
            choices.append(choice)

        if not choices:
            choices = ["Begin the adventure", "Check your surroundings", "Review your status"]

        # Add global variables if features are enabled
        if StoryFeature.VARIABLES in self.story_features:
            self.setup_global_variables()

        start_page = self.create_story_page_advanced("Start", content_prompt, choices, self.story_features)
        self.story_pages["Start"] = start_page

        # Create placeholder pages
        for link in start_page.links:
            if link not in self.story_pages:
                self.story_pages[link] = StoryPage(name=link)

        print(f"\n✓ Created advanced Start page with {len(start_page.links)} choice(s)")
        if start_page.variables:
            print(f"✓ Added {len(start_page.variables)} variables")

    def setup_global_variables(self):
        """Set up global story variables"""
        print("\n--- Global Variables Setup ---")
        print("Enter global variables (name:value:type, Enter on empty line to finish):")
        print("Example: health:100:number or name:Hero:string")

        while True:
            var_input = input("Variable: ").strip()
            if not var_input:
                break

            parts = var_input.split(':')
            if len(parts) >= 2:
                name = parts[0]
                value = parts[1]
                var_type = parts[2] if len(parts) > 2 else "string"

                self.global_variables[name] = StoryVariable(
                    name=name,
                    value=value,
                    var_type=var_type,
                    description="Global story variable"
                )
                print(f"✓ Added global variable: ${name} = {value} ({var_type})")

    def complete_advanced_page(self, page_name: str):
        """Complete a page with advanced features"""
        print(f"\n--- Completing Advanced Page: {page_name} ---")

        content_prompt = input("Describe what happens on this page: ").strip()
        if not content_prompt:
            content_prompt = f"Continue the story from {page_name}"

        # Select features for this page
        print("\nSelect features for this page (enter numbers, or Enter for story defaults):")
        features_list = list(StoryFeature)
        for i, feature in enumerate(features_list, 1):
            default = "✓" if feature in self.story_features else ""
            print(f"  {i}. {feature.value.title()} {default}")

        feature_input = input("Features: ").strip()
        page_features = self.story_features.copy()  # Start with story defaults

        if feature_input:
            try:
                feature_indices = [int(x) - 1 for x in feature_input.split()]
                page_features = {features_list[i] for i in feature_indices if 0 <= i < len(features_list)}
            except ValueError:
                print("Invalid input, using story defaults")

        # Get choices
        choices = []
        print("\nEnter choices for this page:")
        while len(choices) < 4:
            choice = input(f"Choice {len(choices) + 1}: ").strip()
            if not choice:
                break
            choices.append(choice)

        # Generate the page
        updated_page = self.create_story_page_advanced(page_name, content_prompt, choices, page_features)
        self.story_pages[page_name] = updated_page

        # Create placeholder pages for new links
        for link in updated_page.links:
            if link not in self.story_pages:
                self.story_pages[link] = StoryPage(name=link)

        print(f"\n✓ Completed advanced page '{page_name}'")
        self.show_page_details(updated_page)

        # Confirmation
        confirm = input("\nAccept this content? (y/n/e for edit): ").lower()
        if confirm == 'n':
            self.story_pages[page_name] = StoryPage(name=page_name)
        elif confirm == 'e':
            self.edit_advanced_page(page_name)

    def show_page_details(self, page: StoryPage):
        """Show detailed information about a page"""
        print("\n--- Generated Content ---")
        print(page.content)

        if page.variables:
            print("\n--- Variables ---")
            for var in page.variables:
                print(f"  ${var.name} = {var.value} ({var.var_type})")

        if page.conditions:
            print("\n--- Conditions ---")
            for condition in page.conditions:
                print(f"  <<if {condition}>>")

        print("\n--- Links ---")
        for link in page.links:
            print(f"  - {link}")

        if page.features:
            print("\n--- Features ---")
            for feature in page.features:
                print(f"  - {feature.value}")

    def edit_advanced_page(self, page_name: str):
        """Advanced page editing"""
        page = self.story_pages[page_name]
        print(f"\n--- Editing Advanced Page: {page_name} ---")

        while True:
            print("\nWhat would you like to edit?")
            print("  1. Content")
            print("  2. Variables")
            print("  3. Links")
            print("  4. Features")
            print("  5. Done")

            choice = input("Choice: ").strip()

            if choice == '1':
                print("Current content:")
                print(page.content)
                new_content = input("\nNew content (Enter to keep current): ").strip()
                if new_content:
                    page.content = new_content

            elif choice == '2':
                self.edit_page_variables(page)

            elif choice == '3':
                self.edit_page_links(page)

            elif choice == '4':
                self.edit_page_features(page)

            elif choice == '5':
                break

        print("✓ Page editing complete")

    def edit_page_variables(self, page: StoryPage):
        """Edit page variables"""
        print("\nCurrent variables:")
        for i, var in enumerate(page.variables, 1):
            print(f"  {i}. ${var.name} = {var.value} ({var.var_type})")

        print("\nAdd new variable (name:value:type) or Enter to skip:")
        var_input = input("Variable: ").strip()
        if var_input:
            parts = var_input.split(':')
            if len(parts) >= 2:
                name = parts[0]
                value = parts[1]
                var_type = parts[2] if len(parts) > 2 else "string"

                page.variables.append(StoryVariable(name, value, var_type))
                print(f"✓ Added variable: ${name}")

    def edit_page_links(self, page: StoryPage):
        """Edit page links"""
        print("\nCurrent links:")
        for i, link in enumerate(page.links, 1):
            print(f"  {i}. {link}")

        print("\nAdd new link or Enter to skip:")
        new_link = input("Link text: ").strip()
        if new_link:
            sanitized_link = self.sanitize_page_name(new_link)
            if sanitized_link not in page.links:
                page.links.append(sanitized_link)
                print(f"✓ Added link: {sanitized_link}")

    def edit_page_features(self, page: StoryPage):
        """Edit page features"""
        print("\nCurrent features:")
        for feature in page.features:
            print(f"  - {feature.value}")

        print("\nAvailable features:")
        features_list = list(StoryFeature)
        for i, feature in enumerate(features_list, 1):
            status = "✓" if feature in page.features else "○"
            print(f"  {i}. {status} {feature.value}")

        feature_input = input("\nToggle features (numbers): ").strip()
        if feature_input:
            try:
                indices = [int(x) - 1 for x in feature_input.split()]
                for i in indices:
                    if 0 <= i < len(features_list):
                        feature = features_list[i]
                        if feature in page.features:
                            page.features.remove(feature)
                            print(f"✗ Removed {feature.value}")
                        else:
                            page.features.add(feature)
                            print(f"✓ Added {feature.value}")
            except ValueError:
                print("Invalid input")

    def manage_global_variables(self):
        """Manage global story variables"""
        print("\n--- Global Variables Manager ---")

        if self.global_variables:
            print("Current global variables:")
            for name, var in self.global_variables.items():
                print(f"  ${name} = {var.value} ({var.var_type}) - {var.description}")
        else:
            print("No global variables defined.")

        while True:
            print("\nOptions:")
            print("  1. Add variable")
            print("  2. Edit variable")
            print("  3. Remove variable")
            print("  4. Done")

            choice = input("Choice: ").strip()

            if choice == '1':
                self.add_global_variable()
            elif choice == '2':
                self.edit_global_variable()
            elif choice == '3':
                self.remove_global_variable()
            elif choice == '4':
                break

    def add_global_variable(self):
        """Add a global variable"""
        name = input("Variable name: ").strip()
        if not name:
            return

        value = input("Initial value: ").strip()
        var_type = input("Type (string/number/boolean): ").strip() or "string"
        description = input("Description: ").strip()

        self.global_variables[name] = StoryVariable(name, value, var_type, description)
        print(f"✓ Added global variable: ${name}")

    def edit_global_variable(self):
        """Edit a global variable"""
        if not self.global_variables:
            print("No variables to edit.")
            return

        print("\nSelect variable to edit:")
        var_names = list(self.global_variables.keys())
        for i, name in enumerate(var_names, 1):
            print(f"  {i}. ${name}")

        try:
            choice = int(input("Variable: ")) - 1
            if 0 <= choice < len(var_names):
                var_name = var_names[choice]
                var = self.global_variables[var_name]

                new_value = input(f"New value (current: {var.value}): ").strip()
                if new_value:
                    var.value = new_value
                    print(f"✓ Updated ${var_name}")
        except ValueError:
            print("Invalid selection")

    def remove_global_variable(self):
        """Remove a global variable"""
        if not self.global_variables:
            print("No variables to remove.")
            return

        print("\nSelect variable to remove:")
        var_names = list(self.global_variables.keys())
        for i, name in enumerate(var_names, 1):
            print(f"  {i}. ${name}")

        try:
            choice = int(input("Variable: ")) - 1
            if 0 <= choice < len(var_names):
                var_name = var_names[choice]
                del self.global_variables[var_name]
                print(f"✗ Removed ${var_name}")
        except ValueError:
            print("Invalid selection")

    def show_advanced_story_status(self):
        """Show advanced story status"""
        print("\n" + "="*60)
        print(f"Story: {self.story_title}")
        print(f"Features: {', '.join([f.value for f in self.story_features])}")
        print(f"Global Variables: {len(self.global_variables)}")
        print(f"Total Pages: {len(self.story_pages)}")

        completed = [name for name, page in self.story_pages.items() if page.completed]
        incomplete = [name for name, page in self.story_pages.items() if not page.completed]

        print(f"Completed: {len(completed)}, Incomplete: {len(incomplete)}")

        if incomplete:
            print("\nIncomplete pages:")
            for i, name in enumerate(incomplete, 1):
                print(f"  {i}. {name}")

    def get_advanced_user_choice(self):
        """Get user choice for advanced interface"""
        print("\nOptions:")
        incomplete = [name for name, page in self.story_pages.items() if not page.completed]

        if incomplete:
            print("Select a page number to complete, or:")

        print("  (c) Create new page")
        print("  (v) View existing page")
        print("  (g) Manage global variables")
        print("  (e) Export story")
        print("  (q) Quit")

        return input("\nChoice: ").strip().lower()

    def create_advanced_page(self):
        """Create a new advanced page"""
        print("\n--- Create New Advanced Page ---")
        page_name = input("Page name: ").strip()

        if not page_name:
            print("Invalid page name.")
            return

        page_name = self.sanitize_page_name(page_name)

        if page_name in self.story_pages:
            print(f"Page '{page_name}' already exists.")
            return

        content_prompt = input("Describe this page: ").strip()

        # Select features
        print("\nSelect features for this page:")
        features_list = list(StoryFeature)
        for i, feature in enumerate(features_list, 1):
            default = "✓" if feature in self.story_features else ""
            print(f"  {i}. {feature.value.title()} {default}")

        feature_input = input("Features (numbers, or Enter for defaults): ").strip()
        page_features = self.story_features.copy()

        if feature_input:
            try:
                indices = [int(x) - 1 for x in feature_input.split()]
                page_features = {features_list[i] for i in indices if 0 <= i < len(features_list)}
            except ValueError:
                print("Invalid input, using defaults")

        # Get choices
        choices = []
        print("\nEnter choices:")
        while len(choices) < 4:
            choice = input(f"Choice {len(choices) + 1}: ").strip()
            if not choice:
                break
            choices.append(choice)

        new_page = self.create_story_page_advanced(page_name, content_prompt, choices, page_features)
        self.story_pages[page_name] = new_page

        # Create placeholder pages
        for link in new_page.links:
            if link not in self.story_pages:
                self.story_pages[link] = StoryPage(name=link)

        print(f"✓ Created advanced page '{page_name}'")

    def view_advanced_page(self):
        """View an advanced page with full details"""
        print("\nExisting pages:")
        page_names = list(self.story_pages.keys())

        for i, name in enumerate(page_names, 1):
            page = self.story_pages[name]
            status = "✓" if page.completed else "○"
            features = f"[{', '.join([f.value[:3] for f in page.features])}]" if page.features else ""
            print(f"  {i}. {status} {name} {features}")

        try:
            choice = int(input("\nSelect page number: ")) - 1
            if 0 <= choice < len(page_names):
                page_name = page_names[choice]
                page = self.story_pages[page_name]

                print(f"\n--- Advanced Page: {page_name} ---")
                if page.completed:
                    self.show_page_details(page)
                else:
                    print("(Page not completed yet)")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

    def export_advanced_story(self):
        """Export story with full SugarCube features"""
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

                # Write StoryInit with global variables
                if self.global_variables:
                    f.write(":: StoryInit\n")
                    for name, var in self.global_variables.items():
                        if var.var_type == "number":
                            f.write(f"<<set ${name} to {var.value}>>\n")
                        elif var.var_type == "boolean":
                            f.write(f"<<set ${name} to {var.value.lower()}>>\n")
                        else:
                            f.write(f'<<set ${name} to "{var.value}">>\n')
                    f.write("\n")

                # Write all completed pages
                for page_name, page in self.story_pages.items():
                    if page.completed:
                        f.write(page.to_twee())
                        f.write("\n")

            print(f"\n✓ Advanced story exported to: {filename}")
            print(f"✓ Pages exported: {sum(1 for p in self.story_pages.values() if p.completed)}")
            print(f"✓ Global variables: {len(self.global_variables)}")
            print(f"✓ Features used: {', '.join([f.value for f in self.story_features])}")

        except Exception as e:
            print(f"Error exporting story: {e}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Interactive LLM-Guided Twee3/SugarCube Story Generator")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model file")
    parser.add_argument("--context", "-c", type=int, default=4096, help="Context size (default: 4096)")
    parser.add_argument("--gpu-layers", "-g", type=int, default=-1, help="GPU layers (-1 for all, default: -1)")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    try:
        generator = AdvancedTweeGenerator(args.model, args.context, args.gpu_layers)
        generator.start_advanced_story()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
