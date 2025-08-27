import json
import os
import sys
import uuid

import logging
logger = logging.getLogger(__name__)


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def resolve_output_dir(path: str) -> str:
    """
    Ensure a safe output directory with user interaction.

    Options if `path` already exists:
    1. Abort execution
    2. Continue as-is (overwrite contents)
    3. Create a new unique directory (hash suffix)
    
    Returns the final path to use.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return path

    print(f"[WARNING] The directory '{path}' already exists!")
    print("Choose an option:")
    print("  [1] Abort")
    print("  [2] Continue and overwrite existing contents")
    print("  [3] Create a new unique directory")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice not in {"1", "2", "3"}:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        if choice == "1":
            print("Aborting training. Please choose a different output_dir.")
            sys.exit(1)
        elif choice == "2":
            print(f"Continuing. Existing contents of '{path}' may be overwritten.")
            return path
        elif choice == "3":
            unique_suffix = str(uuid.uuid4())[:8]
            base, name = os.path.split(path)
            new_path = os.path.join(base, f"{name}_{unique_suffix}")
            os.makedirs(new_path, exist_ok=True)
            print(f"Using new unique directory: '{new_path}'")
            return new_path