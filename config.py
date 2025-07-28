import json
import os
from typing import List

CONFIG_FILE = "config.json"
MAX_RECENT_FILES = 5

def load_config() -> dict:
    """Loads the configuration from config.json."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_config(config: dict):
    """Saves the configuration to config.json."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def add_recent_path(path: str):
    """Adds a path to the list of recent paths, ensuring no duplicates."""
    config = load_config()
    recent_paths: List[str] = config.get("recent_paths", [])
    
    # If path is already in the list, remove it to re-add it at the top
    if path in recent_paths:
        recent_paths.remove(path)
    
    # Add the new path to the beginning of the list
    recent_paths.insert(0, path)
    
    # Keep the list at a maximum size
    config["recent_paths"] = recent_paths[:MAX_RECENT_FILES]
    
    save_config(config)

def get_recent_paths() -> List[str]:
    """Gets the list of recent paths from the config."""
    return load_config().get("recent_paths", [])
