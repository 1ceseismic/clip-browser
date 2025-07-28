import json
import os
from typing import List, Optional
from pathlib import Path

CONFIG_FILE = "config.json"
MAX_RECENT_FILES = 5

INDEXES_DIR = Path(".indexes")
INDEX_REGISTRY_FILE = INDEXES_DIR / "indexes.json"


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
    """Adds a path to the list of recent paths, ensuring it's at the top."""
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

def get_last_used_path() -> Optional[str]:
    """Gets the most recent path from the config, which is the last one used."""
    paths = get_recent_paths()
    return paths[0] if paths else None

def load_index_registry() -> dict:
    """Loads the index registry from .indexes/indexes.json."""
    INDEXES_DIR.mkdir(exist_ok=True)
    if os.path.exists(INDEX_REGISTRY_FILE):
        with open(INDEX_REGISTRY_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_index_registry(registry: dict):
    """Saves the index registry to .indexes/indexes.json."""
    INDEXES_DIR.mkdir(exist_ok=True)
    with open(INDEX_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
