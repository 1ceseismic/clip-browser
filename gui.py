import dearpygui.dearpygui as dpg
import requests
import threading
import os
import io
import numpy as np
from PIL import Image
import queue
import math

# --- Constants ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
THUMBNAIL_SIZE = 128
GALLERY_ITEM_WIDTH = 150 # Thumbnail width + padding

# --- Global State ---
API_URL = ""
# UI state
g = {
    "dataset_root": None,
    "subdirectories": [],
    "selected_subdir": None,
    "is_indexing": False,
    "is_searching": False,
    "index_loaded": False,
    "loaded_textures": {}, # Cache for loaded thumbnail textures {path: texture_id}
    "loading_texture_id": None,
    "ui_update_queue": queue.Queue(),
}

# --- GUI Update Helpers ---

def update_search_status(text: str):
    """Helper to update the search status bar text from any thread."""
    if dpg.does_item_exist("search_status_text"):
        dpg.set_value("search_status_text", text)

# --- API Communication (threaded) ---

def threaded_api_call(target, on_success=None, on_error=None, **kwargs):
    """Generic wrapper to run API calls in a background thread."""
    def thread_target():
        try:
            response = target(**kwargs)
            response.raise_for_status()
            if on_success:
                on_success(response.json())
        except requests.RequestException as e:
            error_message = f"API Error: {e}"
            if e.response is not None:
                try:
                    detail = e.response.json().get("detail", e.response.text)
                    error_message = f"API Error ({e.response.status_code}): {detail}"
                except requests.exceptions.JSONDecodeError:
                    pass
            print(error_message)
            if on_error:
                on_error(error_message)
            else:
                update_search_status(error_message)

    thread = threading.Thread(target=thread_target)
    thread.daemon = True
    thread.start()

# --- Image & Texture Loading ---

def threaded_load_texture_from_api(rel_image_path: str, widget_tag: int, texture_registry_tag: int):
    """
    Loads a single thumbnail from the API in a background thread
    and applies it to an image widget.
    """
    # Use rel_image_path as the key for caching textures in memory
    if rel_image_path in g["loaded_textures"]:
        if dpg.does_item_exist(widget_tag):
            dpg.set_value(widget_tag, g["loaded_textures"][rel_image_path])
        return

    try:
        url = f"{API_URL}/thumbnail/{rel_image_path}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with Image.open(io.BytesIO(response.content)) as img:
            # The API now returns a JPEG, which is RGB. Convert to RGBA for DPG.
            img = img.convert("RGBA")
            texture_data = np.frombuffer(img.tobytes(), dtype=np.uint8) / 255.0

        texture_id = dpg.add_static_texture(
            width=img.width,
            height=img.height,
            default_value=texture_data,
            parent=texture_registry_tag
        )
        
        if dpg.does_item_exist(widget_tag):
            dpg.set_value(widget_tag, texture_id)
        g["loaded_textures"][rel_image_path] = texture_id

    except Exception as e:
        print(f"Error loading thumbnail for {rel_image_path}: {e}")

def display_gallery_images(sender, app_data, user_data):
    """
    Callback for the main thread. Clears and populates the gallery with image widgets
    in a grid layout, then loads textures asynchronously from the API.
    """
    image_paths = user_data.get("image_paths", [])
    search_scores = user_data.get("search_scores")

    dpg.delete_item("results_gallery", children_only=True)
    
    if not image_paths:
        dpg.add_text("No images found.", parent="results_gallery")
        return
    
    # Calculate how many items can fit in a row
    gallery_width = dpg.get_item_width("results_gallery")
    items_per_row = max(1, math.floor(gallery_width / GALLERY_ITEM_WIDTH))
    
    count = 0
    for i, rel_path in enumerate(image_paths):
        # We no longer need the full path in the GUI
        with dpg.group(parent="results_gallery", width=GALLERY_ITEM_WIDTH):
            img_widget_tag = dpg.add_image(g["loading_texture_id"], width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE)
            dpg.add_text(os.path.basename(rel_path), wrap=GALLERY_ITEM_WIDTH - 10)
            if search_scores:
                dpg.add_text(f"Score: {search_scores[i]:.4f}")
            
            threading.Thread(
                target=threaded_load_texture_from_api, 
                args=(rel_path, img_widget_tag, "texture_registry"),
                daemon=True
            ).start()

        count += 1
        if count % items_per_row != 0:
            dpg.add_same_line(parent="results_gallery")

# --- UI Callbacks ---

def callback_select_dataset_root(sender, app_data):
    dpg.add_file_dialog(directory_selector=True, show=True, callback=callback_dataset_root_selected, tag="dataset_root_dialog", width=700, height=400, modal=True)

def callback_dataset_root_selected(sender, app_data):
    if 'file_path_name' in app_data and app_data['file_path_name']:
        path = app_data['file_path_name']
        update_search_status(f"Setting dataset root to: {path}...")

        def on_success(data):
            g["dataset_root"] = path
            dpg.set_value("dataset_root_text", f"Current Root: {path}")
            update_search_status("Dataset root set. Fetching subdirectories...")
            threaded_api_call(target=requests.get, on_success=on_get_directories_success, url=f"{API_URL}/directories")

        threaded_api_call(target=requests.post, on_success=on_success, url=f"{API_URL}/set-dataset-root", json={"path": path})

def on_get_directories_success(data):
    g["subdirectories"] = data.get("directories", [])
    if not g["subdirectories"]:
        update_search_status("No subdirectories with images found. You can index the root.")
        g["subdirectories"] = ['.']
    else:
        update_search_status("Subdirectories loaded. Please select one to index.")

    dpg.configure_item("subdir_selector", items=g["subdirectories"])
    if g["subdirectories"]:
        g["selected_subdir"] = g["subdirectories"][0]
        dpg.set_value("subdir_selector", g["selected_subdir"])
    
    dpg.enable_item("build_index_button")
    dpg.enable_item("subdir_selector")

def callback_build_index(sender, app_data):
    if not g["selected_subdir"] or g["is_indexing"]: return
    g["is_indexing"] = True
    dpg.disable_item("build_index_button")
    dpg.disable_item("select_root_button")
    update_search_status(f"Building index for '{g['selected_subdir']}'. This may take a while...")

    def on_success(data):
        total = data.get('total', 'N/A')
        update_search_status(f"Index built successfully with {total} images. Loading gallery...")
        g["is_indexing"] = False
        g["index_loaded"] = True
        dpg.enable_item("build_index_button")
        dpg.enable_item("select_root_button")
        dpg.enable_item("search_group")
        load_all_images_from_api()

    def on_error(error_message):
        update_search_status(f"Error building index: {error_message}")
        g["is_indexing"] = False
        dpg.enable_item("build_index_button")
        dpg.enable_item("select_root_button")

    threaded_api_call(target=requests.post, on_success=on_success, on_error=on_error, url=f"{API_URL}/build-index", params={"img_dir": g["selected_subdir"]}, timeout=600)

def callback_subdir_selected(sender, app_data):
    g["selected_subdir"] = app_data

def callback_search(sender, app_data):
    if g["is_searching"] or not g["index_loaded"]: return
    g["is_searching"] = True
    update_search_status("Searching...")
    dpg.disable_item("search_group")

    query = dpg.get_value("search_input")
    top_k = dpg.get_value("top_k_input")

    def on_success(data):
        results = data.get("results", [])
        paths = [r["path"] for r in results]
        scores = [r["score"] for r in results]
        update_search_status(f"Found {len(results)} results for '{query}'.")
        g["ui_update_queue"].put((
            display_gallery_images,
            {"image_paths": paths, "search_scores": scores}
        ))
        g["is_searching"] = False
        dpg.enable_item("search_group")

    def on_error(error_message):
        update_search_status(f"Search error: {error_message}")
        g["is_searching"] = False
        dpg.enable_item("search_group")

    threaded_api_call(target=requests.get, on_success=on_success, on_error=on_error, url=f"{API_URL}/search", params={"q": query, "top_k": top_k})

def load_all_images_from_api():
    """Fetches all image paths from the index and displays them."""
    update_search_status("Loading all indexed images...")
    def on_success(data):
        paths = data.get("images", [])
        update_search_status(f"Displaying {len(paths)} images.")
        g["ui_update_queue"].put((
            display_gallery_images,
            {"image_paths": paths}
        ))

    threaded_api_call(target=requests.get, on_success=on_success, url=f"{API_URL}/all-images")

# --- App Initialization ---

def on_initial_status_success(data):
    """Callback to configure UI based on initial server status."""
    g["dataset_root"] = data.get("dataset_root")
    g["index_loaded"] = data.get("index_loaded", False)
    
    if g["dataset_root"]:
        dpg.set_value("dataset_root_text", f"Current Root: {g['dataset_root']}")
        threaded_api_call(target=requests.get, on_success=on_get_directories_success, url=f"{API_URL}/directories")
    
    if g["index_loaded"]:
        dpg.enable_item("search_group")
        load_all_images_from_api()
        update_search_status(f"Loaded existing index with {data.get('indexed_image_count', 0)} images.")
    else:
        update_search_status("Welcome! Please select a dataset root to begin.")

def initialize_app_state():
    """Fetches initial status from the backend to set up the UI."""
    threaded_api_call(target=requests.get, on_success=on_initial_status_success, url=f"{API_URL}/status")

# --- UI Setup ---

def setup_ui():
    """Creates all the UI elements for the application."""
    with dpg.window(tag="primary_window"):
        with dpg.tab_bar():
            with dpg.tab(label="Search / Browse"):
                # --- Top Control Panel ---
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Select Dataset Root...", callback=callback_select_dataset_root, tag="select_root_button")
                        dpg.add_text("Current Root: Not Set", tag="dataset_root_text")

                    with dpg.group(horizontal=True):
                        dpg.add_text("Index Target:")
                        dpg.add_combo(items=g["subdirectories"], tag="subdir_selector", callback=callback_subdir_selected, width=250, enabled=False)
                        dpg.add_button(label="Build Index", tag="build_index_button", callback=callback_build_index, enabled=False)
                
                dpg.add_separator()

                # --- Search Panel ---
                with dpg.group(tag="search_group", enabled=False):
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="search_input", hint="Enter search query...", width=-150, on_enter=True, callback=callback_search)
                        dpg.add_input_int(tag="top_k_input", label="Top K", width=100, default_value=10, min_value=1, max_value=100)
                        dpg.add_button(label="Search", callback=callback_search)
                
                dpg.add_separator()

                # --- Results Gallery ---
                with dpg.child_window(tag="results_gallery"):
                    dpg.add_text("Build an index or perform a search to see images here.")
                
                dpg.add_separator()
                # --- Status Bar for this tab ---
                dpg.add_text("Initializing...", tag="search_status_text")


            # This is a placeholder for the training UI, as requested.
            with dpg.tab(label="Training"):
                dpg.add_text("Training and data augmentation UI will be implemented here.")
                dpg.add_text("This section will allow fine-tuning the CLIP model on your own datasets.")


def launch_gui(api_url: str):
    global API_URL
    API_URL = api_url

    dpg.create_context()
    
    # Create a texture registry
    with dpg.texture_registry(tag="texture_registry"):
        # Create a placeholder "loading" texture (a simple grey square)
        loading_pixel = np.array([0.5, 0.5, 0.5, 1.0]) # RGBA
        loading_data = np.tile(loading_pixel, (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 1)).flatten()
        g["loading_texture_id"] = dpg.add_static_texture(width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE, default_value=loading_data)

    setup_ui()

    dpg.create_viewport(title="CLIP Semantic Search", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    initialize_app_state()

    # --- Main Render Loop ---
    while dpg.is_dearpygui_running():
        # Check for and process UI updates from the queue
        try:
            while not g["ui_update_queue"].empty():
                callback, user_data = g["ui_update_queue"].get_nowait()
                # Call the function with dummy sender and app_data
                callback(None, None, user_data)
        except queue.Empty:
            pass # This is expected when the queue is empty

        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    print("Running GUI in standalone mode. Make sure the FastAPI server is running.")
    launch_gui(api_url="http://127.0.0.1:8000")
