import dearpygui.dearpygui as dpg
import requests
import threading
import os
import io
import numpy as np
from PIL import Image
import queue
import math
import subprocess
import sys
import config

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

def update_status(text: str, status_widget="search_status_text"):
    """Helper to update a status bar text from any thread."""
    if dpg.does_item_exist(status_widget):
        dpg.set_value(status_widget, text)

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
                # Default to updating the main search status if no other handler
                update_status(error_message)

    thread = threading.Thread(target=thread_target)
    thread.daemon = True
    thread.start()

# --- Image & Texture Loading ---

def threaded_load_texture_from_disk(full_image_path: str, widget_tag: int, texture_registry_tag: int):
    """
    Loads a single thumbnail directly from disk in a background thread
    and applies it to an image widget. This is the most robust method for a desktop app.
    """
    if full_image_path in g["loaded_textures"]:
        if dpg.does_item_exist(widget_tag):
            dpg.set_value(widget_tag, g["loaded_textures"][full_image_path])
        return

    try:
        with Image.open(full_image_path) as img:
            img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            img = img.convert("RGBA")
            texture_data = np.frombuffer(img.tobytes(), dtype=np.uint8) / 255.0

        if not dpg.is_dearpygui_running(): return

        texture_id = dpg.add_static_texture(
            width=img.width, height=img.height, default_value=texture_data, parent=texture_registry_tag
        )
        
        if dpg.does_item_exist(widget_tag):
            dpg.set_value(widget_tag, texture_id)
        g["loaded_textures"][full_image_path] = texture_id

    except Exception as e:
        print(f"Error loading thumbnail from disk for {full_image_path}: {e}")

def open_file_location(sender, app_data, user_data):
    """Opens the directory containing the specified file."""
    path = os.path.dirname(user_data)
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin": # macOS
            subprocess.run(["open", path])
        else: # linux
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"Error opening file location {path}: {e}")

def display_gallery_images(sender, app_data, user_data):
    """
    Callback for the main thread. Clears and populates a gallery with image widgets
    in a grid layout, then loads textures asynchronously from disk.
    """
    gallery_tag = user_data["gallery_tag"]
    image_paths = user_data.get("image_paths", [])
    search_scores = user_data.get("search_scores")

    dpg.delete_item(gallery_tag, children_only=True)
    
    if not image_paths:
        dpg.add_text("No images found.", parent=gallery_tag)
        return
    
    if not g["dataset_root"]:
        dpg.add_text("Error: Dataset Root is not set.", parent=gallery_tag)
        return

    gallery_width = dpg.get_item_width(gallery_tag)
    if gallery_width <= 0: gallery_width = WINDOW_WIDTH - 300 # Fallback
    items_per_row = max(1, math.floor(gallery_width / GALLERY_ITEM_WIDTH))
    
    for i, rel_path in enumerate(image_paths):
        # Add same_line for all items except the first in each row
        if i % items_per_row != 0:
            dpg.add_same_line(parent=gallery_tag)

        full_path = os.path.join(g["dataset_root"], rel_path)
        
        with dpg.group(parent=gallery_tag) as item_group:
            img_widget_tag = dpg.add_image(g["loading_texture_id"], width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE)
            dpg.add_text(os.path.basename(rel_path), wrap=GALLERY_ITEM_WIDTH - 10)
            if search_scores:
                dpg.add_text(f"Score: {search_scores[i]:.4f}")
            
            with dpg.popup(item_group):
                dpg.add_menu_item(label="Open File Location", callback=open_file_location, user_data=full_path)

            threading.Thread(
                target=threaded_load_texture_from_disk, 
                args=(full_path, img_widget_tag, "texture_registry"),
                daemon=True
            ).start()

# --- UI Callbacks ---

def set_dataset_root(path: str):
    """Sets the dataset root and triggers fetching subdirectories."""
    if not path:
        update_status("Error: Tried to set an empty or invalid dataset path.")
        return

    update_status(f"Setting dataset root to: {path}...")
    config.add_recent_path(path) # Save to recents
    rebuild_recent_files_menu()

    def on_success(data):
        g["dataset_root"] = path
        dpg.set_value("dataset_root_text", f"Current Root: {path}")
        update_status("Dataset root set. Fetching subdirectories...")
        threaded_api_call(target=requests.get, on_success=on_get_directories_success, url=f"{API_URL}/directories")

    threaded_api_call(target=requests.post, on_success=on_success, url=f"{API_URL}/set-dataset-root", json={"path": path})

def callback_select_dataset_root(sender, app_data):
    dpg.add_file_dialog(directory_selector=True, show=True, callback=callback_dataset_root_selected, tag="dataset_root_dialog", width=700, height=400, modal=True)

def callback_dataset_root_selected(sender, app_data):
    if 'file_path_name' in app_data and app_data['file_path_name']:
        set_dataset_root(app_data['file_path_name'])

def on_get_directories_success(data):
    g["subdirectories"] = data.get("directories", [])
    if not g["subdirectories"]:
        update_status("No subdirectories with images found. You can index the root.")
        g["subdirectories"] = ['.']
    else:
        update_status("Subdirectories loaded. Please select one to index.")

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
    dpg.disable_item("select_root_menu_item")
    update_status(f"Building index for '{g['selected_subdir']}'. This may take a while...")

    def on_success(data):
        total = data.get('total', 'N/A')
        update_status(f"Index built successfully with {total} images. Loading gallery...")
        g["is_indexing"] = False
        g["index_loaded"] = True
        dpg.enable_item("build_index_button")
        dpg.enable_item("select_root_menu_item")
        dpg.enable_item("search_group")
        load_all_images_from_api()
        # After building index, also fetch clusters
        threaded_api_call(target=requests.get, on_success=on_get_clusters_success, url=f"{API_URL}/clusters")

    def on_error(error_message):
        update_status(f"Error building index: {error_message}")
        g["is_indexing"] = False
        dpg.enable_item("build_index_button")
        dpg.enable_item("select_root_menu_item")

    threaded_api_call(target=requests.post, on_success=on_success, on_error=on_error, url=f"{API_URL}/build-index", params={"img_dir": g["selected_subdir"]}, timeout=600)

def callback_subdir_selected(sender, app_data):
    g["selected_subdir"] = app_data

def callback_search(sender, app_data):
    if g["is_searching"] or not g["index_loaded"]: return
    g["is_searching"] = True
    update_status("Searching...")
    dpg.disable_item("search_group")

    query = dpg.get_value("search_input")
    top_k = dpg.get_value("top_k_input")

    def on_success(data):
        results = data.get("results", [])
        paths = [r["path"] for r in results]
        scores = [r["score"] for r in results]
        update_status(f"Found {len(results)} results for '{query}'.")
        g["ui_update_queue"].put((
            display_gallery_images,
            {"gallery_tag": "search_gallery", "image_paths": paths, "search_scores": scores}
        ))
        g["is_searching"] = False
        dpg.enable_item("search_group")

    def on_error(error_message):
        update_status(f"Search error: {error_message}")
        g["is_searching"] = False
        dpg.enable_item("search_group")

    threaded_api_call(target=requests.get, on_success=on_success, on_error=on_error, url=f"{API_URL}/search", params={"q": query, "top_k": top_k})

def load_all_images_from_api():
    update_status("Loading all indexed images...")
    def on_success(data):
        paths = data.get("images", [])
        update_status(f"Displaying {len(paths)} images.")
        g["ui_update_queue"].put((
            display_gallery_images,
            {"gallery_tag": "search_gallery", "image_paths": paths}
        ))
    threaded_api_call(target=requests.get, on_success=on_success, url=f"{API_URL}/all-images")

# --- Cluster Callbacks ---
def on_get_clusters_success(data):
    clusters = data.get("clusters", [])
    dpg.delete_item("cluster_sidebar", children_only=True) # Clear old clusters
    if not clusters:
        dpg.add_text("No clusters found.", parent="cluster_sidebar")
        return
    
    for cluster in clusters:
        with dpg.group(parent="cluster_sidebar"):
            label = f"Cluster {cluster['cluster_id']} ({cluster['count']} images)"
            with dpg.collapsing_header(label=label, default_open=True):
                dpg.add_button(label="View All", callback=callback_view_cluster, user_data=cluster['cluster_id'], width=-1)
                with dpg.group(horizontal=True):
                    for i, path in enumerate(cluster['preview_paths']):
                        # Small preview images
                        img_widget = dpg.add_image(g["loading_texture_id"], width=50, height=50)
                        full_path = os.path.join(g["dataset_root"], path)
                        threading.Thread(target=threaded_load_texture_from_disk, args=(full_path, img_widget, "texture_registry"), daemon=True).start()

def callback_view_cluster(sender, app_data, user_data):
    cluster_id = user_data
    update_status(f"Loading images for cluster {cluster_id}...", status_widget="cluster_status_text")
    
    def on_success(data):
        paths = data.get("image_paths", [])
        update_status(f"Displaying {len(paths)} images from cluster {cluster_id}.", status_widget="cluster_status_text")
        g["ui_update_queue"].put((
            display_gallery_images,
            {"gallery_tag": "cluster_gallery", "image_paths": paths}
        ))

    threaded_api_call(target=requests.get, on_success=on_success, url=f"{API_URL}/cluster/{cluster_id}")

# --- App Initialization ---

def on_initial_status_success(data):
    g["dataset_root"] = data.get("dataset_root")
    g["index_loaded"] = data.get("index_loaded", False)
    
    if g["dataset_root"]:
        set_dataset_root(g["dataset_root"]) # This will trigger subdirectory loading
    
    if g["index_loaded"]:
        dpg.enable_item("search_group")
        load_all_images_from_api()
        update_status(f"Loaded existing index with {data.get('indexed_image_count', 0)} images.")
        if data.get("has_clusters"):
            threaded_api_call(target=requests.get, on_success=on_get_clusters_success, url=f"{API_URL}/clusters")
    else:
        update_status("Welcome! Please select a dataset root to begin.")

def initialize_app_state():
    threaded_api_call(target=requests.get, on_success=on_initial_status_success, url=f"{API_URL}/status")

def rebuild_recent_files_menu():
    dpg.delete_item("recent_files_menu", children_only=True)
    recent_paths = config.get_recent_paths()
    if not recent_paths:
        dpg.add_text("No recent paths", parent="recent_files_menu")
        return
    for path in recent_paths:
        dpg.add_menu_item(label=path, parent="recent_files_menu", callback=lambda s, a, p=path: set_dataset_root(p))

# --- UI Setup ---

def setup_ui():
    with dpg.window(tag="primary_window"):
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Select Dataset Root...", callback=callback_select_dataset_root, tag="select_root_menu_item")
                with dpg.menu(label="Recent", tag="recent_files_menu"):
                    dpg.add_text("No recent paths")

        with dpg.tab_bar():
            with dpg.tab(label="Search / Browse"):
                with dpg.group():
                    dpg.add_text("Current Root: Not Set", tag="dataset_root_text")
                    with dpg.group(horizontal=True):
                        dpg.add_text("Index Target:")
                        dpg.add_combo(items=g["subdirectories"], tag="subdir_selector", callback=callback_subdir_selected, width=250, enabled=False)
                        dpg.add_button(label="Build Index", tag="build_index_button", callback=callback_build_index, enabled=False)
                dpg.add_separator()
                with dpg.group(tag="search_group", enabled=False):
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="search_input", hint="Enter search query...", width=-150, on_enter=True, callback=callback_search)
                        dpg.add_input_int(tag="top_k_input", label="Top K", width=100, default_value=10, min_value=1, max_value=100)
                        dpg.add_button(label="Search", callback=callback_search)
                dpg.add_separator()
                with dpg.child_window(tag="search_gallery"):
                    dpg.add_text("Build an index or perform a search to see images here.")
                dpg.add_separator()
                dpg.add_text("Initializing...", tag="search_status_text")

            with dpg.tab(label="Clusters"):
                with dpg.group(horizontal=True):
                    with dpg.child_window(tag="cluster_sidebar", width=250):
                        dpg.add_text("Clusters will appear here after indexing.")
                    with dpg.child_window(tag="cluster_gallery"):
                        dpg.add_text("Select a cluster to view images.")
                dpg.add_separator()
                dpg.add_text("Status", tag="cluster_status_text")

            with dpg.tab(label="Training"):
                dpg.add_text("Training and data augmentation UI will be implemented here.")

def launch_gui(api_url: str):
    global API_URL
    API_URL = api_url

    dpg.create_context()
    
    with dpg.texture_registry(tag="texture_registry"):
        loading_pixel = np.array([0.5, 0.5, 0.5, 1.0]) # RGBA
        loading_data = np.tile(loading_pixel, (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 1)).flatten()
        g["loading_texture_id"] = dpg.add_static_texture(width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE, default_value=loading_data)

    setup_ui()
    rebuild_recent_files_menu()

    dpg.create_viewport(title="CLIP Semantic Search", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    initialize_app_state()

    while dpg.is_dearpygui_running():
        try:
            while not g["ui_update_queue"].empty():
                callback, user_data = g["ui_update_queue"].get_nowait()
                callback(None, None, user_data)
        except queue.Empty:
            pass

        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    print("Running GUI in standalone mode. Make sure the FastAPI server is running.")
    launch_gui(api_url="http://127.0.0.1:8000")
