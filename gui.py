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
import colorsys
import time
from scipy.spatial import KDTree

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
    "is_loading_model": False,
    "model_loaded": False,
    "index_loaded": False,
    "loading_texture_id": None,
    "ui_update_queue": queue.Queue(),
    "umap_series_tags": [],
    "score_plot_series_tag": None,
    "umap_kdtree": None,
    "umap_metadata": None,
    "last_umap_hover_idx": -1,
    "models": [],
    "selected_model": None,
    "pretrained_tags": [],
    "selected_pretrained": None,
}

# --- GUI Update Helpers ---

def update_status(text: str, status_widget="search_status_text"):
    """Helper to update a status bar text from any thread."""
    if dpg.does_item_exist(status_widget):
        g["ui_update_queue"].put((
            lambda s, a, u: dpg.set_value(u['widget'], u['text']),
            {'widget': status_widget, 'text': text}
        ))

# --- API Communication (threaded) ---

def threaded_api_call(target, on_success=None, on_error=None, **kwargs):
    """Generic wrapper to run API calls in a background thread."""
    kwargs.setdefault('timeout', 30)

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
                update_status(error_message)

    thread = threading.Thread(target=thread_target)
    thread.daemon = True
    thread.start()

# --- Image & Texture Loading ---

def _apply_texture(sender, app_data, user_data):
    """
    (Main thread) Creates a DPG texture from data and applies it to a widget, resizing the widget.
    """
    widget_tag = user_data['widget_tag']
    texture_data = user_data['texture_data']
    width = user_data['width']
    height = user_data['height']
    texture_registry_tag = user_data['texture_registry_tag']

    if not dpg.does_item_exist(widget_tag) or not dpg.is_dearpygui_running():
        return

    texture_id = dpg.add_static_texture(
        width=width, height=height, default_value=texture_data, parent=texture_registry_tag
    )
    
    # Configure the image widget to use the new texture and its correct aspect-ratio dimensions
    dpg.configure_item(widget_tag, texture_tag=texture_id, width=width, height=height)

def threaded_load_texture_from_disk(full_image_path: str, widget_tag: int, texture_registry_tag: int, max_size: tuple):
    """
    (Background thread) Loads image data, resizes it to fit within max_size, and queues a UI update.
    """
    try:
        with Image.open(full_image_path) as img:
            img.thumbnail(max_size) # This preserves aspect ratio
            thumb_width, thumb_height = img.size
            img = img.convert("RGBA")
            texture_data = (np.frombuffer(img.tobytes(), dtype=np.uint8) / 255.0).astype(np.float32)

        g["ui_update_queue"].put((
            _apply_texture,
            {
                'widget_tag': widget_tag,
                'texture_data': texture_data,
                'width': thumb_width,
                'height': thumb_height,
                'texture_registry_tag': texture_registry_tag
            }
        ))
    except Exception as e:
        if os.path.exists(full_image_path):
            print(f"Error processing thumbnail for {full_image_path}: {e}")

def open_file(sender, app_data, user_data):
    """Opens the specified file with the default application."""
    path = user_data
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin": # macOS
            subprocess.run(["open", path], check=True)
        else: # linux
            subprocess.run(["xdg-open", path], check=True)
    except Exception as e:
        print(f"Error opening file {path}: {e}")
        update_status(f"Error opening file: {e}")

def display_gallery_images(sender, app_data, user_data):
    """
    (Main thread) Clears and populates a gallery with image widgets.
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
    if gallery_width is None or gallery_width <= 0: gallery_width = WINDOW_WIDTH - 300
    items_per_row = max(1, math.floor(gallery_width / GALLERY_ITEM_WIDTH))
    
    for i in range(0, len(image_paths), items_per_row):
        with dpg.group(horizontal=True, parent=gallery_tag):
            for j in range(i, min(i + items_per_row, len(image_paths))):
                rel_path = image_paths[j]
                full_path = os.path.join(g["dataset_root"], rel_path)
                
                with dpg.group() as item_group:
                    # Add placeholder, will be resized later by _apply_texture
                    img_widget_tag = dpg.add_image(g["loading_texture_id"], width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE)
                    
                    threading.Thread(
                        target=threaded_load_texture_from_disk, 
                        args=(full_path, img_widget_tag, "texture_registry", (THUMBNAIL_SIZE, THUMBNAIL_SIZE)),
                        daemon=True
                    ).start()

                    dpg.add_text(os.path.basename(rel_path), wrap=GALLERY_ITEM_WIDTH - 10)
                    if search_scores and j < len(search_scores):
                        dpg.add_text(f"Score: {search_scores[j]:.4f}")
                    
                    with dpg.popup(item_group):
                        dpg.add_menu_item(label="Open File", callback=open_file, user_data=full_path)

# --- UI Callbacks ---

def _update_score_distribution_plot(sender, app_data, user_data):
    """(Main Thread) Updates the score distribution plot with new scores."""
    scores = user_data
    x_axis_tag = "score_plot_x_axis"
    y_axis_tag = "score_plot_y_axis"

    # Clear previous series if it exists
    if g.get("score_plot_series_tag") and dpg.does_item_exist(g["score_plot_series_tag"]):
        dpg.delete_item(g["score_plot_series_tag"])
        g["score_plot_series_tag"] = None

    if not scores:
        dpg.set_axis_limits(x_axis_tag, 0, 1)
        dpg.set_axis_limits(y_axis_tag, 0, 1) # Reset to default
        return

    x_data = list(range(len(scores)))
    
    # Add new series with wider bars to reduce space between them
    g["score_plot_series_tag"] = dpg.add_bar_series(x_data, scores, label="Scores", parent=y_axis_tag, weight=0.8)
    
    # Fit X axis to data
    dpg.set_axis_limits(x_axis_tag, -0.5, len(scores) - 0.5)

    # Dynamically fit Y axis to data
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    # Add padding to the y-axis to avoid bars touching the plot edges
    if score_range < 1e-6: # Handle case where all scores are the same
        padding = 0.1
    else:
        padding = score_range * 0.1
        
    y_min = min_score - padding
    y_max = max_score + padding
    
    dpg.set_axis_limits(y_axis_tag, y_min, y_max)

def _handle_status_update(sender, app_data, user_data):
    """Shared logic to update UI based on app status response."""
    data = user_data
    
    # Handle model loading state
    was_loading_model = g["is_loading_model"]
    g["model_loaded"] = data.get("model_loaded", False)
    if was_loading_model and g["model_loaded"]:
        g["is_loading_model"] = False
        dpg.enable_item("model_controls_group")
        update_status("Model loaded successfully.")

    model_name = data.get("model_name", "N/A")
    pretrained = data.get("pretrained_tag", "N/A")
    if dpg.does_item_exist("loaded_model_text"):
        dpg.set_value("loaded_model_text", f"Loaded Model: {model_name} ({pretrained})")

    g["dataset_root"] = data.get("dataset_root")
    g["index_loaded"] = data.get("index_loaded", False)

    if g["dataset_root"]:
        dpg.set_value("dataset_root_text", f"Current Root: {g['dataset_root']}")
    else:
        dpg.set_value("dataset_root_text", "Current Root: Not Set")
    
    if g["dataset_root"] and g["model_loaded"]:
        threaded_api_call(target=requests.get, on_success=on_get_directories_success, url=f"{API_URL}/directories")

    if g["index_loaded"]:
        dpg.enable_item("search_group")
        load_all_images_from_api()
        update_status(f"Loaded existing index with {data.get('indexed_image_count', 0)} images.")
        if data.get("has_clusters"):
            threaded_api_call(target=requests.get, on_success=on_get_clusters_success, url=f"{API_URL}/clusters")
            load_umap_data()
    else:
        dpg.disable_item("search_group")
        _clear_main_views(None, None, None) # Clear views first

        status_msg, placeholder_gallery, placeholder_sidebar, placeholder_umap = "", "", "", ""

        if g["is_loading_model"]:
            status_msg = "Loading new model, please wait..."
            placeholder_gallery = "A new model is loading. The UI will update when it's ready."
            placeholder_sidebar = "Waiting for new model to load."
            placeholder_umap = "Waiting for new model to load."
        elif not g["model_loaded"]:
            status_msg = "Loading model, please wait..."
            placeholder_gallery = "Model is loading, please wait. This can take a moment on first launch."
            placeholder_sidebar = "Waiting for model to load."
            placeholder_umap = "Waiting for model to load."
        elif not g["dataset_root"]:
            status_msg = "Model loaded. Please select a dataset root to begin."
            placeholder_gallery = "Please select a dataset root from the File menu."
            placeholder_sidebar = "Please select a dataset root."
            placeholder_umap = "Please select a dataset root."
        else:
            status_msg = "Model loaded. No index found for this root. Please build an index."
            placeholder_gallery = "No index found for this directory.\n\nPlease select a subdirectory and click 'Build Index'."
            placeholder_sidebar = "Please build an index first."
            placeholder_umap = "Please build an index to see UMAP data."
        
        update_status(status_msg)
        if dpg.does_item_exist("search_gallery"): dpg.add_text(placeholder_gallery, parent="search_gallery")
        if dpg.does_item_exist("cluster_sidebar"): dpg.add_text(placeholder_sidebar, parent="cluster_sidebar")
        if dpg.does_item_exist("cluster_gallery"): dpg.add_text("Select a cluster to view images.", parent="cluster_gallery")
        update_status(placeholder_umap, "umap_status_text")

def _clear_main_views(sender, app_data, user_data):
    """(Main Thread) Clears all data views in the UI to prepare for new data."""
    if dpg.does_item_exist("search_gallery"):
        dpg.delete_item("search_gallery", children_only=True)
    if dpg.does_item_exist("cluster_gallery"):
        dpg.delete_item("cluster_gallery", children_only=True)
    if dpg.does_item_exist("cluster_sidebar"):
        dpg.delete_item("cluster_sidebar", children_only=True)
    if g["umap_series_tags"]:
        for tag in g["umap_series_tags"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        g["umap_series_tags"].clear()
    if g.get("score_plot_series_tag") and dpg.does_item_exist(g["score_plot_series_tag"]):
        dpg.delete_item(g["score_plot_series_tag"])
        g["score_plot_series_tag"] = None

def set_dataset_root(path: str):
    """Sets the dataset root, clears the UI, and triggers a backend status update."""
    if not path:
        update_status("Error: Tried to set an empty or invalid dataset path.")
        return

    update_status(f"Setting dataset root to: {path}...")
    
    # Clear existing UI state before loading new data
    g["ui_update_queue"].put((_clear_main_views, None))

    config.add_recent_path(path)
    rebuild_recent_files_menu()

    def on_set_root_success(data):
        # The status poller will automatically pick up the change and update the UI
        pass

    threaded_api_call(target=requests.post, on_success=on_set_root_success, url=f"{API_URL}/set-dataset-root", json={"path": path})

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
    
    if g.get("model_loaded"):
        dpg.enable_item("build_index_button")
    dpg.enable_item("subdir_selector")

def callback_build_index(sender, app_data):
    if not g["selected_subdir"] or g["is_indexing"]: return
    g["is_indexing"] = True
    dpg.configure_item("indexing_progress_bar", show=True)
    dpg.disable_item("build_index_button")
    dpg.disable_item("select_root_menu_item")
    update_status(f"Building index for '{g['selected_subdir']}'. This may take a while...")

    def on_success(data):
        # The status poller will pick up the change.
        g["is_indexing"] = False
        dpg.configure_item("indexing_progress_bar", show=False)
        dpg.enable_item("build_index_button")
        dpg.enable_item("select_root_menu_item")

    def on_error(error_message):
        update_status(f"Error building index: {error_message}")
        g["is_indexing"] = False
        dpg.configure_item("indexing_progress_bar", show=False)
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
        
        g["ui_update_queue"].put((_update_score_distribution_plot, scores))
        
        g["ui_update_queue"].put((
            display_gallery_images,
            {"gallery_tag": "search_gallery", "image_paths": paths, "search_scores": scores}
        ))
        g["is_searching"] = False
        dpg.enable_item("search_group")
        g["ui_update_queue"].put((lambda s, a, u: dpg.focus_item(u), "search_input"))


    def on_error(error_message):
        update_status(f"Search error: {error_message}")
        g["is_searching"] = False
        dpg.enable_item("search_group")
        g["ui_update_queue"].put((lambda s, a, u: dpg.focus_item(u), "search_input"))

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

# --- Cluster & UMAP Callbacks ---
def on_get_clusters_success(data):
    clusters = data.get("clusters", [])
    g["ui_update_queue"].put((
        lambda s, a, u: dpg.delete_item("cluster_sidebar", children_only=True), None
    ))
    if not clusters:
        g["ui_update_queue"].put((
            lambda s, a, u: dpg.add_text("No clusters found.", parent="cluster_sidebar"), None
        ))
        return
    
    for cluster in clusters:
        g["ui_update_queue"].put((_create_cluster_card, cluster))

def _create_cluster_card(sender, app_data, cluster):
    with dpg.group(parent="cluster_sidebar"):
        label = f"Cluster {cluster['cluster_id']} ({cluster['count']} images)"
        with dpg.collapsing_header(label=label, default_open=True):
            dpg.add_button(label="View All", callback=callback_view_cluster, user_data=cluster['cluster_id'], width=-1)
            with dpg.group(horizontal=True):
                for path in cluster['preview_paths']:
                    img_widget = dpg.add_image(g["loading_texture_id"], width=50, height=50)
                    full_path = os.path.join(g["dataset_root"], path)
                    threading.Thread(target=threaded_load_texture_from_disk, args=(full_path, img_widget, "texture_registry", (50, 50)), daemon=True).start()

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

def _get_cluster_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        colors.append([int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)])
    return colors

def _update_umap_plot(metadata):
    if not dpg.does_item_exist("umap_plot"): return

    for tag in g["umap_series_tags"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    g["umap_series_tags"].clear()

    if not metadata:
        update_status("No UMAP data to display.", "umap_status_text")
        return

    clusters = {}
    for item in metadata:
        if 'umap' not in item or not item['umap']: continue
        cid = item.get('cluster_id', -1)
        if cid not in clusters:
            clusters[cid] = {'x': [], 'y': []}
        clusters[cid]['x'].append(item['umap'][0])
        clusters[cid]['y'].append(item['umap'][1])

    if not clusters:
        update_status("No valid UMAP coordinates in metadata.", "umap_status_text")
        return

    cluster_ids = sorted(clusters.keys())
    colors = _get_cluster_colors(len(cluster_ids))
    color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}

    for cid in cluster_ids:
        data = clusters[cid]
        series_tag = dpg.add_scatter_series(x=data['x'], y=data['y'], label=f"Cluster {cid}", parent="umap_plot_y_axis")
        g["umap_series_tags"].append(series_tag)
        
        theme = dpg.generate_uuid()
        with dpg.theme(tag=theme):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color_map[cid], category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(series_tag, theme)
    
    update_status(f"Displayed {len(metadata)} points across {len(clusters)} clusters.", "umap_status_text")

def _on_get_umap_data_success(data):
    metadata = data.get("metadata", [])
    
    # Store metadata and build KD-Tree for hover-preview
    g["umap_metadata"] = [item for item in metadata if 'umap' in item and item['umap']]
    if g["umap_metadata"]:
        umap_coords = np.array([item['umap'] for item in g["umap_metadata"]])
        g["umap_kdtree"] = KDTree(umap_coords)
    else:
        g["umap_kdtree"] = None
        g["umap_metadata"] = None

    g["ui_update_queue"].put((lambda s, a, u: _update_umap_plot(u), metadata))

def load_umap_data():
    update_status("Loading UMAP data...", "umap_status_text")
    threaded_api_call(
        target=requests.get,
        on_success=_on_get_umap_data_success,
        on_error=lambda e: update_status(f"Could not load UMAP data: {e}", "umap_status_text"),
        url=f"{API_URL}/all-metadata"
    )

# --- App Initialization ---

def status_poller():
    """
    Runs in a background thread, periodically checking the server status
    and triggering a UI update only when the state changes.
    """
    last_known_status = {"model_loaded": None, "index_loaded": None, "dataset_root": None}
    server_was_connected = False # Track connection state

    while dpg.is_dearpygui_running():
        try:
            response = requests.get(f"{API_URL}/status", timeout=5)
            response.raise_for_status()
            current_status = response.json()

            if not server_was_connected:
                server_was_connected = True
                print("Successfully connected to server.")

            # Check if the relevant state has changed
            if (current_status.get("model_loaded") != last_known_status.get("model_loaded") or
                current_status.get("index_loaded") != last_known_status.get("index_loaded") or
                current_status.get("dataset_root") != last_known_status.get("dataset_root")):
                
                print(f"Status changed: {current_status}. Triggering UI update.")
                last_known_status = current_status
                g["ui_update_queue"].put((_handle_status_update, current_status))

        except requests.RequestException:
            if server_was_connected:
                # We were connected, but now we're not.
                update_status("Connection to server lost. Reconnecting...")
                server_was_connected = False
                # Reset state to force a full UI refresh on reconnect
                g["model_loaded"] = False
                g["index_loaded"] = None 
                last_known_status = {"model_loaded": None, "index_loaded": None, "dataset_root": None}
                g["ui_update_queue"].put((_clear_main_views, None))
            else:
                # Still trying to connect for the first time.
                update_status("Connecting to server...")

        time.sleep(2) # Poll every 2 seconds

def initialize_app_state():
    """Starts the background thread that polls for server status and loads initial data."""
    def load_initial_data():
        def on_success(data):
            g["models"] = data.get("models", [])
            print(f"DEBUG: Received models from API: {g['models']}")
            dpg.configure_item("model_selector", items=g["models"])
        
        threaded_api_call(target=requests.get, on_success=on_success, url=f"{API_URL}/models")

    load_initial_data()
    poller_thread = threading.Thread(target=status_poller, daemon=True)
    poller_thread.start()

def callback_set_recent_root(sender, app_data, user_data):
    """Callback for recent file menu items to set the dataset root."""
    if user_data:
        set_dataset_root(user_data)

def rebuild_recent_files_menu():
    if not dpg.does_item_exist("recent_files_menu"): return
    dpg.delete_item("recent_files_menu", children_only=True)
    recent_paths = config.get_recent_paths()
    if not recent_paths:
        dpg.add_text("No recent paths", parent="recent_files_menu")
        return
    for path in recent_paths:
        dpg.add_menu_item(label=path, parent="recent_files_menu", callback=callback_set_recent_root, user_data=path)

def on_model_selected(sender, app_data):
    g["selected_model"] = app_data
    dpg.configure_item("pretrained_selector", items=[])
    dpg.set_value("pretrained_selector", "")
    dpg.disable_item("load_model_button")
    
    def on_success(data):
        g["pretrained_tags"] = data.get("tags", [])
        dpg.configure_item("pretrained_selector", items=g["pretrained_tags"])
        if g["pretrained_tags"]:
            g["selected_pretrained"] = g["pretrained_tags"][0]
            dpg.set_value("pretrained_selector", g["selected_pretrained"])
            dpg.enable_item("load_model_button")

    threaded_api_call(target=requests.get, on_success=on_success, url=f"{API_URL}/pretrained-for-model", params={"model_name": app_data})

def on_pretrained_selected(sender, app_data):
    g["selected_pretrained"] = app_data
    if app_data:
        dpg.enable_item("load_model_button")
    else:
        dpg.disable_item("load_model_button")

def callback_load_model(sender, app_data):
    if not g["selected_model"] or not g["selected_pretrained"] or g["is_loading_model"]:
        return
    
    g["is_loading_model"] = True
    dpg.disable_item("model_controls_group")
    update_status(f"Loading model: {g['selected_model']} ({g['selected_pretrained']})...")
    _clear_main_views(None, None, None)

    def on_success(data):
        # Status poller will detect the change and update the UI state
        pass
    
    def on_error(error_message):
        update_status(f"Failed to start model load: {error_message}")
        g["is_loading_model"] = False
        dpg.enable_item("model_controls_group")

    threaded_api_call(
        target=requests.post,
        on_success=on_success,
        on_error=on_error,
        url=f"{API_URL}/load-model",
        json={"model_name": g["selected_model"], "pretrained": g["selected_pretrained"]}
    )

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
                with dpg.group(tag="model_controls_group"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("CLIP Model:")
                        dpg.add_combo(items=g["models"], tag="model_selector", callback=on_model_selected, width=200)
                        dpg.add_combo(items=g["pretrained_tags"], tag="pretrained_selector", callback=on_pretrained_selected, width=200)
                        dpg.add_button(label="Load Model", tag="load_model_button", callback=callback_load_model, enabled=False)
                    dpg.add_text("Loaded Model: N/A", tag="loaded_model_text")
                dpg.add_separator()
                with dpg.group(tag="search_group", enabled=False):
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="search_input", hint="Enter search query...", width=-150, on_enter=True, callback=callback_search)
                        dpg.add_input_int(tag="top_k_input", label="Top K", width=100, default_value=10, min_value=1, max_value=100)
                        dpg.add_button(label="Search", callback=callback_search)
                dpg.add_separator()
                with dpg.plot(label="Score Distribution", height=120, width=-1, tag="score_plot"):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Result Rank", tag="score_plot_x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Similarity Score", tag="score_plot_y_axis")
                dpg.add_separator()
                with dpg.child_window(tag="search_gallery", width=-1):
                    dpg.add_text("Build an index or perform a search to see images here.")
                dpg.add_separator()
                dpg.add_progress_bar(tag="indexing_progress_bar", show=False, overlay="Indexing...", width=-1)
                dpg.add_text("Initializing...", tag="search_status_text")

            with dpg.tab(label="Clusters"):
                with dpg.group(horizontal=True):
                    with dpg.child_window(tag="cluster_sidebar", width=250):
                        dpg.add_text("Clusters will appear here after indexing.")
                    with dpg.child_window(tag="cluster_gallery", width=-1):
                        dpg.add_text("Select a cluster to view images.")
                dpg.add_separator()
                dpg.add_text("Status", tag="cluster_status_text")

            with dpg.tab(label="UMAP Visualization"):
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=-270):
                        with dpg.plot(label="UMAP 2D Projection", height=-1, width=-1, tag="umap_plot"):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="UMAP 1", tag="umap_plot_x_axis")
                            dpg.add_plot_axis(dpg.mvYAxis, label="UMAP 2", tag="umap_plot_y_axis")
                    with dpg.child_window(width=250):
                        dpg.add_text("Image Preview")
                        dpg.add_image(g["loading_texture_id"], tag="umap_preview_image", width=240, height=240)
                dpg.add_separator()
                dpg.add_text("Status", tag="umap_status_text")

            with dpg.tab(label="Training"):
                dpg.add_text("Training and data augmentation UI will be implemented here.")

def launch_gui(api_url: str):
    global API_URL
    API_URL = api_url

    dpg.create_context()
    
    with dpg.texture_registry(tag="texture_registry"):
        loading_pixel = np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32)
        loading_data = np.tile(loading_pixel, (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 1))
        g["loading_texture_id"] = dpg.add_static_texture(
            width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE, default_value=loading_data.flatten()
        )

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
                if callback:
                    callback(None, None, user_data)
        except queue.Empty:
            pass
        
        # UMAP hover logic
        if dpg.is_item_hovered("umap_plot") and g.get("umap_kdtree"):
            mouse_pos = dpg.get_plot_mouse_pos()
            if mouse_pos and g.get("dataset_root"):
                dist, idx = g["umap_kdtree"].query(mouse_pos)
                if idx != g.get("last_umap_hover_idx"):
                    g["last_umap_hover_idx"] = idx
                    item = g["umap_metadata"][idx]
                    rel_path = item['path']
                    full_path = os.path.join(g["dataset_root"], rel_path)
                    
                    threading.Thread(
                        target=threaded_load_texture_from_disk,
                        args=(full_path, "umap_preview_image", "texture_registry", (240, 240)),
                        daemon=True
                    ).start()

        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    print("Running GUI in standalone mode. Make sure the FastAPI server is running.")
    launch_gui(api_url="http://127.0.0.1:8000")
