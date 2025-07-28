import dearpygui.dearpygui as dpg
import requests
import threading
import os

# --- Constants ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# --- Global State ---
API_URL = ""
# UI state
g = {
    "dataset_root": None,
    "subdirectories": [],
    "selected_subdir": None,
    "status_text": "Welcome! Please select a dataset root directory.",
    "is_indexing": False,
}

# --- API Communication (threaded) ---

def update_status(text: str):
    """Helper to update the status bar text from any thread."""
    if dpg.does_item_exist("status_text"):
        dpg.set_value("status_text", text)

def threaded_api_call(target, on_success=None, on_error=None, **kwargs):
    """Generic wrapper to run API calls in a background thread."""
    def thread_target():
        try:
            # The 'target' is a function like requests.post
            response = target(**kwargs)
            response.raise_for_status()
            if on_success:
                # Pass the JSON response to the success callback
                on_success(response.json())
        except requests.RequestException as e:
            error_message = f"API Error: {e}"
            if e.response is not None:
                try:
                    detail = e.response.json().get("detail", e.response.text)
                    error_message = f"API Error ({e.response.status_code}): {detail}"
                except requests.exceptions.JSONDecodeError:
                    pass # Use the original error message
            print(error_message)
            if on_error:
                on_error(error_message)
            else:
                update_status(error_message)

    thread = threading.Thread(target=thread_target)
    thread.daemon = True
    thread.start()

# --- Callbacks for UI actions ---

def callback_select_dataset_root(sender, app_data):
    """Opens the directory selection dialog."""
    dpg.add_file_dialog(
        directory_selector=True,
        show=True,
        callback=callback_dataset_root_selected,
        tag="dataset_root_dialog",
        width=700, height=400,
        modal=True
    )

def callback_dataset_root_selected(sender, app_data):
    """Handles the response from the directory selection dialog."""
    # Check if a path was selected (and not cancelled)
    if 'file_path_name' in app_data and app_data['file_path_name']:
        path = app_data['file_path_name']
        update_status(f"Setting dataset root to: {path}...")

        def on_success(data):
            g["dataset_root"] = path
            dpg.set_value("dataset_root_text", f"Current Root: {path}")
            update_status("Dataset root set. Fetching subdirectories...")
            # Now fetch the subdirectories
            threaded_api_call(
                target=requests.get,
                on_success=on_get_directories_success,
                url=f"{API_URL}/directories"
            )

        threaded_api_call(
            target=requests.post,
            on_success=on_success,
            url=f"{API_URL}/set-dataset-root",
            json={"path": path}
        )

def on_get_directories_success(data):
    """Handles successful fetching of subdirectories."""
    g["subdirectories"] = data.get("directories", [])
    if not g["subdirectories"]:
        update_status("No subdirectories with images found. You can index the root.")
        g["subdirectories"] = ['.'] # Add root if no other dirs
    else:
        update_status("Subdirectories loaded. Please select one to index.")

    dpg.configure_item("subdir_selector", items=g["subdirectories"])
    if g["subdirectories"]:
        g["selected_subdir"] = g["subdirectories"][0]
        dpg.set_value("subdir_selector", g["selected_subdir"])
    
    dpg.enable_item("build_index_button")
    dpg.enable_item("subdir_selector")


def callback_build_index(sender, app_data):
    """Starts the indexing process for the selected subdirectory."""
    if not g["selected_subdir"] or g["is_indexing"]:
        return

    g["is_indexing"] = True
    dpg.disable_item("build_index_button")
    dpg.disable_item("select_root_button")
    update_status(f"Building index for '{g['selected_subdir']}'. This may take a while...")

    def on_success(data):
        total = data.get('total', 'N/A')
        update_status(f"Index built successfully for '{g['selected_subdir']}' with {total} images.")
        g["is_indexing"] = False
        dpg.enable_item("build_index_button")
        dpg.enable_item("select_root_button")
        # TODO: Trigger loading of all images into the gallery view

    def on_error(error_message):
        update_status(f"Error building index: {error_message}")
        g["is_indexing"] = False
        dpg.enable_item("build_index_button")
        dpg.enable_item("select_root_button")

    threaded_api_call(
        target=requests.post,
        on_success=on_success,
        on_error=on_error,
        url=f"{API_URL}/build-index",
        params={"img_dir": g["selected_subdir"]},
        timeout=600 # 10 minute timeout for indexing
    )

def callback_subdir_selected(sender, app_data):
    """Stores the selected subdirectory from the dropdown."""
    g["selected_subdir"] = app_data

# --- UI Setup ---

def setup_ui():
    """Creates all the UI elements for the application."""
    with dpg.window(label="Main", tag="primary_window"):
        # --- Top Control Panel ---
        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Select Dataset Root...",
                    callback=callback_select_dataset_root,
                    tag="select_root_button"
                )
                dpg.add_text("Current Root: Not Set", tag="dataset_root_text")

            with dpg.group(horizontal=True):
                dpg.add_text("Index Target:")
                dpg.add_combo(
                    items=g["subdirectories"],
                    tag="subdir_selector",
                    callback=callback_subdir_selected,
                    width=250,
                    enabled=False # Enabled after root is set
                )
                dpg.add_button(
                    label="Build Index",
                    tag="build_index_button",
                    callback=callback_build_index,
                    enabled=False # Enabled after root is set
                )

        dpg.add_separator()

        # Placeholder for search and results
        dpg.add_text("Search and results will go here.")

        dpg.add_separator()

        # --- Status Bar ---
        dpg.add_text(g["status_text"], tag="status_text")


def launch_gui(api_url: str):
    """
    Sets up and runs the Dear PyGui interface.
    """
    global API_URL
    API_URL = api_url

    dpg.create_context()

    setup_ui()

    # --- Viewport Setup ---
    dpg.create_viewport(
        title="CLIP Semantic Search",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    # --- Main Render Loop ---
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    # This allows running the GUI directly for development,
    # but it won't have a running server unless you start it manually.
    print("Running GUI in standalone mode. Make sure the FastAPI server is running.")
    launch_gui(api_url="http://127.0.0.1:8000")
