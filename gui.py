import dearpygui.dearpygui as dpg

# --- Constants ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# --- Global State ---
# This will hold the base URL for the API, passed from run.py
API_URL = ""

def launch_gui(api_url: str):
    """
    Sets up and runs the Dear PyGui interface.
    """
    global API_URL
    API_URL = api_url

    dpg.create_context()

    # --- Main Window ---
    with dpg.window(label="Main", tag="primary_window"):
        dpg.add_text("Semantic Image Search UI")
        dpg.add_text(f"Connected to API at: {API_URL}")

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
