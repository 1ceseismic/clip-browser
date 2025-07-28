import sys
import atexit
import time
import requests
import threading
import uvicorn

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 8000
API_URL = f"http://{HOST}:{PORT}"

server_thread = None
# Use a global variable for the server instance to allow shutdown from a different thread
server_instance = None

def run_server():
    """Runs the Uvicorn server in the current thread (blocking)."""
    global server_instance
    config = uvicorn.Config(
        "app:app",
        host=HOST,
        port=PORT,
        log_level="warning"
    )
    server_instance = uvicorn.Server(config)
    server_instance.run()

def start_server():
    """Starts the FastAPI server in a background daemon thread."""
    global server_thread
    # A daemon thread will be automatically terminated when the main program exits.
    # This is crucial for preventing orphaned server processes.
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("Server thread started.")

def stop_server():
    """
    Signals the Uvicorn server to shut down gracefully.
    This function is registered with atexit to be called on program termination.
    """
    if server_instance:
        print("Signaling server to shut down...")
        # Uvicorn's graceful shutdown mechanism
        server_instance.should_exit = True
        # The daemon thread will exit shortly after this.
        # We don't join it here as atexit handlers have limitations.


if __name__ == "__main__":
    # Register the cleanup function to run when the script exits.
    # This is the key to preventing the "address already in use" error.
    atexit.register(stop_server)

    # Start the server in the background.
    start_server()

    # Launch the GUI immediately without waiting for the server.
    # The GUI's internal poller will handle the connection status.
    print("Launching GUI...")
    try:
        from gui import launch_gui
        launch_gui(api_url=API_URL)
    except Exception as e:
        print(f"An error occurred while running the GUI: {e}")

    # When the GUI window is closed, the main thread will exit.
    # The atexit handler will run, signaling the daemon server thread to stop.
    # The process then terminates.
    sys.exit(0)
