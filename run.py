import subprocess
import sys
import atexit
import time
import requests

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 8000
API_URL = f"http://{HOST}:{PORT}"

server_process = None

def start_server():
    """Starts the FastAPI server as a background process."""
    global server_process
    print("Starting backend server...")
    command = [
        sys.executable,
        "-m", "uvicorn",
        "app:app",
        "--host", HOST,
        "--port", str(PORT),
        "--log-level", "warning"
    ]
    # Use Popen to run the server in the background
    server_process = subprocess.Popen(command)
    print(f"Server process started with PID: {server_process.pid}")

def stop_server():
    """Stops the FastAPI server if it's running."""
    if server_process:
        print("Stopping backend server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("Server stopped.")
        except subprocess.TimeoutExpired:
            print("Server did not terminate in time, killing it.")
            server_process.kill()

def wait_for_server():
    """Waits for the server to become responsive."""
    start_time = time.time()
    while time.time() - start_time < 20:  # 20-second timeout
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                print("Server is up and running.")
                return True
        except requests.ConnectionError:
            time.sleep(0.5) # Wait before retrying
    print("Error: Server did not start within the timeout period.")
    return False


if __name__ == "__main__":
    # Register the cleanup function to run on exit
    atexit.register(stop_server)

    start_server()

    if wait_for_server():
        try:
            # Import and launch the GUI
            from gui import launch_gui
            launch_gui(api_url=API_URL)
        except Exception as e:
            print(f"An error occurred while running the GUI: {e}")
    else:
        print("Could not start the GUI because the server failed to start.")

    # The stop_server function will be called automatically on exit
    sys.exit(0)
