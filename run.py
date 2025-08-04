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
server_instance = None

def run_server():
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
    global server_thread
    #daemon thread will be terminated when the main program exit to stop orphans
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("Server thread started.")
    
    # Wait a moment for the server to start
    time.sleep(2)

def stop_server():
    if server_instance:
        print("Signaling server to shut down...")
        server_instance.should_exit = True

if __name__ == "__main__":
    #force exit server process to not be reused
    atexit.register(stop_server)

    #start server in background
    start_server()

    print("Launching GUI...")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("Server is running and responding.")
        else:
            print(f"Server responded with status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Warning: Could not connect to server: {e}")
        print("The GUI may not work properly if the server is not running.")
    
    try:
        from gui import launch_gui
        launch_gui(api_url=API_URL)
    except Exception as e:
        print(f"An error occurred while running the GUI: {e}")
        import traceback
        traceback.print_exc()
    sys.exit(0)
