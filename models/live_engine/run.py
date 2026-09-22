import subprocess
import sys
import os

python_exe = sys.executable  # path to current Python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "live_engine")
script_path = os.path.join(MODEL_DIR, "main.py")

subprocess.Popen(
    ["cmd", "/k", python_exe, script_path],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
