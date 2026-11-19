"""Execute the SVM Safe notebook sequentially in a clean Jupyter kernel.

Run from the repository root using the same Python environment as the notebook.
This is intentionally a full integration check and can take several minutes.
"""
import json
import os
from pathlib import Path
from queue import Empty
import sys
import tempfile
import time

from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / "src/notebooks/SVM_Safe.ipynb").read_text(encoding="utf-8"))
manager = KernelManager(kernel_name="python3")
temporary_profile = tempfile.TemporaryDirectory(prefix="svm-notebook-")
env = os.environ.copy()
env["IPYTHONDIR"] = temporary_profile.name
manager.start_kernel(cwd=str(root), env=env)
client = manager.client()
client.start_channels()
try:
    client.wait_for_ready(timeout=60)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell_id = cell.get("id", "code")
        print(f"Running cell {cell_id}", flush=True)
        message_id = client.execute("".join(cell["source"]))
        deadline = time.monotonic() + 3600
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Notebook cell {cell_id} did not finish within one hour")
            try:
                message = client.get_iopub_msg(timeout=min(remaining, 20))
            except Empty:
                continue
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "stream":
                print(content["text"], end="", flush=True)
            elif kind == "error":
                print("\n".join(content.get("traceback", [])), file=sys.stderr, flush=True)
                raise RuntimeError(f"Notebook cell {cell_id} failed")
            elif kind == "status" and content.get("execution_state") == "idle":
                break
    print("All notebook cells completed successfully.", flush=True)
finally:
    client.stop_channels()
    manager.shutdown_kernel(now=True)
    temporary_profile.cleanup()
