"""Worker threads used by Settings for Cross-Encoder rerank models."""

import os
import subprocess
import sys
import time

from aqt.qt import QThread, pyqtSignal

from .dependency_install import _is_real_python_executable, _resolve_external_python_exe


class RerankModelDownloadWorker(QThread):
    """Download/warm a Cross-Encoder model without blocking Settings."""

    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, python_path, model_name):
        super().__init__()
        self.python_path = python_path
        self.model_name = model_name

    def run(self):
        python_exe = _resolve_external_python_exe(self.python_path) if self.python_path else None
        if not python_exe or not _is_real_python_executable(python_exe):
            self.finished_signal.emit(False, "Select a valid Python first.")
            return

        script = r'''
import fnmatch
import json
import sys

from huggingface_hub import HfApi, hf_hub_download

model = sys.argv[1]

def emit(percent, message):
    print(json.dumps({"percent": int(percent), "message": message}), flush=True)

def fmt_size(size):
    if not size:
        return "unknown size"
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"

allow_patterns = (
    "*.bin",
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "merges.txt",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer*",
    "vocab*",
)
ignore_patterns = (
    ".gitattributes",
    "README*",
    "*.md",
    "flax_model*",
    "onnx/*",
    "openvino/*",
    "rust_model*",
    "tf_model*",
)

def allowed(name):
    if any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns)

emit(1, f"Reading model file list for {model}...")
try:
    info = HfApi().model_info(model, files_metadata=True)
except TypeError:
    info = HfApi().model_info(model)
files = [
    sibling
    for sibling in info.siblings
    if sibling.rfilename and allowed(sibling.rfilename)
]
if not files:
    raise RuntimeError("No downloadable model files were found.")

total = len(files)
total_size = sum((getattr(sibling, "size", 0) or 0) for sibling in files)
emit(3, f"Found {total} files ({fmt_size(total_size)} total).")

for index, sibling in enumerate(files, start=1):
    name = sibling.rfilename
    size = fmt_size(getattr(sibling, "size", 0))
    start_pct = 5 + int((index - 1) * 75 / total)
    done_pct = 5 + int(index * 75 / total)
    emit(start_pct, f"Downloading {index}/{total}: {name} ({size})")
    hf_hub_download(repo_id=model, filename=name)
    emit(done_pct, f"Downloaded {index}/{total}: {name}")

emit(85, "Warming model in Python...")
from sentence_transformers import CrossEncoder
CrossEncoder(model)
emit(100, f"Model is ready: {model}")
print("ok", flush=True)
'''
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            proc = subprocess.Popen(
                [python_exe, "-c", script, self.model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )
            recent_output = []
            start = time.time()
            while True:
                if time.time() - start > 1800:
                    proc.kill()
                    self.finished_signal.emit(False, "Download/warmup timed out after 30 minutes.")
                    return
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    stripped = line.strip()
                    if stripped:
                        recent_output.append(stripped)
                        recent_output = recent_output[-40:]
                    if stripped == "ok":
                        continue
                    try:
                        import json

                        event = json.loads(stripped)
                        self.progress_signal.emit(
                            int(event.get("percent", 0)),
                            str(event.get("message") or ""),
                        )
                    except Exception:
                        self.progress_signal.emit(0, stripped)
                    continue
                if proc.poll() is not None:
                    break
                time.sleep(0.1)

            if proc.returncode == 0:
                self.finished_signal.emit(True, f"Model is ready: {self.model_name}")
                return
            detail = "\n".join(recent_output).strip() or "Unknown error"
            self.finished_signal.emit(False, detail[-1200:])
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))


class RerankModelVerifyWorker(QThread):
    """Verify a selected Cross-Encoder model from local cache only."""

    finished_signal = pyqtSignal(bool, str, str)

    def __init__(self, python_path, model_name):
        super().__init__()
        self.python_path = python_path
        self.model_name = model_name

    def run(self):
        python_exe = _resolve_external_python_exe(self.python_path) if self.python_path else None
        if not python_exe or not _is_real_python_executable(python_exe):
            self.finished_signal.emit(False, self.model_name, "Select a valid Python first.")
            return

        script = r'''
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import CrossEncoder

model = sys.argv[1]
CrossEncoder(model)
print("ok", flush=True)
'''
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            proc = subprocess.run(
                [python_exe, "-c", script, self.model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=180,
                creationflags=creationflags,
            )
            output = (proc.stdout or "").strip()
            if proc.returncode == 0 and "ok" in output:
                self.finished_signal.emit(True, self.model_name, f"Model is ready in local cache: {self.model_name}")
                return
            self.finished_signal.emit(False, self.model_name, (output or "Model is not available in local cache.")[-1200:])
        except subprocess.TimeoutExpired:
            self.finished_signal.emit(False, self.model_name, "Cache-only verification timed out after 180 seconds.")
        except Exception as exc:
            self.finished_signal.emit(False, self.model_name, str(exc))
