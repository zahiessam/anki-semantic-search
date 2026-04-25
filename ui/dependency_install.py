"""Dependency detection and repair helpers for the settings UI."""

# ============================================================================
# Imports
# ============================================================================

import os
import subprocess
import sys
import urllib.request

from aqt import mw
from aqt.qt import *
from aqt.utils import showInfo, tooltip

from ..core.compat import _ensure_stderr_patched, _patch_colorama_early
from ..utils.log import log_debug


# ============================================================================
# Dependency Detection And Installation Helpers
# ============================================================================

def check_vc_redistributables():



    """Check if Visual C++ Redistributables are installed"""



    import os



    import winreg







    if os.name != 'nt':  # Not Windows



        return True  # Assume OK on non-Windows







    try:



        # Check for Visual C++ 2015-2022 Redistributables (x64)



        # They're registered in the Windows registry



        vc_versions = [



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"),



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"),



        ]







        # Also check for newer versions (2017-2022)



        for major_version in range(14, 18):  # 14.0 to 17.0



            vc_versions.extend([



                (winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\Microsoft\\VisualStudio\\{major_version}.0\\VC\\Runtimes\\x64"),



                (winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\WOW6432Node\\Microsoft\\VisualStudio\\{major_version}.0\\VC\\Runtimes\\x64"),



            ])







        # Check if any version is installed - need to actually read a value to verify



        found_any = False



        for hkey, key_path in vc_versions:



            try:



                with winreg.OpenKey(hkey, key_path) as key:



                    # Try to read a value to ensure the key is valid



                    try:



                        version = winreg.QueryValueEx(key, "Version")[0]



                        if version:  # If we got a version, it's installed



                            found_any = True



                            log_debug(f"Found VC++ Redistributables: {key_path}, Version: {version}")



                            break



                    except (FileNotFoundError, OSError):



                        # Key exists but no Version value - still might be installed



                        # Check for other indicators



                        try:



                            # Try to enumerate values (with safety limit to prevent infinite loop)



                            i = 0



                            max_iterations = 1000  # Safety limit



                            while i < max_iterations:



                                try:



                                    name, value, _ = winreg.EnumValue(key, i)



                                    if name and value:



                                        found_any = True



                                        break



                                    i += 1



                                except OSError:



                                    break



                            if found_any:



                                break



                        except:



                            pass



            except (FileNotFoundError, OSError):



                continue







        if found_any:



            return True



        return False  # No VC++ redistributables found



    except Exception as e:



        log_debug(f"Error checking VC++ redistributables: {e}")



        # If we can't check, return None (unknown) instead of assuming True



        return None  # Unknown status







def install_vc_redistributables():



    """Download and install Visual C++ Redistributables"""



    import os



    import sys



    import urllib.request



    import subprocess



    import tempfile







    if os.name != 'nt':  # Not Windows



        showInfo("Visual C++ Redistributables are only needed on Windows.")



        return False







    vc_redist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"



    vc_redist_filename = "vc_redist.x64.exe"







    try:



        # Check if already installed - but be more thorough



        vc_status = check_vc_redistributables()



        if vc_status is True:



            # Double-check by trying to actually use a DLL that requires VC++



            # If PyTorch is failing, VC++ might not actually be working



            reply = QMessageBox.question(



                mw,



                "VC++ Redistributables Check",



                "VC++ Redistributables appear to be installed according to the registry.\n\n"



                "However, if you're still experiencing PyTorch DLL errors, they may not be working correctly.\n\n"



                "Options:\n"



                "1. Reinstall VC++ Redistributables anyway (recommended)\n"



                "2. Use keyword-only search (no PyTorch needed)\n"



                "Reinstall VC++ Redistributables?",



                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,



                QMessageBox.StandardButton.Yes



            )



            if reply != QMessageBox.StandardButton.Yes:



                return False



            # Continue with installation anyway



        elif vc_status is None:



            # Unknown status - proceed with installation



            pass



        # If False, continue with installation







        # Create a progress dialog



        progress_dialog = QDialog(mw)



        progress_dialog.setWindowTitle("Installing Visual C++ Redistributables")



        progress_dialog.setMinimumWidth(500)



        progress_dialog.setMinimumHeight(200)



        progress_dialog.setModal(True)



        layout = QVBoxLayout(progress_dialog)







        status_label = QLabel("Downloading Visual C++ Redistributables...")



        status_label.setWordWrap(True)



        layout.addWidget(status_label)







        progress_bar = QProgressBar()



        progress_bar.setRange(0, 0)  # Indeterminate



        layout.addWidget(progress_bar)







        close_btn = QPushButton("Close")



        close_btn.setEnabled(False)



        layout.addWidget(close_btn)







        progress_dialog.show()



        QApplication.processEvents()







        # Download the installer



        temp_dir = tempfile.gettempdir()



        installer_path = os.path.join(temp_dir, vc_redist_filename)







        def download_installer():



            try:



                status_label.setText("Downloading Visual C++ Redistributables installer...\nThis may take a minute.")



                QApplication.processEvents()







                urllib.request.urlretrieve(vc_redist_url, installer_path)







                status_label.setText("Download complete. Launching installer...\n\nYou may need to grant administrator privileges.")



                QApplication.processEvents()







                # Launch the installer



                # /quiet = silent install, /norestart = don't restart



                # /passive = show progress but no user interaction needed



                subprocess.Popen([installer_path, "/passive", "/norestart"], shell=True)







                status_label.setText("\u2705 Installer launched!\n\nPlease follow the installation wizard.\nAfter installation completes, restart Anki.")



                close_btn.setEnabled(True)



                progress_bar.setRange(0, 100)



                progress_bar.setValue(100)







                log_debug("VC++ Redistributables installer launched successfully")



                return True



            except Exception as e:



                status_label.setText(f"\xe2\x9d\u0152 Error: {str(e)}\n\nYou can manually download and install from:\n{vc_redist_url}")



                close_btn.setEnabled(True)



                log_debug(f"Error installing VC++ redistributables: {e}")



                return False







        # Run download in a thread to avoid blocking



        import threading



        thread = threading.Thread(target=download_installer, daemon=True)



        thread.start()







        # Don't wait for thread - let user close dialog when ready



        close_btn.clicked.connect(progress_dialog.close)







        return True



    except Exception as e:



        error_msg = (



            f"Error preparing VC++ Redistributables installation: {str(e)}\n\n"



            f"Please manually download and install from:\n{vc_redist_url}\n\n"



            "After installation, restart Anki."



        )



        showInfo(error_msg)



        log_debug(f"Error in install_vc_redistributables: {e}")



        return False







# --- PyTorch And Python Environment Helpers ---

def get_pytorch_dll_error_guidance():



    """Get guidance message for PyTorch DLL loading errors"""



    import sys



    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"







    vc_status = check_vc_redistributables()



    vc_message = ""



    if vc_status is False:



        vc_message = "\n\xe2\u0161\xa0\ufe0f Visual C++ Redistributables appear to be MISSING!\n   Click 'Install VC++ Redistributables' button below.\n\n"



    elif vc_status is None:



        vc_message = "\n\xe2\u0161\xa0\ufe0f Could not verify Visual C++ Redistributables installation.\n   You may need to install them manually.\n\n"







    guidance = (



        "PyTorch DLL Loading Error Detected\n\n"



        f"Python version: {python_version}\n\n"



        f"{vc_message}"



        "Common causes and solutions:\n\n"



        "1. Missing Visual C++ Redistributables:\n"



        "   - Click 'Install VC++ Redistributables' button below\n"



        "   - Or download manually from:\n"



        "   - https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"



        "2. Python 3.13 compatibility:\n"



        "   - Python 3.13 is very new and PyTorch may not have full support yet\n"



        "   - Try reinstalling PyTorch with CPU-only version:\n"



        "   - Use 'Fix PyTorch DLL Issue' button below\n\n"



        "3. Corrupted installation:\n"



        "   - Try: pip uninstall sentence-transformers torch\n"



        "   - Then reinstall using 'Install / show command for external Python' in settings\n\n"



        "4. Alternative: Use Anki with Python 3.11 or 3.12 for better compatibility"



    )



    return guidance







def _patch_colorama_for_transformers():



    """Patch colorama ErrorHandler to add flush attribute for transformers compatibility.



    This is a wrapper that ensures the patch is applied (the actual patch runs at module load)."""



    # The actual patching happens at module load time via _patch_colorama_early()



    # This function is kept for backward compatibility and to ensure patch is applied



    # if called before module initialization completes



    try:



        _patch_colorama_early()



    except:



        pass  # Silently fail







def check_dependency_installed(package_name):



    """Check if a Python package is installed"""



    try:



        # Patch colorama before importing sentence_transformers to avoid AttributeError



        if 'sentence_transformers' in package_name or 'transformers' in package_name:



            _patch_colorama_for_transformers()



            _ensure_stderr_patched()



        __import__(package_name.replace('-', '_'))



        return True



    except (ImportError, OSError, ModuleNotFoundError, AttributeError, Exception) as e:



        # OSError can occur when PyTorch DLLs fail to load (e.g., missing Visual C++ Redistributables)



        # ModuleNotFoundError is a subclass of ImportError but we catch it explicitly for clarity



        # AttributeError can occur due to library compatibility issues (e.g., colorama/transformers)



        if isinstance(e, OSError) and 'torch' in str(e).lower():



            log_debug(f"PyTorch DLL error detected: {e}")



        elif isinstance(e, AttributeError):



            log_debug(f"AttributeError during import (likely compatibility issue): {e}")



        return False







def _resolve_external_python_exe(python_path):



    """Resolve 'Python for Cross-Encoder' path to python executable. Returns None if invalid."""



    import os



    path = (python_path or "").strip()



    if not path:



        return None



    if os.path.isfile(path):



        return path



    if os.path.isdir(path):



        exe = os.path.join(path, "python.exe")



        if os.path.isfile(exe):



            return exe



        exe = os.path.join(path, "python")



        if os.path.isfile(exe):



            return exe



    return None











def try_alternative_pytorch_install():



    """Try alternative PyTorch installation methods"""



    import sys



    import subprocess







    methods = [



        {



            "name": "Method 1: PyTorch 2.0.1 (Older, more stable)",



            "command": [sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2", "--index-url", "https://download.pytorch.org/whl/cpu"]



        },



        {



            "name": "Method 2: PyTorch 2.1.0 (Mid-version)",



            "command": [sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0", "--index-url", "https://download.pytorch.org/whl/cpu"]



        },



        {



            "name": "Method 3: Latest PyTorch (Current default)",



            "command": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]



        },



        {



            "name": "Method 4: PyTorch without CUDA (pip default)",



            "command": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]



        }



    ]







    dialog = QDialog(mw)



    dialog.setWindowTitle("Try Alternative PyTorch Installation")



    dialog.setMinimumWidth(500)



    layout = QVBoxLayout(dialog)







    info_label = QLabel(



        "If the standard installation failed, try these alternative methods:\n\n"



        "Select a method to try:"



    )



    info_label.setWordWrap(True)



    layout.addWidget(info_label)







    method_combo = QComboBox()



    for method in methods:



        method_combo.addItem(method["name"])



    layout.addWidget(method_combo)







    button_layout = QHBoxLayout()



    try_btn = QPushButton("Try This Method")



    cancel_btn = QPushButton("Cancel")



    button_layout.addWidget(try_btn)



    button_layout.addWidget(cancel_btn)



    layout.addLayout(button_layout)







    def try_method():



        selected_idx = method_combo.currentIndex()



        method = methods[selected_idx]



        dialog.close()







        # Show progress



        progress = QDialog(mw)



        progress.setWindowTitle("Installing PyTorch")



        progress_layout = QVBoxLayout(progress)



        status = QLabel(f"Trying: {method['name']}\n\nThis may take several minutes...")



        status.setWordWrap(True)



        progress_layout.addWidget(status)



        progress.show()



        QApplication.processEvents()







        try:



            result = subprocess.run(



                method["command"],



                capture_output=True,



                text=True,



                timeout=600



            )







            if result.returncode == 0:



                status.setText("\u2705 Installation successful!\n\nTesting import...")



                QApplication.processEvents()







                # Test import



                try:



                    _patch_colorama_for_transformers()



                    _ensure_stderr_patched()



                    import torch



                    status.setText(f"\u2705 Success! PyTorch {torch.__version__} installed and working.\n\nNow install sentence-transformers.")



                    showInfo(f"PyTorch {torch.__version__} installed successfully!\n\nNow use 'Install / show command for external Python' to install sentence-transformers.")



                except Exception as e:



                    status.setText(f"\xe2\u0161\xa0\ufe0f Installed but import failed: {e}\n\nTry installing VC++ Redistributables.")



                    showInfo(f"PyTorch installed but import failed: {e}\n\nTry installing VC++ Redistributables first.")



            else:



                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"



                status.setText(f"\xe2\x9d\u0152 Installation failed:\n{error_msg}")



                showInfo(f"Installation failed. Try another method or install VC++ Redistributables.")



        except Exception as e:



            status.setText(f"Γ¥î Error: {e}")



            showInfo(f"Error: {e}")







        close_btn = QPushButton("Close")



        close_btn.clicked.connect(progress.close)



        progress_layout.addWidget(close_btn)







    try_btn.clicked.connect(try_method)



    cancel_btn.clicked.connect(dialog.close)







    dialog.exec()







def fix_pytorch_dll_issue():



    """Fix PyTorch DLL issues by reinstalling with CPU-only version"""



    import sys



    import subprocess







    reply = QMessageBox.question(



        mw,



        "Fix PyTorch DLL Issue",



        "This will reinstall PyTorch with a CPU-only version that's more compatible.\n\n"



        "Steps:\n"



        "1. Uninstall existing PyTorch packages\n"



        "2. Install CPU-only PyTorch from official repository\n"



        "3. Reinstall sentence-transformers\n\n"



        "\xe2\u0161\xa0\ufe0f IMPORTANT: If this fails, you may need to:\n"



        "- Install Visual C++ Redistributables first\n"



        "- Try alternative PyTorch versions\n"



        "- Use keyword-only search (no embeddings needed)\n\n"



        "This may take a few minutes. Continue?",



        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,



        QMessageBox.StandardButton.Yes



    )







    if reply != QMessageBox.StandardButton.Yes:



        return







    # Create progress dialog



    progress_dialog = QDialog(mw)



    progress_dialog.setWindowTitle("Fixing PyTorch DLL Issue")



    progress_dialog.setMinimumWidth(600)



    progress_dialog.setMinimumHeight(500)



    progress_dialog.setModal(False)



    progress_layout = QVBoxLayout(progress_dialog)







    status_label = QLabel("Preparing...")



    status_label.setWordWrap(True)



    progress_layout.addWidget(status_label)







    log_text = QTextEdit()



    log_text.setReadOnly(True)



    log_text.setMaximumHeight(300)



    log_text.setFont(QFont("Courier", 9))



    log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")



    progress_layout.addWidget(log_text)







    close_button = QPushButton("Close")



    close_button.setEnabled(False)



    close_button.clicked.connect(progress_dialog.close)



    progress_layout.addWidget(close_button)







    progress_dialog.show()



    QApplication.processEvents()







    def log(msg):



        log_text.append(msg)



        log_text.verticalScrollBar().setValue(log_text.verticalScrollBar().maximum())



        QApplication.processEvents()



        log_debug(msg)







    try:



        # Step 1: Uninstall PyTorch packages



        status_label.setText("Step 1/3: Uninstalling existing PyTorch packages...")



        log("Uninstalling torch, torchvision, torchaudio...")







        packages_to_uninstall = ['torch', 'torchvision', 'torchaudio', 'sentence-transformers']



        for pkg in packages_to_uninstall:



            try:



                result = subprocess.run(



                    [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],



                    capture_output=True,



                    text=True,



                    timeout=120



                )



                if result.returncode == 0:



                    log(f"\u2705 Uninstalled {pkg}")



                else:



                    log(f"\xe2\u0161\xa0\ufe0f {pkg} may not have been installed")



            except Exception as e:



                log(f"\xe2\u0161\xa0\ufe0f Error uninstalling {pkg}: {e}")







        # Step 2: Install CPU-only PyTorch



        status_label.setText("Step 2/3: Installing CPU-only PyTorch...")



        log("Installing PyTorch CPU-only version from official repository...")



        log("This may take several minutes...")







        result = subprocess.run(



            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",



             "--index-url", "https://download.pytorch.org/whl/cpu"],



            capture_output=True,



            text=True,



            timeout=600



        )







        if result.returncode == 0:



            log("\u2705 PyTorch CPU-only installed successfully")



        else:



            log(f"Γ¥î Error installing PyTorch:")



            for line in result.stderr.split('\n')[-10:]:



                if line.strip():



                    log(line)



            raise Exception("PyTorch installation failed")







        # Step 3: Reinstall sentence-transformers



        status_label.setText("Step 3/3: Reinstalling sentence-transformers...")



        log("Installing sentence-transformers...")







        result = subprocess.run(



            [sys.executable, "-m", "pip", "install", "sentence-transformers"],



            capture_output=True,



            text=True,



            timeout=300



        )







        if result.returncode == 0:



            log("\u2705 sentence-transformers installed successfully")



        else:



            log(f"Γ¥î Error installing sentence-transformers:")



            for line in result.stderr.split('\n')[-10:]:



                if line.strip():



                    log(line)



            raise Exception("sentence-transformers installation failed")







        # Verify installation



        status_label.setText("Verifying installation...")



        log("Testing import...")







        try:



            _patch_colorama_for_transformers()



            _ensure_stderr_patched()



            from sentence_transformers import SentenceTransformer



            log("\u2705 Import test successful!")



            status_label.setText("\u2705 Fix completed successfully!")



            status_label.setStyleSheet("color: green; font-weight: bold;")



            showInfo("PyTorch DLL issue fixed! You may need to restart Anki for changes to take effect.")



        except Exception as e:



            log(f"\xe2\x9d\u0152 Import test failed: {e}")



            status_label.setText("\xe2\u0161\xa0\ufe0f Installation completed but import test failed")



            status_label.setStyleSheet("color: orange; font-weight: bold;")







            # Add helpful buttons



            button_layout = QHBoxLayout()







            try_alt_btn = QPushButton("Try Alternative PyTorch Version")



            try_alt_btn.clicked.connect(lambda: (progress_dialog.close(), try_alternative_pytorch_install()))



            button_layout.addWidget(try_alt_btn)







            vc_btn = QPushButton("Install VC++ Redistributables")



            vc_btn.clicked.connect(lambda: (progress_dialog.close(), install_vc_redistributables()))



            button_layout.addWidget(vc_btn)







            use_keyword_btn = QPushButton("Use Keyword-Only Search (No PyTorch)")



            use_keyword_btn.clicked.connect(lambda: (



                progress_dialog.close(),



                showInfo("You can use the addon in keyword-only mode!\n\n"



                        "1. Go to Settings\n"



                        "2. Change 'Search Method' to 'Keyword Only'\n"



                        "3. The addon will work without embeddings.")



            ))



            button_layout.addWidget(use_keyword_btn)







            progress_layout.addLayout(button_layout)







            error_msg = (



                f"Installation completed but verification failed: {e}\n\n"



                "Options:\n"



                "1. Try alternative PyTorch version (button above)\n"



                "2. Install VC++ Redistributables (button above)\n"



                "3. Use keyword-only search mode (no embeddings needed)\n"



                "4. Check the log for details"



            )



            showInfo(error_msg)







    except Exception as e:



        log(f"\xe2\x9d\u0152 Error: {e}")



        status_label.setText(f"\xe2\x9d\u0152 Error: {e}")



        status_label.setStyleSheet("color: red; font-weight: bold;")



        showInfo(f"Error fixing PyTorch: {e}")



    finally:



        close_button.setEnabled(True)







def _check_sentence_transformers_installed_subprocess():



    """Check if sentence-transformers is usable in Anki's Python via subprocess (avoids in-process import failures on Python 3.13)."""



    try:



        import subprocess



        import sys



        result = subprocess.run(



            [sys.executable, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],



            capture_output=True, text=True, timeout=15,



            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



        )



        return result.returncode == 0 and 'ok' in (result.stdout or '')



    except Exception:



        return False











# --- Optional Dependency Installer ---

def install_dependencies(python_exe=None):



    """Show manual install instructions for optional dependencies (no auto pip install).



    python_exe: None = Anki's Python; else path to external python.exe for Cross-Encoder."""



    import sys







    if python_exe:



        target_python = python_exe



        target_label = "External Python for Cross-Encoder"



    else:



        target_python = sys.executable



        target_label = "Anki's Python"







    # Check if already installed



    try:



        import subprocess



        result = subprocess.run(



            [target_python, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],



            capture_output=True, text=True, timeout=15,



            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



        )



        if result.returncode == 0 and 'ok' in (result.stdout or ''):



            showInfo("\u2705 sentence-transformers is already available.\n\nClick 'Check again' in Settings to enable Cross-Encoder.")



            return



    except Exception:



        pass







    pip_cmd = f'"{target_python}" -m pip install sentence-transformers'



    msg = (



        "Optional: Cross-Encoder re-ranking (better retrieval quality)\n\n"



        f"Python executable: {target_python}\n\n"



        "Copy this command and run it in a terminal:\n\n"



        f"  {pip_cmd}\n\n"



        f"Where to run: Use the Python above, or the one set under 'Use external Python' in Settings.\n\n"



        "See config.md in the add-on folder for troubleshooting."



    )



    dlg = QMessageBox(mw)



    dlg.setWindowTitle("Manual Install: sentence-transformers")



    dlg.setText(msg)



    dlg.setIcon(QMessageBox.Icon.Information)



    copy_btn = dlg.addButton("Copy command", QMessageBox.ButtonRole.ActionRole)



    dlg.addButton(QMessageBox.StandardButton.Ok)



    dlg.exec()



    if dlg.clickedButton() == copy_btn:



        QApplication.clipboard().setText(pip_cmd)



        tooltip("Command copied to clipboard")
