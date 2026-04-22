"""Colorama/transformers compatibility patches. Call before sentence_transformers import."""

import os


def _patch_colorama_early():
    """Patch colorama ErrorHandler early to prevent transformers import errors."""
    try:
        import sys
        import types

        if not hasattr(sys, "__stderr_original__"):
            sys.__stderr_original__ = sys.stderr

        try:
            import colorama
            if hasattr(colorama, "ansitowin32") and hasattr(colorama.ansitowin32, "ErrorHandler"):
                ErrorHandler = colorama.ansitowin32.ErrorHandler

                def _get_real_stream(obj):
                    if hasattr(sys, "__stderr_original__"):
                        return sys.__stderr_original__
                    wrapped = getattr(obj, "__wrapped__", None)
                    if wrapped is None:
                        return None
                    depth = 0
                    while hasattr(wrapped, "__wrapped__") and depth < 10:
                        next_wrapped = wrapped.__wrapped__
                        if next_wrapped is None:
                            break
                        if hasattr(wrapped, "__class__") and wrapped.__class__.__name__ == "ErrorHandler":
                            wrapped = next_wrapped
                            depth += 1
                        else:
                            break
                    return wrapped

                def _flush_method(self):
                    real_stream = _get_real_stream(self)
                    if real_stream and hasattr(real_stream, "flush"):
                        return real_stream.flush()
                    if hasattr(sys, "__stderr_original__"):
                        return sys.__stderr_original__.flush()
                    return None

                _original_getattr = getattr(ErrorHandler, "__getattr__", None)

                def _patched_getattr(self, name):
                    if name == "flush":
                        if not hasattr(self, "_cached_flush"):
                            self._cached_flush = types.MethodType(_flush_method, self)
                        return self._cached_flush
                    if _original_getattr:
                        try:
                            return _original_getattr(self, name)
                        except AttributeError:
                            raise
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

                ErrorHandler.__getattr__ = _patched_getattr

                if hasattr(sys, "stderr"):
                    stderr = sys.stderr
                    if hasattr(stderr, "__class__") and (
                        stderr.__class__.__name__ == "ErrorHandler"
                        or "colorama" in str(type(stderr))
                        or isinstance(stderr, ErrorHandler)
                    ):
                        real_stream = _get_real_stream(stderr)
                        if real_stream and hasattr(real_stream, "flush"):
                            stderr.flush = types.MethodType(lambda: real_stream.flush(), stderr)
                        else:
                            stderr.flush = types.MethodType(_flush_method, stderr)

                if hasattr(colorama, "init"):
                    original_init = colorama.init

                    def patched_init(*args, **kwargs):
                        result = original_init(*args, **kwargs)
                        if hasattr(sys, "stderr"):
                            stderr = sys.stderr
                            if hasattr(stderr, "__class__") and stderr.__class__.__name__ == "ErrorHandler":
                                real_stream = _get_real_stream(stderr)
                                if real_stream and hasattr(real_stream, "flush"):
                                    stderr.flush = types.MethodType(lambda: real_stream.flush(), stderr)
                                else:
                                    stderr.flush = types.MethodType(_flush_method, stderr)
                        return result

                    colorama.init = patched_init
        except ImportError:
            pass
        except Exception as e:
            try:
                addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                log_file = os.path.join(addon_dir, "debug_log.txt")
                import datetime
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.datetime.now()}] Colorama patch error: {e}\n")
            except Exception:
                pass
    except Exception:
        pass


def _ensure_stderr_patched():
    """Ensure sys.stderr has flush if it's an ErrorHandler - call before importing transformers."""
    try:
        import sys
        import types

        if not hasattr(sys, "stderr"):
            return

        stderr = sys.stderr
        if (
            hasattr(stderr, "__class__")
            and (
                stderr.__class__.__name__ == "ErrorHandler"
                or "colorama" in str(type(stderr))
            )
        ):
            if not hasattr(stderr, "flush"):
                wrapped = getattr(stderr, "__wrapped__", None)
                if wrapped:
                    while hasattr(wrapped, "__wrapped__"):
                        next_wrapped = wrapped.__wrapped__
                        if next_wrapped is None:
                            break
                        if hasattr(wrapped, "__class__") and wrapped.__class__.__name__ == "ErrorHandler":
                            wrapped = next_wrapped
                        else:
                            break
                    if wrapped and hasattr(wrapped, "flush"):
                        stderr.flush = types.MethodType(lambda: wrapped.flush(), stderr)
                        return

                if hasattr(sys, "__stderr_original__"):
                    stderr.flush = types.MethodType(lambda: sys.__stderr_original__.flush(), stderr)
                else:
                    stderr.flush = lambda: None
    except Exception:
        pass
