# PyInstaller runtime hook to ensure sys.stdout and sys.stderr are not None in windowed (console=False) mode
import sys

class _SafeStream:
    def __init__(self):
        self._buf = []
    def write(self, s):
        if s:
            self._buf.append(s)
            if len(self._buf) > 1000:
                self._buf = self._buf[-500:]
    def flush(self):
        pass

if getattr(sys, 'stdout', None) is None:
    sys.stdout = _SafeStream()
if getattr(sys, 'stderr', None) is None:
    sys.stderr = _SafeStream()
