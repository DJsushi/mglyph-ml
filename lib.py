"""Compatibility shim that loads the real `lib` package from `src/`.

This module ensures ``src/`` is on ``sys.path`` and then loads the package
implementation from ``src/__init__.py``, replacing the entry in
``sys.modules['lib']`` so downstream imports behave as if the package were
installed. This keeps scripts working before installing the project.
"""

from pathlib import Path
import sys
from importlib.util import spec_from_file_location, module_from_spec

_root = Path(__file__).resolve().parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_init_path = _src / "__init__.py"
spec = spec_from_file_location("lib", str(_init_path))
module = module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules["lib"] = module

# Expose the real package symbols on this module
from lib.lib import *  # noqa: F401,F403

