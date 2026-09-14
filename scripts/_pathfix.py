"""Import this first in every script so `import aqf` works without `pip install -e .`,
and so API keys in a project-root `.env` file (see .env.example) are loaded automatically.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed -- fall back to real environment variables only
