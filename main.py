"""AutoClipper root CLI entrypoint.
Imports and runs the CLI from backend/main.py.
"""
import importlib.util
import os
import sys

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

spec = importlib.util.spec_from_file_location("backend_main", os.path.join(backend_dir, "main.py"))
backend_module = importlib.util.module_from_spec(spec)
sys.modules["backend_main"] = backend_module
spec.loader.exec_module(backend_module)

main = backend_module.main

if __name__ == "__main__":
    sys.exit(main())
