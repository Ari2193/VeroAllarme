"""
Pytest configuration for compare-events tests.
"""

import sys
from pathlib import Path
import importlib.util

# Setup compare-events import with proper module registration
compare_events_path = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "compare_events",
    compare_events_path / "__init__.py"
)
compare_events_module = importlib.util.module_from_spec(spec)
sys.modules["compare_events"] = compare_events_module
spec.loader.exec_module(compare_events_module)

