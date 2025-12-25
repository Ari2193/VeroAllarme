"""Load heat-map components with a fallback for direct file imports.

Pytest can import this package as a standalone module (no package context)
because of the hyphen in the folder name. To keep that working, try the
package-relative imports first and fall back to manual loading if needed.
"""

from importlib import util
from pathlib import Path
import sys
import types

try:  # Preferred path when package context is available
	from .heatmap import HeatMapAnalyzer
	from .filter import FilterHeatMap
except Exception:  # Fallback for direct/implicit imports (e.g., pytest collection)
	pkg_dir = Path(__file__).parent
	pkg_name = "filter_heat_map"

	pkg_module = sys.modules.get(pkg_name) or types.ModuleType(pkg_name)
	pkg_module.__path__ = [str(pkg_dir)]
	sys.modules[pkg_name] = pkg_module

	heatmap_name = f"{pkg_name}.heatmap"
	heatmap_spec = util.spec_from_file_location(heatmap_name, pkg_dir / "heatmap.py")
	heatmap_module = util.module_from_spec(heatmap_spec)
	assert heatmap_spec and heatmap_spec.loader and heatmap_module
	sys.modules[heatmap_name] = heatmap_module
	heatmap_spec.loader.exec_module(heatmap_module)
	HeatMapAnalyzer = heatmap_module.HeatMapAnalyzer

	filter_name = f"{pkg_name}.filter"
	filter_spec = util.spec_from_file_location(filter_name, pkg_dir / "filter.py")
	filter_module = util.module_from_spec(filter_spec)
	assert filter_spec and filter_spec.loader and filter_module
	sys.modules[filter_name] = filter_module
	filter_spec.loader.exec_module(filter_module)
	FilterHeatMap = filter_module.FilterHeatMap

__all__ = ["HeatMapAnalyzer", "FilterHeatMap"]
