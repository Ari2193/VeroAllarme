import pathlib

# Avoid pytest trying to import the package __init__.py (hyphenated path causes issues)
collect_ignore = [str((pathlib.Path(__file__).parent.parent / "__init__.py").resolve())]
