"""
Dependency Analyzer for AI Agent
Parses import/include statements and builds a simple dependency graph
for multi-file project analysis.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Simple dependency graph for a project."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # file_path -> metadata
        self.edges: List[Tuple[str, str]] = []  # (from_file, to_file)

    def add_file(self, file_path: str, language: str = "unknown") -> None:
        """Add a file node to the graph."""
        self.nodes[file_path] = {"language": language, "imports": [], "imported_by": []}

    def add_dependency(self, from_file: str, to_file: str, import_name: str = "") -> None:
        """Add a dependency edge."""
        self.edges.append((from_file, to_file))
        if from_file in self.nodes:
            self.nodes[from_file]["imports"].append({"target": to_file, "name": import_name})
        if to_file in self.nodes:
            self.nodes[to_file]["imported_by"].append(from_file)

    def get_related_files(self, file_path: str) -> List[str]:
        """Get files that are directly related to the given file."""
        related = set()
        if file_path in self.nodes:
            # Files this file imports
            for imp in self.nodes[file_path].get("imports", []):
                related.add(imp["target"])
            # Files that import this file
            for importer in self.nodes[file_path].get("imported_by", []):
                related.add(importer)
        return list(related)

    def get_entry_points(self) -> List[str]:
        """Files that are not imported by any other file (potential entry points)."""
        imported_files = set()
        for _, to_file in self.edges:
            imported_files.add(to_file)
        return [f for f in self.nodes if f not in imported_files]

    def get_most_imported(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get the most imported files (highest fan-in)."""
        import_counts: Dict[str, int] = {}
        for _, to_file in self.edges:
            import_counts[to_file] = import_counts.get(to_file, 0) + 1
        sorted_files = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:n]

    def to_dict(self) -> Dict:
        return {
            "file_count": len(self.nodes),
            "dependency_count": len(self.edges),
            "files": {
                path: {
                    "language": meta["language"],
                    "import_count": len(meta["imports"]),
                    "imported_by_count": len(meta["imported_by"]),
                }
                for path, meta in self.nodes.items()
            },
            "entry_points": self.get_entry_points(),
            "most_imported": self.get_most_imported(),
        }


class DependencyAnalyzer:
    """Analyzes a project directory to build a dependency graph."""

    # Language extension map
    LANG_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "cpp",
        ".hpp": "cpp",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
    }

    def __init__(self, project_dir: str):
        self.project_dir = os.path.abspath(project_dir)
        self.graph = DependencyGraph()

    def analyze(self) -> DependencyGraph:
        """Analyze the entire project and build the dependency graph."""
        # Discover source files
        source_files = self._discover_files()

        # Add all files to graph
        for file_path, language in source_files:
            rel_path = os.path.relpath(file_path, self.project_dir)
            self.graph.add_file(rel_path, language)

        # Parse dependencies for each file
        for file_path, language in source_files:
            rel_path = os.path.relpath(file_path, self.project_dir)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                imports = self._extract_imports(content, language, file_path)
                for imp_name, imp_path in imports:
                    if imp_path:
                        imp_rel = os.path.relpath(imp_path, self.project_dir)
                        self.graph.add_dependency(rel_path, imp_rel, imp_name)
            except (IOError, OSError) as e:
                logger.debug(f"Could not read {file_path}: {e}")

        return self.graph

    def _discover_files(self) -> List[Tuple[str, str]]:
        """Discover all source files in the project."""
        files = []
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"}

        for root, dirs, filenames in os.walk(self.project_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.LANG_MAP:
                    file_path = os.path.join(root, filename)
                    files.append((file_path, self.LANG_MAP[ext]))

        return files

    def _extract_imports(
        self, content: str, language: str, file_path: str
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Extract imports and try to resolve them to file paths.
        Returns list of (import_name, resolved_file_path_or_None).
        """
        imports = []

        if language == "python":
            imports = self._extract_python_imports(content, file_path)
        elif language in ("cpp", "c"):
            imports = self._extract_c_imports(content, file_path)
        elif language in ("javascript", "typescript"):
            imports = self._extract_js_imports(content, file_path)
        elif language == "java":
            imports = self._extract_java_imports(content, file_path)

        return imports

    def _extract_python_imports(
        self, content: str, file_path: str
    ) -> List[Tuple[str, Optional[str]]]:
        """Extract Python imports."""
        results = []
        file_dir = os.path.dirname(file_path)

        for line in content.splitlines():
            stripped = line.strip()
            # from X import Y
            match = re.match(r"from\s+([\w.]+)\s+import", stripped)
            if match:
                module = match.group(1)
                resolved = self._resolve_python_module(module, file_dir)
                results.append((module, resolved))
                continue

            # import X
            match = re.match(r"import\s+([\w.]+)", stripped)
            if match:
                module = match.group(1)
                resolved = self._resolve_python_module(module, file_dir)
                results.append((module, resolved))

        return results

    def _resolve_python_module(self, module: str, from_dir: str) -> Optional[str]:
        """Try to resolve a Python module name to a file path."""
        parts = module.split(".")
        # Try relative resolution
        for base_dir in [from_dir, self.project_dir]:
            candidate = os.path.join(base_dir, *parts) + ".py"
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.join(base_dir, *parts, "__init__.py")
            if os.path.exists(candidate):
                return candidate
        return None

    def _extract_c_imports(
        self, content: str, file_path: str
    ) -> List[Tuple[str, Optional[str]]]:
        """Extract C/C++ includes."""
        results = []
        file_dir = os.path.dirname(file_path)

        for line in content.splitlines():
            match = re.match(r'\s*#include\s*"([^"]+)"', line)
            if match:
                header = match.group(1)
                resolved = os.path.join(file_dir, header)
                if os.path.exists(resolved):
                    results.append((header, resolved))
                else:
                    results.append((header, None))

        return results

    def _extract_js_imports(
        self, content: str, file_path: str
    ) -> List[Tuple[str, Optional[str]]]:
        """Extract JavaScript/TypeScript imports."""
        results = []
        file_dir = os.path.dirname(file_path)

        for line in content.splitlines():
            # import ... from './module'
            match = re.search(r"""(?:import|from)\s+.*?['"](\.[\w/.]+)['"]""", line)
            if not match:
                # require('./module')
                match = re.search(r"""require\s*\(\s*['"](\.[\w/.]+)['"]""", line)

            if match:
                module_path = match.group(1)
                # Try resolving
                for ext in ["", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"]:
                    resolved = os.path.normpath(os.path.join(file_dir, module_path + ext))
                    if os.path.exists(resolved):
                        results.append((module_path, resolved))
                        break
                else:
                    results.append((module_path, None))

        return results

    def _extract_java_imports(
        self, content: str, file_path: str
    ) -> List[Tuple[str, Optional[str]]]:
        """Extract Java imports (resolved to file paths when possible)."""
        results = []
        for line in content.splitlines():
            match = re.match(r"import\s+([\w.]+);", line.strip())
            if match:
                import_name = match.group(1)
                # Try to map to a file
                parts = import_name.split(".")
                candidate = os.path.join(self.project_dir, *parts) + ".java"
                resolved = candidate if os.path.exists(candidate) else None
                results.append((import_name, resolved))

        return results
