import ast
import argparse
import json
from pathlib import Path

class CodeCartographer(ast.NodeVisitor):
    def __init__(self):
        self.structure = {"classes": [], "functions": [], "imports": []}
        self.current_class = None

    def visit_Import(self, node):
        for alias in node.names:
            self.structure["imports"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            self.structure["imports"].append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "bases": [b.id for b in node.bases if isinstance(b, ast.Name)]
        }
        self.current_class = class_info
        self.structure["classes"].append(class_info)
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        func_info = {
            "name": node.name,
            "args": [a.arg for a in node.args.args],
            "docstring": ast.get_docstring(node),
            "returns": self._get_annotation(node.returns)
        }
        
        if self.current_class:
            self.current_class["methods"].append(func_info)
        else:
            self.structure["functions"].append(func_info)
        
        self.generic_visit(node)

    def _get_annotation(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return "ComplexType" # Simplified for prototype
        return "None"

def analyze_file(file_path: str):
    print(f"🔍 Analyzing {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        
        cartographer = CodeCartographer()
        cartographer.visit(tree)
        
        return cartographer.structure
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test on itself!
    result = analyze_file("scripts/learn_repo_prototype.py")
    print(json.dumps(result, indent=2))
    print("
✅ Knowledge Map Generated. This structure can be injected into Vector Memory.")
