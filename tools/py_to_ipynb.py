"""Turn a lab script into a Jupyter notebook (no nbformat needed).

The script's module docstring becomes the first markdown cell. Each banner block
    # ====
    # PHASE 1 · TITLE
    # ====
    # comment lines...
becomes a markdown heading (plus the comment lines as text) followed by a code cell.

Usage: python tools/py_to_ipynb.py path/to/lab.py   ->  writes path/to/lab.ipynb
"""
import ast
import json
import sys
from pathlib import Path


def cell(kind, text):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def convert(path):
    text = Path(path).read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(text)) or ""
    body = text.split('"""', 2)[2] if text.lstrip().startswith('"""') else text
    cells = [cell("markdown", "# " + doc.strip().replace("\n", "\n\n"))] if doc else []

    blocks = body.split("# " + "=" * 60)
    code_buf = blocks[0]
    i = 1
    while i < len(blocks):
        title = blocks[i].strip().lstrip("# ").strip()
        rest = blocks[i + 1] if i + 1 < len(blocks) else ""
        if code_buf.strip():
            cells.append(cell("code", code_buf))
        lines = rest.split("\n")
        notes = []
        while lines and (lines[0].startswith("#") or not lines[0].strip()):
            l = lines.pop(0)
            if l.strip():
                notes.append(l.lstrip("#")[1:] if l.startswith("# ") else l.lstrip("#"))
        md = "## " + title + ("\n\n" + "\n".join(notes) if notes else "")
        cells.append(cell("markdown", md))
        code_buf = "\n".join(lines)
        i += 2
    if code_buf.strip():
        cells.append(cell("code", code_buf))

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = Path(path).with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    convert(sys.argv[1])
