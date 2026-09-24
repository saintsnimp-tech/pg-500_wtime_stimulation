from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main() -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description="Execute and save a project notebook with nbclient")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--kernel", default="cqfenv")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    project_root = notebook_path.parent.parent
    notebook = nbformat.read(notebook_path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="visa-wait-ipython-") as runtime_directory:
        os.environ["IPYTHONDIR"] = runtime_directory
        os.environ["JUPYTER_RUNTIME_DIR"] = str(Path(runtime_directory) / "runtime")
        client = NotebookClient(
            notebook,
            kernel_name=args.kernel,
            timeout=args.timeout,
            resources={"metadata": {"path": str(project_root)}},
        )
        client.execute(cwd=str(project_root))
    nbformat.write(notebook, notebook_path)
    print(f"Executed notebook saved to {notebook_path}")
    if args.html_output:
        html_output = args.html_output.resolve()
        html_output.parent.mkdir(parents=True, exist_ok=True)
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(notebook)
        html_output.write_text(body, encoding="utf-8")
        print(f"Notebook HTML preview saved to {html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
