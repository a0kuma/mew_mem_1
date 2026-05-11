#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional
from rich.pretty import pprint
from rich.console import Console
from rich.pretty import Pretty
from rich.text import Text
from rich.cells import cell_len
from rich.style import Style
import pyfiglet
console = Console()

CATEGORY_NAMES = {
    "A": "Model Parameters",
    "B": "Optimizer States (AdamW)",
    "C": "cuBLAS Workspace",
    "D": "Pipeline Boundary Copies",
    "E": "Gradient Boundary Copies",
    "F": "Parameter Gradients",
    "G": "Recomputed Activations",
    "H": "Final Backward Tensors",
    "I": "OTHERS",
}

LATEX_SPECIALS = {
    "\\": r"\\textbackslash{}",
    "&": r"\\&",
    "%": r"\\%",
    "$": r"\\$",
    "#": r"\\#",
    "_": r"\\_",
    "{": r"\\{",
    "}": r"\\}",
    "~": r"\\textasciitilde{}",
    "^": r"\\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in text)


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of allocation events in JSON.")
    return data


def read_file_lines(path: str, cache: Dict[str, Optional[List[str]]]) -> Optional[List[str]]:
    if path in cache:
        return cache[path]
    try:
        with open(path, "r", encoding="utf-8") as f:
            cache[path] = f.readlines()
    except OSError:
        cache[path] = None
    return cache[path]


def get_code_line(path: str, line_no: int, cache: Dict[str, Optional[List[str]]]) -> str:
    if not path or line_no <= 0:
        return ""
    lines = read_file_lines(path, cache)
    if not lines or line_no > len(lines):
        return ""
    return lines[line_no - 1].rstrip("\n")


def is_candidate_frame(frame: dict) -> bool:
    filename = frame.get("filename") or ""
    line = frame.get("line") or 0
    if not filename or filename == "??" or line <= 0:
        return False
    if not filename.endswith(".py"):
        return False
    return True


def score_frame(filename: str, workspace_root: str) -> int:
    score = 0
    if filename.startswith(workspace_root):
        score += 100
    if "torchgpipe" in filename:
        score += 80
    elif "/torch/optim/" in filename:
        score += 70
    elif "/torch/nn/" in filename:
        score += 60
    elif "/torch/autograd/" in filename:
        score += 50
    else:
        score += 10
    return score


def select_frame(frames: List[dict], workspace_root: str) -> Optional[dict]:
    best = None
    best_score = -1
    for frame in frames:
        if not is_candidate_frame(frame):
            continue
        filename = frame.get("filename") or ""
        s = score_frame(filename, workspace_root)
        if s > best_score:
            best = frame
            best_score = s
    if best is not None:
        return best
    for frame in frames:
        filename = frame.get("filename") or ""
        line = frame.get("line") or 0
        if filename and filename != "??" and line > 0:
            return frame
    return None


def classify_event(
    frames: List[dict],
    selected: Optional[dict],
    code_line: str,
    workspace_root: str,
) -> str:
    files = [f.get("filename") or "" for f in frames]
    names = [f.get("name") or "" for f in frames]
    code_low = (code_line or "").lower()

    def has_file(substr: str) -> bool:
        return any(substr in f for f in files)

    def has_name(substr: str) -> bool:
        return any(substr in n for n in names)

    def has_name_lower(substr: str) -> bool:
        return any(substr in n.lower() for n in names)

    checkpoint_lines = [
        int(f.get("line") or 0)
        for f in frames
        if "torchgpipe/checkpoint.py" in (f.get("filename") or "")
    ]
    has_checkpoint_backward = any(258 <= ln <= 273 for ln in checkpoint_lines)
    has_checkpoint_recompute = any(295 <= ln <= 308 for ln in checkpoint_lines)

    if has_name_lower("cublas") or any("cublas" in f.lower() for f in files) or any("cudnn" in f.lower() for f in files):
        return "C"

    if has_file("torchgpipe/copy.py"):
        for f in frames:
            if "torchgpipe/copy.py" in (f.get("filename") or "") and "backward" in (f.get("name") or ""):
                return "E"
        return "D"

    if has_checkpoint_backward:
        return "H"

    if has_checkpoint_recompute:
        return "G"

    if has_file("torchgpipe/checkpoint.py") or "checkpoint" in code_low or "recompute" in code_low:
        return "G"

    if has_file("/torch/optim/adam.py") or has_file("/torch/optim/optimizer.py"):
        return "B"

    selected_file = (selected or {}).get("filename") or ""
    selected_name = (selected or {}).get("name") or ""
    selected_line = int((selected or {}).get("line") or 0)

    if (
        "partition.to" in code_low
        or "gpipe(" in code_low
        or "nn.sequential" in code_low
        or "nn.linear" in code_low
        or "nn.embedding" in code_low
        or "nn.layernorm" in code_low
        or ("torchgpipe/gpipe.py" in selected_file and "split_module" in selected_name)
        or (selected_file.startswith(workspace_root) and 140 <= selected_line <= 210)
    ):
        return "A"

    if ".grad" in code_low or "param_grad" in code_low or "grad_applied" in code_low:
        return "F"

    if (
        has_file("/torch/autograd/graph.py")
        or has_file("/torch/autograd/__init__.py")
        or has_name("backward")
        or any("backward" in n.lower() for n in names)
        or any("Backward" in n for n in names)
    ):
        return "H"

    return "I"


def render_latex(events: List[dict], workspace_root: str) -> str:
    cache: Dict[str, Optional[List[str]]] = {}
    rows = []

    for idx, event in enumerate(events, start=1):
        frames = event.get("frames") or []
        selected = select_frame(frames, workspace_root)
        filename = (selected or {}).get("filename") or ""
        line_no = int((selected or {}).get("line") or 0)
        code_line = get_code_line(filename, line_no, cache)
        category = classify_event(frames, selected, code_line, workspace_root)

        code_out = code_line if code_line else "(source not available)"
        file_out = filename if filename else "(unknown)"
        size = event.get("size", "")

        rows.append(
            (
                idx,
                category,
                size,
                file_out,
                line_no if line_no > 0 else "",
                code_out,
            )
        )

    lines = []
    lines.append("\\documentclass{article}")
    lines.append("\\usepackage[margin=1in]{geometry}")
    lines.append("\\usepackage{longtable}")
    lines.append("\\usepackage[T1]{fontenc}")
    lines.append("\\usepackage{textcomp}")
    lines.append("\\usepackage{array}")
    lines.append("\\begin{document}")
    lines.append("\\section*{Peak Allocation Events}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll}")
    for key in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        lines.append(f"{key} & {latex_escape(CATEGORY_NAMES[key])} \\")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.5em}")
    lines.append("\\begin{longtable}{r l r p{6.5cm} r p{7.0cm}}")
    lines.append("\\hline")
    lines.append("Idx & Cat & Size(B) & File & Line & Code \\")
    lines.append("\\hline")
    lines.append("\\endfirsthead")
    lines.append("\\hline")
    lines.append("Idx & Cat & Size(B) & File & Line & Code \\")
    lines.append("\\hline")
    lines.append("\\endhead")
    lines.append("\\hline")
    lines.append("\\endfoot")

    for idx, category, size, file_out, line_no, code_out in rows:
        file_tex = r"\\texttt{" + latex_escape(str(file_out)) + "}"
        code_tex = r"\\texttt{" + latex_escape(str(code_out)) + "}"
        size_tex = latex_escape(str(size))
        line_tex = latex_escape(str(line_no))
        lines.append(f"{idx} & {category} & {size_tex} & {file_tex} & {line_tex} & {code_tex} \\")

    lines.append("\\end{longtable}")
    lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX report for peak allocation events.")
    parser.add_argument("--input", default="peak_alloc_events.json", help="Input JSON path")
    parser.add_argument("--output", default="peak_alloc_events_report.tex", help="Output LaTeX path")
    args = parser.parse_args()

    events = load_json(args.input)
    workspace_root = os.path.abspath(os.path.dirname(args.input))
    latex = render_latex(events, workspace_root)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Wrote LaTeX report to {args.output}")


if __name__ == "__main__":
    main()
