#!/usr/bin/env python3
#python generate_peak_alloc_latex.py --input peak_alloc_events.json --output peak_alloc_events_report.tex
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
    "D": "Pipeline Boundary Copies",
    "E": "Gradient Boundary Copies",
    "F": "Parameter Gradients",
    "G": "Recomputed Activations",
    "H": "Final Backward Tensors",
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

LST_INLINE_DELIMS = ["|", "!", "+", ";", ":", "?", "=", "@", "~"]
FINAL_BACKWARD_COUNT = 4


def ascii_sanitize(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("ascii", errors="replace").decode("ascii")


def latex_escape(text: str) -> str:
    text = ascii_sanitize(text)
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in text)


def latex_code_inline(text: str) -> str:
    text = ascii_sanitize(text)
    if text == "":
        return ""
    if "\n" in text:
        text = text.replace("\n", " ")
    for delim in LST_INLINE_DELIMS:
        if delim not in text:
            return f"\\lstinline{delim}{text}{delim}"
    return r"\\texttt{" + latex_escape(text) + "}"


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


def get_checkpoint_flags(frames: List[dict]) -> tuple[bool, bool]:
    checkpoint_lines = [
        int(f.get("line") or 0)
        for f in frames
        if "torchgpipe/checkpoint.py" in (f.get("filename") or "")
    ]
    has_checkpoint_backward = any(258 <= ln <= 273 for ln in checkpoint_lines)
    has_checkpoint_recompute = any(295 <= ln <= 308 for ln in checkpoint_lines)
    return has_checkpoint_backward, has_checkpoint_recompute


def is_backward_event(frames: List[dict], has_checkpoint_backward: bool | None = None) -> bool:
    if has_checkpoint_backward is None:
        has_checkpoint_backward, _ = get_checkpoint_flags(frames)
    names = [f.get("name") or "" for f in frames]
    files = [f.get("filename") or "" for f in frames]
    return (
        has_checkpoint_backward
        or any("/torch/autograd/graph.py" in f for f in files)
        or any("/torch/autograd/__init__.py" in f for f in files)
        or any("backward" in n.lower() for n in names)
        or any("Backward" in n for n in names)
    )


def is_model_param_event(
    frames: List[dict],
    selected: Optional[dict],
    code_line: str,
    workspace_root: str,
) -> bool:
    code_low = (code_line or "").lower()
    selected_file = (selected or {}).get("filename") or ""
    selected_name = (selected or {}).get("name") or ""
    selected_line = int((selected or {}).get("line") or 0)

    return (
        "partition.to" in code_low
        or "gpipe(" in code_low
        or "nn.sequential" in code_low
        or "nn.linear" in code_low
        or "nn.embedding" in code_low
        or "nn.layernorm" in code_low
        or ("torchgpipe/gpipe.py" in selected_file and "split_module" in selected_name)
        or (selected_file.startswith(workspace_root) and 140 <= selected_line <= 210)
    )


def classify_event(
    frames: List[dict],
    selected: Optional[dict],
    code_line: str,
    size: Optional[int],
    event_index: int,
    workspace_root: str,
    param_sizes: Optional[set],
    final_backward_indices: set,
) -> str:
    files = [f.get("filename") or "" for f in frames]
    names = [f.get("name") or "" for f in frames]
    code_low = (code_line or "").lower()

    def has_file(substr: str) -> bool:
        return any(substr in f for f in files)

    def has_name(substr: str) -> bool:
        return any(substr in n for n in names)

    has_checkpoint_backward, has_checkpoint_recompute = get_checkpoint_flags(frames)
    is_backward = is_backward_event(frames, has_checkpoint_backward)

    if has_file("torchgpipe/copy.py"):
        for f in frames:
            if "torchgpipe/copy.py" in (f.get("filename") or "") and "backward" in (f.get("name") or ""):
                return "E"
        return "D"

    if event_index in final_backward_indices:
        return "H"

    if is_backward and param_sizes and size is not None and size in param_sizes:
        return "F"

    if has_checkpoint_recompute:
        return "G"

    if has_file("torchgpipe/checkpoint.py") or "checkpoint" in code_low or "recompute" in code_low:
        return "G"

    if has_file("/torch/optim/adam.py") or has_file("/torch/optim/optimizer.py"):
        return "B"

    if is_model_param_event(frames, selected, code_line, workspace_root):
        return "A"

    if is_backward:
        return "G"

    return "G"


def render_latex(events: List[dict], workspace_root: str) -> str:
    cache: Dict[str, Optional[List[str]]] = {}
    rows = []
    home_prefix = os.path.expanduser("~")
    param_sizes: set = set()
    summary = {key: {"count": 0, "bytes": 0} for key in ["A", "B", "D", "E", "F", "G", "H"]}
    backward_events: list[tuple[int, int]] = []

    for idx, event in enumerate(events, start=1):
        frames = event.get("frames") or []
        has_checkpoint_backward, _ = get_checkpoint_flags(frames)
        if is_backward_event(frames, has_checkpoint_backward):
            time_us = event.get("time_us")
            backward_events.append((idx, int(time_us) if isinstance(time_us, int) else 0))
        selected = select_frame(frames, workspace_root)
        filename = (selected or {}).get("filename") or ""
        line_no = int((selected or {}).get("line") or 0)
        code_line = get_code_line(filename, line_no, cache)
        if is_model_param_event(frames, selected, code_line, workspace_root):
            size = event.get("size")
            if isinstance(size, int):
                param_sizes.add(size)

    backward_events.sort(key=lambda item: item[1])
    final_backward_indices = {idx for idx, _ in backward_events[-FINAL_BACKWARD_COUNT:]}

    for idx, event in enumerate(events, start=1):
        frames = event.get("frames") or []
        selected = select_frame(frames, workspace_root)
        filename = (selected or {}).get("filename") or ""
        line_no = int((selected or {}).get("line") or 0)
        code_line = get_code_line(filename, line_no, cache)
        size = event.get("size")
        size_value = size if isinstance(size, int) else None
        category = classify_event(
            frames,
            selected,
            code_line,
            size_value,
            idx,
            workspace_root,
            param_sizes,
            final_backward_indices,
        )

        code_out = code_line if code_line else "(source not available)"
        if filename and home_prefix and filename.startswith(home_prefix):
            file_out = "~" + filename[len(home_prefix):]
        else:
            file_out = filename if filename else "(unknown)"
        size = event.get("size", "")
        if category in summary:
            summary[category]["count"] += 1
            if isinstance(size, int):
                summary[category]["bytes"] += size

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
    lines.append("\\usepackage{listings}")
    lines.append("\\usepackage[T1]{fontenc}")
    lines.append("\\usepackage{textcomp}")
    lines.append("\\usepackage{array}")
    lines.append("\\lstset{basicstyle=\\ttfamily\\footnotesize,breaklines=true,breakatwhitespace=true}")
    lines.append("\\begin{document}")
    lines.append("\\section*{Peak Allocation Events}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll}")
    for key in ["A", "B", "D", "E", "F", "G", "H"]:
        lines.append(f"{key} & {latex_escape(CATEGORY_NAMES[key])} \\\\")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.5em}")
    lines.append("\\section*{Category Summary}")
    lines.append("\\begin{tabular}{l r r}")
    lines.append("Category & Count & Total(B) \\\\")
    lines.append("\\hline")
    for key in ["A", "B", "D", "E", "F", "G", "H"]:
        count = summary[key]["count"]
        total_bytes = summary[key]["bytes"]
        lines.append(f"{key} & {count} & {total_bytes} \\\\")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.5em}")
    lines.append("\\begin{longtable}{r l r p{6.5cm} r p{7.0cm}}")
    lines.append("\\hline")
    lines.append("Idx & Cat & Size(B) & File & Line & Code \\\\")
    lines.append("\\hline")
    lines.append("\\endfirsthead")
    lines.append("\\hline")
    lines.append("Idx & Cat & Size(B) & File & Line & Code \\\\")
    lines.append("\\hline")
    lines.append("\\endhead")
    lines.append("\\hline")
    lines.append("\\endfoot")

    for idx, category, size, file_out, line_no, code_out in rows:
        file_tex = latex_code_inline(str(file_out))
        code_tex = latex_code_inline(str(code_out))
        size_tex = latex_escape(str(size))
        line_tex = latex_escape(str(line_no))
        lines.append(f"{idx} & {category} & {size_tex} & {file_tex} & {line_tex} & {code_tex} \\\\")

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
