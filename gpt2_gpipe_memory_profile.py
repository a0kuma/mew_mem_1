"""

***************************************************************************************************************************************************************************************************
how to run
conda activate gpt2_gpipe_mem
python gpt2_gpipe_memory_profile.py
CUBLAS_WORKSPACE_CONFIG=:0:0
***************************************************************************************************************************************************************************************************

"""

from importlib.resources import path
import os
import json
import time
import datetime
import re
import sys
import torch
import pyfiglet
import pickle
import wandb
import subprocess
import torch.nn as nn
from torchgpipe import GPipe
from transformers import GPT2Config
from rich.pretty import pprint
from rich.console import Console
from rich.pretty import Pretty
from rich.text import Text
from rich.cells import cell_len
from rich.style import Style


console = Console()

# ──────────────────────── Configuration ────────────────────────
NUM_GPUS = 4
BATCH_SIZE = 8
MICRO_BATCHES = NUM_GPUS       # chunks for pipeline parallelism
SEQ_LEN = 256
NUM_STEPS = 5
LEARNING_RATE = 3e-4
OUTPUT_FILE_PICKLE = f"memory_reports_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pickle"


# ──────────────────────── GPT-2 Pipeline Layers ────────────────
class EmbeddingBlock(nn.Module):
    """Token + Position embeddings (runs on first GPU)."""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.n_positions = config.n_positions

    def forward(self, input_ids):
        # input_ids: (B, S)
        device = input_ids.device
        seq_len = input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        hidden = self.wte(input_ids) + self.wpe(position_ids)
        hidden = self.drop(hidden)
        return hidden


class TransformerBlock(nn.Module):
    """Single GPT-2 transformer layer (used as one unit in the GPipe balance)."""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Attention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class GPT2MLP(nn.Module):
    """GPT-2 MLP (feed-forward)."""
    def __init__(self, config: GPT2Config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, inner_dim)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(inner_dim, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LMHead(nn.Module):
    """Final LayerNorm + linear head (runs on last GPU)."""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# ──────────────────────── Build Pipeline ───────────────────────
def build_pipeline(num_gpus: int):
    """
    Build a GPT-2 model split across `num_gpus` GPUs using torchgpipe.GPipe.
    GPipe handles device placement, micro-batch splitting, and checkpointing.
    """
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
    )

    print("*****************************************************************")
    print(config)
    print("*****************************************************************")

    # Build a flat nn.Sequential: Embedding, 12x TransformerBlock, LMHead
    # Total modules = 1 (emb) + 12 (transformer) + 1 (head) = 14
    layers = []
    layers.append(EmbeddingBlock(config))                       # module 0
    for i in range(config.n_layer):                             # modules 1..12
        layers.append(TransformerBlock(config))
    layers.append(LMHead(config))                               # module 13

    print("*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")
    pprint(layers)
    print("*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")
    model = nn.Sequential(*layers)

    # Balance: distribute 14 modules across num_gpus
    # e.g. 4 GPUs -> [4, 3, 3, 4] or similar
    total_modules = len(layers)  # 14
    base = total_modules // num_gpus
    remainder = total_modules % num_gpus
    balance = []
    for i in range(num_gpus):
        balance.append(base + (1 if i < remainder else 0))

    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

    print(f"[INFO] Model has {total_modules} sequential modules")
    print(f"[INFO] Balance across {num_gpus} GPUs: {balance}")
    print(f"[INFO] Devices: {devices}")

    # Create GPipe pipeline
    # checkpoint='always' -> GPipe's built-in activation checkpointing on all micro-batches
    pipe_model = GPipe(
        model,
        balance=balance,
        devices=devices,
        chunks=MICRO_BATCHES,
        checkpoint='always',
    )

    print(f"[INFO] GPipe pipeline built with {num_gpus} partitions, {MICRO_BATCHES} chunks")
    print(f"[INFO] Activation checkpointing: 'always' (torchgpipe built-in)")

    return pipe_model, config



# ──────────────────────── Training Loop ────────────────────────
def train():
    # ── Build model ──
    pipe_model, config = build_pipeline(NUM_GPUS)
    pipe_model.train()



    # ── Optimizer ──
    optimizer = torch.optim.AdamW(pipe_model.parameters(), lr=LEARNING_RATE)

    # ── Loss function (on last GPU) ──
    last_device = torch.device(f"cuda:{NUM_GPUS - 1}")

    print(f"\n[INFO] Starting training for {NUM_STEPS} steps...")
    print(f"[INFO] Batch size={BATCH_SIZE}, Seq len={SEQ_LEN}, Micro-batches={MICRO_BATCHES}")

    for step in range(NUM_STEPS):
        t0 = time.time()

        # Create synthetic input on first GPU
        input_ids = torch.randint(
            0, config.vocab_size, (BATCH_SIZE, SEQ_LEN),
            device=torch.device("cuda:0")
        )
        labels = input_ids.clone().to(last_device)

        # ── Forward pass through GPipe pipeline ──
        logits = pipe_model(input_ids)

        # ── Compute loss on last device ──
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1)
        )

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.time() - t0

        print(f"  Step {step:3d}  |  Loss: {loss.item():.4f}  |  Time: {dt:.3f}s")


    print("[INFO] To view memory snapshots, use:")
    print("  https://pytorch.org/memory_viz  (upload the .pickle files)")
    print("\n[DONE] All profiling complete.")


if __name__ == "__main__":
    
    console.print(Text(pyfiglet.figlet_format("start-main", font="slant"), style="bold cyan"))
    torch.cuda.memory._record_memory_history()
    train()
    torch.cuda.memory._dump_snapshot(OUTPUT_FILE_PICKLE)

    #===========================以下的東東與主邏輯無關===========================

    with wandb.init(project="pytorch-memory", save_code=True) as run:

        #===========================前置動作===========================

        OUTPUT_FILE_JSON = f"memory_reports_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"

        PEAK_ACTIVE_RE = re.compile(
            r"Peak active memory:\s+([0-9.]+[A-Za-z]+B)\s+\((\d+)\s+bytes\)"
            r"(?:\s+at\s+([0-9T:\.\-Z]+)\s+\(([^)]+)\))?",
        )
        BLOCKS_AT_PEAK_RE = re.compile(r"Blocks at peak memory.*?:\s*(\d+)")


        def parse_peak_metrics(stdout_text: str):
            if not stdout_text:
                return {}

            peak_entries = []
            blocks_entries = []
            for line in stdout_text.splitlines():
                peak_match = PEAK_ACTIVE_RE.search(line)
                if peak_match:
                    peak_entries.append({
                        "human": peak_match.group(1).strip(),
                        "bytes": int(peak_match.group(2)),
                        "iso": peak_match.group(3),
                        "local": peak_match.group(4),
                    })

                blocks_match = BLOCKS_AT_PEAK_RE.search(line)
                if blocks_match:
                    blocks_entries.append(int(blocks_match.group(1)))

            metrics = {}
            if peak_entries:
                last_peak = peak_entries[-1]
                metrics["peak_active_memory_human"] = last_peak["human"]
                metrics["peak_active_memory_bytes"] = last_peak["bytes"]
                if last_peak["iso"]:
                    metrics["peak_active_memory_time_iso"] = last_peak["iso"]
                if last_peak["local"]:
                    metrics["peak_active_memory_time_local"] = last_peak["local"]

            if blocks_entries:
                metrics["blocks_at_peak_memory"] = blocks_entries[-1]

            if len(peak_entries) > 1 or len(blocks_entries) > 1:
                print(
                    f"[WARN] Multiple peak logs found: peaks={len(peak_entries)}, "
                    f"blocks={len(blocks_entries)}. Using last values."
                )

            return metrics
        #^^^===========================前置動作===========================^^^


        final_pickle_path = os.path.abspath(OUTPUT_FILE_PICKLE)
        final_max_json_path = os.path.abspath(OUTPUT_FILE_JSON)
        result = subprocess.run(
            ["node", "pickle_to_json.mjs", "--input", final_pickle_path, "--output", final_max_json_path],
            cwd="pytorchMemoryVizAuto/autoScript",
            text=True,
            capture_output=True,
            check=True,
        )
        print(result)
        log_payload = {
            "CUBLAS_WORKSPACE_CONFIG":os.getenv("CUBLAS_WORKSPACE_CONFIG"),
            "PYTORCH_NO_CUDA_MEMORY_CACHING":os.getenv("PYTORCH_NO_CUDA_MEMORY_CACHING"),
            "pickle_to_json_returncode": result.returncode,
            "pickle_to_json_stdout": result.stdout,
            "pickle_to_json_stderr": result.stderr,
        }
        log_payload.update(parse_peak_metrics(result.stdout))
        run.log(log_payload)
        artifact = wandb.Artifact("final_pickle_and_max_json", type="dataset")
        artifact.add_file(final_pickle_path)
        artifact.add_file(final_max_json_path)
        #===========================pdf===========================
        latex_report_path = os.path.abspath(
            os.path.join(
                os.path.dirname(final_max_json_path),
                f"{os.path.splitext(os.path.basename(final_max_json_path))[0]}_peak_alloc_events_report.tex",
            )
        )
        latex_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "generate_peak_alloc_latex.py",
        )
        latex_result = subprocess.run(
            [
                sys.executable,
                latex_script_path,
                "--input",
                final_max_json_path,
                "--output",
                latex_report_path,
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        print(latex_result.stdout)
        print(latex_result.stderr)
        #^^^===========================pdf===========================^^^
        run.log({
            "generate_peak_alloc_latex_returncode": latex_result.returncode,
            "generate_peak_alloc_latex_stdout": latex_result.stdout,
            "generate_peak_alloc_latex_stderr": latex_result.stderr,
            "peak_alloc_latex_path": latex_report_path,
        })
        artifact.add_file(latex_report_path)
        run.log_artifact(artifact)

        os.remove(OUTPUT_FILE_JSON)
        os.remove(OUTPUT_FILE_PICKLE)