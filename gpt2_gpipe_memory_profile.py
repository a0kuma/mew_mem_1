"""

***************************************************************************************************************************************************************************************************
how to run
conda activate gpt2_gpipe_mem
python gpt2_gpipe_memory_profile.py
***************************************************************************************************************************************************************************************************


GPT-2 with torchgpipe (GPipe pipeline parallelism) + built-in checkpoint
+ Comprehensive Memory Profiling on ALL GPUs.

torchgpipe features used:
  - torchgpipe.GPipe: pipeline-parallel wrapper with micro-batch splitting
  - checkpoint='always': GPipe's built-in activation checkpointing

Memory reporting methods used:
  1. torch.profiler with profile_memory=True
  2. torch.cuda.memory_allocated(device)
  3. torch.cuda.max_memory_allocated(device)
  4. torch.cuda.max_memory_reserved(device)
  5. torch.cuda.memory._record_memory_history()
  6. torch.cuda.memory._dump_snapshot(path)
"""

import os
import json
import time
import datetime
import torch
import torch.nn as nn
from torchgpipe import GPipe
from transformers import GPT2Config

# ──────────────────────── Configuration ────────────────────────
NUM_GPUS = torch.cuda.device_count()
assert NUM_GPUS >= 2, f"Need at least 2 GPUs, found {NUM_GPUS}"

BATCH_SIZE = 8
MICRO_BATCHES = NUM_GPUS       # chunks for pipeline parallelism
SEQ_LEN = 256
NUM_STEPS = 10
LEARNING_RATE = 3e-4
OUTPUT_DIR = "memory_reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Found {NUM_GPUS} GPUs")
for i in range(NUM_GPUS):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}  |  {props.total_memory / 1024**3:.1f} GB")


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

    # Build a flat nn.Sequential: Embedding, 12x TransformerBlock, LMHead
    # Total modules = 1 (emb) + 12 (transformer) + 1 (head) = 14
    layers = []
    layers.append(EmbeddingBlock(config))                       # module 0
    for i in range(config.n_layer):                             # modules 1..12
        layers.append(TransformerBlock(config))
    layers.append(LMHead(config))                               # module 13

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


# ──────────────────────── Memory Helpers ───────────────────────
def mb(x):
    return x / (1024 ** 2)


def gb(x):
    return x / (1024 ** 3)


def report_memory_all_gpus(step_label: str):
    """Print memory stats for every GPU using torch.cuda API."""
    print(f"\n{'='*80}")
    print(f"  Memory Report: {step_label}")
    print(f"{'='*80}")
    report = {}
    for i in range(NUM_GPUS):
        dev = torch.device(f"cuda:{i}")
        allocated = torch.cuda.memory_allocated(dev)
        max_allocated = torch.cuda.max_memory_allocated(dev)
        reserved = torch.cuda.max_memory_reserved(dev)
        report[f"gpu_{i}"] = {
            "memory_allocated_MB": round(mb(allocated), 2),
            "max_memory_allocated_MB": round(mb(max_allocated), 2),
            "max_memory_reserved_MB": round(mb(reserved), 2),
        }
        print(f"  GPU {i}:")
        print(f"    memory_allocated       = {mb(allocated):10.2f} MB")
        print(f"    max_memory_allocated   = {mb(max_allocated):10.2f} MB")
        print(f"    max_memory_reserved    = {mb(reserved):10.2f} MB")
    print(f"{'='*80}\n")
    return report


def start_memory_history_all_gpus():
    """Start recording memory history on ALL GPUs."""
    print("[INFO] Starting memory history recording on all GPUs...")
    for i in range(NUM_GPUS):
        with torch.cuda.device(i):
            torch.cuda.memory._record_memory_history(
                max_entries=1048576
            )


def dump_memory_snapshot_all_gpus(label: str):
    """Dump memory snapshots for ALL GPUs."""
    snapshot_paths = []
    for i in range(NUM_GPUS):
        path = os.path.join(OUTPUT_DIR, f"memory_snapshot_gpu{i}_{label}.pickle")
        with torch.cuda.device(i):
            torch.cuda.memory._dump_snapshot(path)
        snapshot_paths.append(path)
        print(f"[INFO] Memory snapshot GPU {i} saved: {path}")
    return snapshot_paths


def stop_memory_history_all_gpus():
    """Stop recording memory history on ALL GPUs."""
    for i in range(NUM_GPUS):
        with torch.cuda.device(i):
            torch.cuda.memory._record_memory_history(enabled=None)
    print("[INFO] Stopped memory history recording on all GPUs.")


# ──────────────────────── Training Loop ────────────────────────
def train():
    print("\n" + "=" * 80)
    print("  GPT-2 Pipeline Parallel Training with Memory Profiling")
    print("=" * 80)

    # ── Reset memory stats ──
    for i in range(NUM_GPUS):
        torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()

    # ── Build model ──
    pipe_model, config = build_pipeline(NUM_GPUS)
    pipe_model.train()

    report_memory_all_gpus("After Model Load")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(pipe_model.parameters(), lr=LEARNING_RATE)

    # ── Loss function (on last GPU) ──
    last_device = torch.device(f"cuda:{NUM_GPUS - 1}")

    # ── Start memory recording ──
    start_memory_history_all_gpus()

    # ── Profiler setup ──
    profiler_schedule = torch.profiler.schedule(
        wait=1, warmup=1, active=3, repeat=1
    )

    profile_dir = os.path.join(OUTPUT_DIR, "torch_profiler_trace")
    os.makedirs(profile_dir, exist_ok=True)

    all_step_reports = []

    print(f"\n[INFO] Starting training for {NUM_STEPS} steps...")
    print(f"[INFO] Batch size={BATCH_SIZE}, Seq len={SEQ_LEN}, Micro-batches={MICRO_BATCHES}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,           # <<< KEY: memory profiling
        with_stack=True,
        with_flops=True,
    ) as prof:

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

            # Step profiler
            prof.step()

            dt = time.time() - t0

            print(f"  Step {step:3d}  |  Loss: {loss.item():.4f}  |  Time: {dt:.3f}s")

            # Collect per-step memory
            step_report = report_memory_all_gpus(f"Step {step}")
            step_report["step"] = step
            step_report["loss"] = round(loss.item(), 4)
            step_report["time_s"] = round(dt, 3)
            all_step_reports.append(step_report)

    # ── Print profiler summary (memory) ──
    print("\n" + "=" * 80)
    print("  Profiler Key Averages (sorted by CUDA memory usage)")
    print("=" * 80)
    profiler_table = prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=25
    )
    print(profiler_table)

    # Save profiler table to file
    profiler_table_path = os.path.join(OUTPUT_DIR, "profiler_key_averages.txt")
    with open(profiler_table_path, "w") as f:
        f.write(profiler_table)
    print(f"[INFO] Profiler table saved to {profiler_table_path}")

    # ── Also sort by cuda_time ──
    print("\n" + "=" * 80)
    print("  Profiler Key Averages (sorted by CUDA time)")
    print("=" * 80)
    profiler_table_time = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25
    )
    print(profiler_table_time)

    profiler_table_time_path = os.path.join(OUTPUT_DIR, "profiler_key_averages_by_time.txt")
    with open(profiler_table_time_path, "w") as f:
        f.write(profiler_table_time)

    # ── Final memory report ──
    final_report = report_memory_all_gpus("Final (After Training)")

    # ── Dump memory snapshots ──
    snapshot_paths = dump_memory_snapshot_all_gpus("final")

    # ── Stop memory history ──
    stop_memory_history_all_gpus()

    # ── Save JSON summary ──
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_gpus": NUM_GPUS,
        "batch_size": BATCH_SIZE,
        "micro_batches": MICRO_BATCHES,
        "seq_len": SEQ_LEN,
        "num_steps": NUM_STEPS,
        "model": "GPT-2 (12-layer, 768-hidden, 12-head)",
        "pipeline_parallelism": True,
        "activation_checkpointing": True,
        "gpu_info": {
            f"gpu_{i}": {
                "name": torch.cuda.get_device_properties(i).name,
                "total_memory_GB": round(gb(torch.cuda.get_device_properties(i).total_memory), 2),
            }
            for i in range(NUM_GPUS)
        },
        "final_memory": final_report,
        "per_step_memory": all_step_reports,
        "snapshot_files": snapshot_paths,
        "profiler_trace_dir": profile_dir,
    }

    summary_path = os.path.join(OUTPUT_DIR, "memory_report_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Full memory report saved to {summary_path}")

    # ── Print final summary table ──
    print("\n" + "=" * 80)
    print("  FINAL MEMORY SUMMARY (All GPUs)")
    print("=" * 80)
    print(f"  {'GPU':>4} | {'Allocated (MB)':>15} | {'Peak Alloc (MB)':>16} | {'Peak Reserved (MB)':>18}")
    print(f"  {'-'*4}-+-{'-'*15}-+-{'-'*16}-+-{'-'*18}")
    for i in range(NUM_GPUS):
        dev = torch.device(f"cuda:{i}")
        alloc = mb(torch.cuda.memory_allocated(dev))
        peak_alloc = mb(torch.cuda.max_memory_allocated(dev))
        peak_res = mb(torch.cuda.max_memory_reserved(dev))
        print(f"  {i:>4} | {alloc:>15.2f} | {peak_alloc:>16.2f} | {peak_res:>18.2f}")
    print("=" * 80)

    print(f"\n[INFO] Output files in '{OUTPUT_DIR}/':")
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        f_path = os.path.join(OUTPUT_DIR, f_name)
        if os.path.isfile(f_path):
            size = os.path.getsize(f_path)
            print(f"  {f_name:50s}  ({mb(size):.2f} MB)")
        elif os.path.isdir(f_path):
            print(f"  {f_name}/  (directory)")

    print("\n[INFO] To view profiler traces, run:")
    print(f"  tensorboard --logdir={profile_dir}")
    print("[INFO] To view memory snapshots, use:")
    print("  https://pytorch.org/memory_viz  (upload the .pickle files)")
    print("\n[DONE] All profiling complete.")


if __name__ == "__main__":
    train()
