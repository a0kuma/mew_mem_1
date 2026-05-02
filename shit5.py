import torch
import os

NUM_GPUS = torch.cuda.device_count()
OUTPUT_DIR = "./snapshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────── Memory Helpers ───────────────────────
def start_memory_history_all_gpus():
    print("[INFO] Starting memory history recording on all GPUs...")
    for i in range(NUM_GPUS):
        with torch.cuda.device(i):
            torch.cuda.memory._record_memory_history(max_entries=1048576)


def dump_memory_snapshot_all_gpus(label: str):
    snapshot_paths = []
    for i in range(NUM_GPUS):
        path = os.path.join(OUTPUT_DIR, f"memory_snapshot_gpu{i}_{label}.pickle")
        with torch.cuda.device(i):
            torch.cuda.memory._dump_snapshot(path)
        snapshot_paths.append(path)
        print(f"[INFO] Memory snapshot GPU {i} saved: {path}")
    return snapshot_paths


def stop_memory_history_all_gpus():
    for i in range(NUM_GPUS):
        with torch.cuda.device(i):
            torch.cuda.memory._record_memory_history(enabled=None)
    print("[INFO] Stopped memory history recording on all GPUs.")


# ──────────────────────── GPU Load Test ───────────────────────
def allocate_5gb_and_compute(device):
    if device ==  2 :
        return None, None
    print(f"[INFO] Allocating ~5GB on GPU {device}...")

    # float32 = 4 bytes → 5GB ≈ 1.25 billion elements
    num_elements = int(5 * 1024**3 / 4)

    # Split into chunks to avoid allocation failure
    chunk_size = int(256 * 1024**2 / 4)  # 256MB chunks
    tensors = []

    allocated = 0
    while allocated < num_elements:
        size = min(chunk_size, num_elements - allocated)
        t = torch.randn(size, device=device)
        tensors.append(t)
        allocated += size

    print(f"[INFO] Allocated {allocated * 4 / (1024**3):.2f} GB on GPU {device}")

    # Do some GPU math ops
    print(f"[INFO] Running compute on GPU {device}...")
    result = None
    for i in range(10):
        a = tensors[i % len(tensors)]
        b = tensors[(i + 1) % len(tensors)]

        # random math ops
        c = torch.sin(a) + torch.cos(b)
        d = torch.matmul(c[:1024].view(32, 32), c[:1024].view(32, 32))
        result = d if result is None else result + d

    torch.cuda.synchronize(device)
    print(f"[INFO] Compute finished on GPU {device}")

    return tensors, result


# ──────────────────────── Main ───────────────────────
def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    start_memory_history_all_gpus()

    all_tensors = []

    for i in range(NUM_GPUS):
        tensors, _ = allocate_5gb_and_compute(i)
        all_tensors.append(tensors)

    dump_memory_snapshot_all_gpus("after_compute")

    # Keep tensors alive for a bit
    input("[INFO] Press Enter to free memory...")

    # Free memory
    del all_tensors
    torch.cuda.empty_cache()

    dump_memory_snapshot_all_gpus("after_free")

    stop_memory_history_all_gpus()


if __name__ == "__main__":
    main()