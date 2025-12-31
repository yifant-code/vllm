"""Demo runner that can run a GPU-backed vLLM demo with safe multiprocessing.

This script prefers a guarded `spawn` start method for multiprocessing to
avoid CUDA reinitialization errors in forked subprocesses. It attempts to run
the `vllm` high-level `LLM` demo on CUDA; if that fails, it falls back to
loading `vllm/demo_local_module.py` directly (useful for iterating on local
demo code without importing the full package).
"""

import multiprocessing as mp
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys


def run_vllm_gpu_demo():
    try:
        # Set spawn start method when running as a script
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set
        pass

    try:
        from vllm.entrypoints.llm import LLM
        from vllm.sampling_params import SamplingParams
        import torch

        print("torch.cuda.is_available =", torch.cuda.is_available())
        print("torch.cuda.device_count =", torch.cuda.device_count())

        print("Instantiating LLM (gpt2)")
        llm = LLM("gpt2", gpu_memory_utilization=0.5, kv_cache_memory_bytes=None)
        print("LLM instantiated")

        params = SamplingParams(max_tokens=20)
        print("Generating...")
        outputs = llm.generate("Hello from vllm GPU demo", sampling_params=params)
        print("Generate returned, outputs type:", type(outputs))
        try:
            text = outputs[0].outputs[0].text
            print("Generated text:\n", text)
        except Exception as e:
            print("Could not extract text from outputs:", e)
        return True
    except Exception as e:
        print("vllm GPU demo failed:", e)
        return False


def run_local_demo_loader():
    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "vllm" / "demo_local_module.py"
    if not demo_path.exists():
        print(f"Demo module not found at: {demo_path}")
        return

    loader = SourceFileLoader("vllm.demo_local_module", str(demo_path))
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location("vllm.demo_local_module", str(demo_path), loader=loader)
    if spec is None or spec.loader is None:
        print(f"Could not create spec/loader for demo module: {demo_path}")
        return
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    msg = getattr(mod, "demo_message", lambda: "(no demo_message)")()
    print(msg)


def main():
    # Prefer the vllm GPU demo; fall back to local demo loader
    ok = run_vllm_gpu_demo()
    if not ok:
        print("Falling back to local demo loader")
        run_local_demo_loader()


if __name__ == "__main__":
    # Ensure downstream code that uses spawn can detect our preference
    mp.set_start_method("spawn", force=False)
    # Also export env var used by vLLM internals when relevant
    import os

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
