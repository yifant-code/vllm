from importlib.machinery import SourceFileLoader
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def test_demo_local_module_message():
    repo_root = Path(__file__).resolve().parents[2]
    demo_path = repo_root / "vllm" / "demo_local_module.py"
    loader = SourceFileLoader("vllm.demo_local_module", str(demo_path))
    spec = spec_from_file_location("vllm.demo_local_module", str(demo_path), loader=loader)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "demo_message")
    assert mod.demo_message() == "hello from demo_local_module"
