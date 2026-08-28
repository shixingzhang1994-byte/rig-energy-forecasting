from __future__ import annotations

import json

import torch


info = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
}
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    info.update(
        {
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "memory_gb": round(properties.total_memory / 1024**3, 2),
            "tensor_test": float(torch.randn(512, 512, device="cuda").square().mean()),
        }
    )
print(json.dumps(info, ensure_ascii=False, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("PyTorch 未检测到 CUDA GPU")

