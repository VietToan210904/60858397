"""Synthetic CUDA numerical check only; not 3D model inference or a benchmark."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from texture_baker import TextureBaker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["spar3d", "sf3d"], required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no synthetic GPU check performed")
    uv = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=torch.float32, device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    attributes = torch.tensor([[0.2, 0.4, 0.6]] * 3, dtype=torch.float32, device="cuda")
    baker = TextureBaker()
    raster = baker.rasterize(uv, faces, 16)
    mask = baker.get_mask(raster)
    interpolated = baker.interpolate(attributes, raster, faces)
    torch.cuda.synchronize()
    assert raster.shape == (16, 16, 4) and interpolated.shape == (16, 16, 3)
    assert 0 < int(mask.sum()) < 256
    assert torch.isfinite(interpolated).all()
    assert torch.allclose(interpolated[mask], attributes[0].expand_as(interpolated[mask]), atol=1e-5)
    assert torch.allclose(raster[mask][:, :3].sum(dim=-1), torch.ones_like(raster[mask][:, 0]), atol=1e-5)
    receipt = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "model_environment": args.model,
               "purpose": "synthetic native CUDA extension numerical test only", "synthetic": True,
               "model_inference_performed": False, "benchmark_record": False, "passed": True,
               "torch": torch.__version__, "compiled_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
               "resolution": 16, "occupied_pixels": int(mask.sum()),
               "assertions": ["nonempty proper occupancy", "finite RGB interpolation", "constant attributes preserved", "barycentric weights sum to one"]}
    path = Path(__file__).resolve().parents[1] / "environment" / f"synthetic-native-cuda-{args.model}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
