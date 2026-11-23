"""Alvo NCU: inicialização/validação fora da região NVTX PROFILE."""

import argparse
import sys
from pathlib import Path

# No checkout, o kernel vive em src; na imagem Modal ambos ficam em /root.
local_src = Path(__file__).resolve().parent.parent.parent / "src"
if (local_src / "tma_prologue_prototype.py").is_file():
    sys.path.insert(0, str(local_src))

import torch
import triton

from tma_prologue_prototype import TMAPrologueCase, materialize_prologue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--mode", choices=("fused", "matmul", "pointwise"), required=True)
    parser.add_argument("--bm", type=int, default=128)
    parser.add_argument("--bn", type=int, default=128)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--warps", type=int, default=4)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    x = torch.randn((args.size, args.size), device="cuda", dtype=torch.float16)
    scale, weight = torch.randn_like(x), torch.randn_like(x)
    case = TMAPrologueCase(x, scale, weight, args.bm, args.bn, args.bk, args.stages, args.warps)
    fused, split = case.fused(), case.split()
    torch.cuda.synchronize()
    torch.testing.assert_close(fused, split, atol=0, rtol=0)
    if args.mode == "pointwise":
        def runner():
            return materialize_prologue[(triton.cdiv(x.numel(), 256),)](
                x, scale, case.q, SIZE=x.numel(), BLOCK=256, enable_fp_fusion=False,
            )
    else:
        runner = case.fused if args.mode == "fused" else case.split
    for _ in range(5):
        runner()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("PROFILE")
    runner()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
