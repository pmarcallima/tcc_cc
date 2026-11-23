"""Controle manual de fusão: mesmo laço persistente, cargas comuns ou TMA.

Não representa o scheduler do Inductor. O compilador pode produzir instruções
diferentes; a comparação entre famílias inclui os respectivos lowerings.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def prologue(X, S, Q, SIZE: tl.constexpr, CAST: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, other=0).to(tl.float32)
    if not CAST:
        s = tl.load(S + i, i < SIZE, other=1).to(tl.float32)
        x = x * s + (s - 1.)
    tl.store(Q + i, x.to(tl.float16), i < SIZE)


@triton.jit
def projection(X, S, W, XD, SD, WD, Y,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               SMS: tl.constexpr, TMA: tl.constexpr, FUSED: tl.constexpr,
               CAST: tl.constexpr):
    nt = tl.cdiv(N, BN)
    for tile in tl.range(tl.program_id(0), tl.cdiv(M, BM) * nt, SMS):
        rm = (tile // nt) * BM
        rn = (tile % nt) * BN
        mi = rm + tl.arange(0, BM)
        ni = rn + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for rk in range(tl.cdiv(K, BK)):
            ki = rk * BK + tl.arange(0, BK)
            if TMA:
                a = XD.load([rm, rk * BK])
                b = WD.load([rk * BK, rn])
            else:
                a = tl.load(X + mi[:, None] * K + ki[None, :],
                            (mi[:, None] < M) & (ki[None, :] < K), other=0)
                b = tl.load(W + ki[:, None] * N + ni[None, :],
                            (ki[:, None] < K) & (ni[None, :] < N), other=0)
            if FUSED:
                a = a.to(tl.float32)
                if not CAST:
                    if TMA:
                        s = SD.load([rm, rk * BK]).to(tl.float32)
                    else:
                        s = tl.load(S + mi[:, None] * K + ki[None, :],
                                    (mi[:, None] < M) & (ki[None, :] < K),
                                    other=1).to(tl.float32)
                    a = a * s + (s - 1.)
                a = a.to(tl.float16)
            acc = tl.dot(a, b, acc)
        tl.store(Y + mi[:, None] * N + ni[None, :], acc.to(tl.float16),
                 (mi[:, None] < M) & (ni[None, :] < N))


class FourPathCase:
    """Buffers/descritores pré-alocados; mesma geometria nos quatro caminhos."""
    def __init__(self, x, scale, weight, cast, config):
        self.x, self.scale, self.weight = x, scale, weight
        self.m, self.k = x.shape
        self.n = weight.shape[1]
        self.cast = cast
        assert scale.shape == x.shape and weight.shape[0] == self.k
        assert x.dtype == (torch.int8 if cast else torch.float16)
        assert scale.dtype == weight.dtype == torch.float16
        assert all(t.is_contiguous() for t in (x, scale, weight))
        bm, bn, bk = (config[key] for key in ("bm", "bn", "bk"))
        assert self.m % bm == self.n % bn == self.k % bk == 0
        self.q = torch.empty(x.shape, device=x.device, dtype=torch.float16)
        self.desc = {
            "x": TensorDescriptor.from_tensor(x, [bm, bk]),
            "s": TensorDescriptor.from_tensor(scale, [bm, bk]),
            "q": TensorDescriptor.from_tensor(self.q, [bm, bk]),
            "w": TensorDescriptor.from_tensor(weight, [bk, bn]),
        }
        self.outputs = {p: torch.empty((self.m, self.n), device=x.device,
                                      dtype=torch.float16) for p in PATHS}
        self.tiles = triton.cdiv(self.m, bm) * triton.cdiv(self.n, bn)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        self.grid = (min(sms, self.tiles),)
        self.meta = dict(M=self.m, N=self.n, K=self.k, BM=bm, BN=bn, BK=bk,
                         SMS=sms, CAST=cast, num_warps=config["warps"],
                         num_stages=config["stages"], enable_fp_fusion=False)
        self.compiled = {}

    def run(self, path):
        tma, fused = path.startswith("tma"), path.endswith("fused")
        if not fused:
            self.compiled["pointwise"] = prologue[(triton.cdiv(self.m*self.k, 256),)](
                self.x, self.scale, self.q, self.m*self.k, self.cast, 256,
                enable_fp_fusion=False)
        a = self.x if fused else self.q
        # Unused descriptor arguments are eliminated for conventional loads.
        self.compiled[path] = projection[self.grid](
            a, self.scale, self.weight, self.desc["x" if fused else "q"],
            self.desc["s"], self.desc["w"], self.outputs[path],
            TMA=tma, FUSED=fused, **self.meta)
        return self.outputs[path]


PATHS = ("std_split", "std_fused", "tma_split", "tma_fused")
