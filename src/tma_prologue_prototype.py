"""Protótipo manual persistent+TMA: prologue integrado versus materializado.

Inspirado no tutorial 09-persistent-matmul.py do Triton v3.7.1.
Contrato numérico explícito: prologue em fp32, um cast para fp16 antes do dot.
Entradas: X[M,K], scale[M,K], W[K,N], todas fp16 contíguas.
Saída: Y[M,N] fp16; acumulador do matmul fp32.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def materialize_prologue(X, Scale, Q, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < SIZE, other=0).to(tl.float32)
    scale = tl.load(Scale + index, index < SIZE, other=0).to(tl.float32)
    q = (x * scale + (scale - 1.0)).to(tl.float16)
    tl.store(Q + index, q, index < SIZE)


@triton.jit
def persistent_tma_prologue(
    Input_desc, Scale_desc, W_desc, Output,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    NUM_SMS: tl.constexpr, FUSED: tl.constexpr,
):
    # Um CTA pode processar vários tiles de saída, com passo NUM_SMS.
    tiles_n = tl.cdiv(N, BN)
    tiles = tl.cdiv(M, BM) * tiles_n
    for tile in tl.range(tl.program_id(0), tiles, NUM_SMS):
        row = (tile // tiles_n) * BM
        col = (tile % tiles_n) * BN
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            # FUSED: Input_desc aponta para X. Separado: aponta para Q pronto.
            a = Input_desc.load([row, ki * BK])
            b = W_desc.load([ki * BK, col])
            if FUSED:
                scale = Scale_desc.load([row, ki * BK]).to(tl.float32)
                x = a.to(tl.float32)
                # Q existe apenas como valor local, nunca como buffer global.
                a = (x * scale + (scale - 1.0)).to(tl.float16)
            acc = tl.dot(a, b, acc)
        rows = row + tl.arange(0, BM)
        cols = col + tl.arange(0, BN)
        tl.store(
            Output + rows[:, None] * N + cols[None, :], acc.to(tl.float16),
            (rows[:, None] < M) & (cols[None, :] < N),
        )


class TMAPrologueCase:
    """Pré-aloca saídas e descritores: alocações ficam fora das medições."""

    def __init__(self, x, scale, weight, bm=128, bn=128, bk=64, stages=2, warps=4):
        self.m, self.k = x.shape
        self.n = weight.shape[1]
        assert scale.shape == x.shape and weight.shape[0] == self.k
        assert all(t.dtype == torch.float16 and t.is_contiguous() for t in (x, scale, weight))
        # Restrição deste protótipo: K sem cauda; linhas contíguas alinhadas a 16 bytes.
        assert self.k % bk == 0 and self.n % 8 == 0
        self.x, self.scale, self.weight = x, scale, weight
        self.q = torch.empty_like(x)
        self.fused_output = torch.empty((self.m, self.n), device=x.device, dtype=x.dtype)
        self.split_output = torch.empty_like(self.fused_output)
        self.x_desc = TensorDescriptor.from_tensor(x, [bm, bk])
        self.q_desc = TensorDescriptor.from_tensor(self.q, [bm, bk])
        self.scale_desc = TensorDescriptor.from_tensor(scale, [bm, bk])
        self.w_desc = TensorDescriptor.from_tensor(weight, [bk, bn])
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        self.grid = (min(sms, triton.cdiv(self.m, bm) * triton.cdiv(self.n, bn)),)
        self.meta = dict(M=self.m, N=self.n, K=self.k, BM=bm, BN=bn, BK=bk,
                         NUM_SMS=sms, num_warps=warps, num_stages=stages, enable_fp_fusion=False)
        self.compiled = {}

    def fused(self):
        self.compiled["fused"] = persistent_tma_prologue[self.grid](
            self.x_desc, self.scale_desc, self.w_desc, self.fused_output,
            FUSED=True, **self.meta,
        )
        return self.fused_output

    def split(self):
        self.compiled["pointwise"] = materialize_prologue[(triton.cdiv(self.m * self.k, 256),)](
            self.x, self.scale, self.q, SIZE=self.m * self.k, BLOCK=256,
            enable_fp_fusion=False,
        )
        self.compiled["matmul"] = persistent_tma_prologue[self.grid](
            self.q_desc, self.scale_desc, self.w_desc, self.split_output,
            FUSED=False, **self.meta,
        )
        return self.split_output
