# AOT ID: ['4_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/tcc_plain_cache/tmp70xmn_n0/la/cla5ecdyw4h5boiat5a2jzvdmwazuzfel25irx7aqy4g3g2xp6x4.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %arg0_1 : Tensor "f16[2048, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %relu : Tensor "f16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%arg0_1,), kwargs = {})
#   return %relu
triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3145728}, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/tcc_plain_cache/tmp70xmn_n0/aj/cajcnshu5bk6xtivuuef6schvvaiazozv2g6qnerkaur3eev6ijh.py
# Topologically Sorted Source Nodes: [relu, matmul], Original ATen: [aten.relu, aten.mm]
# Source node to ATen node mapping:
#   matmul => mm
#   relu => relu
# Graph fragment:
#   %relu : Tensor "f16[2048, 256][256, 1]cuda:0" = PlaceHolder[target=relu]
#   %arg1_1 : Tensor "f16[256, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %relu : Tensor "f16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%arg0_1,), kwargs = {})
#   %mm : Tensor "f16[2048, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%relu, %arg1_1), kwargs = {})
#   return %mm
triton_tem_fused_mm_relu_1 = async_compile.triton('triton_tem_fused_mm_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16', 'ws_ptr': '*u8'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_mm_relu_1', 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'A_ROW_MAJOR': True, 'B_ROW_MAJOR': True, 'NUM_SMS': 132, 'TMA_SIZE': 128, 'TMA_EXPERIMENTAL_API': False, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_mm_relu_1(arg_A, arg_B, out_ptr0, ws_ptr):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    A_ROW_MAJOR : tl.constexpr = True
    B_ROW_MAJOR : tl.constexpr = True
    NUM_SMS : tl.constexpr = 132
    TMA_SIZE : tl.constexpr = 128
    TMA_EXPERIMENTAL_API : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2048
    N = 1024
    K = 256
    # TCC_RESTRICTED_TMA_PROLOGUE: A passa pelo hook do Inductor.
    b_desc = tl.make_tensor_descriptor(
        base=B, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
    )
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    for tile in tl.range(tl.program_id(0), tiles_m * tiles_n, NUM_SMS):
        rm = (tile // tiles_n) * BLOCK_M
        rn = (tile % tiles_n) * BLOCK_N
        rm_desc = rm.to(tl.int32)
        rn_desc = rn.to(tl.int32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), ACC_TYPE)
        for rk in tl.range(0, K, BLOCK_K):
            rk_desc = rk.to(tl.int32)
            idx_m = rm + tl.arange(0, BLOCK_M)[:, None]
            idx_k = rk + tl.arange(0, BLOCK_K)[None, :]
            a = tl.load_tensor_descriptor(tl.make_tensor_descriptor(base=A, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]), [rm_desc, rk_desc])
            b = tl.load_tensor_descriptor(b_desc, [rk_desc, rn_desc])
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
        idx_m = rm + tl.arange(0, BLOCK_M)[:, None]
        idx_n = rn + tl.arange(0, BLOCK_N)[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 1024*idx_m
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2048, 256), (256, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((2048, 256), (256, 1), torch.float16)
            # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_0.run(arg0_1, buf0, 524288, stream=raw_stream0)
            del arg0_1
            assert_size_stride(arg1_1, (256, 1024), (1024, 1), 'input')
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float16)
            # Topologically Sorted Source Nodes: [relu, matmul], Original ATen: [aten.relu, aten.mm]
            workspace_1 = empty_strided_cuda((33792, ), (1, ), torch.uint8)
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused_mm_relu_1.run(buf0, arg1_1, buf1, workspace_1, 132, 1, 1, stream=raw_stream0)
            del workspace_1
            del arg1_1
            del buf0
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
