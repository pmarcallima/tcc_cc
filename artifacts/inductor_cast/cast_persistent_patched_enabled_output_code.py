# AOT ID: ['5_inference']
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
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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


# kernel path: /tmp/tcc_plain_cache/tmp_5hrp0oi/gz/cgznweukpgccdrxg6lte3gio76d6i4b4xxgqhj57x2qhrvdhmebq.py
# Topologically Sorted Source Nodes: [to, matmul], Original ATen: [aten._to_copy, aten.mm]
# Source node to ATen node mapping:
#   matmul => mm
#   to => convert_element_type
# Graph fragment:
#   %arg0_1 : Tensor "i8[2048, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f16[2048, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type]
#   %arg1_1 : Tensor "f16[256, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float16), kwargs = {})
#   %mm : Tensor "f16[2048, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type, %arg1_1), kwargs = {})
#   return %convert_element_type,%mm
triton_tem_fused__to_copy_mm_0 = async_compile.triton('triton_tem_fused__to_copy_mm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_B': '*fp16', 'in_ptr1': '*i8', 'out_ptr0': '*fp16', 'ws_ptr': '*u8'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_mm_0', 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'A_ROW_MAJOR': True, 'B_ROW_MAJOR': True, 'NUM_SMS': 132, 'TMA_SIZE': 128, 'TMA_EXPERIMENTAL_API': False, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_mm_0(arg_B, in_ptr1, out_ptr0, ws_ptr):
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
            xindex = idx_k + 256*idx_m
            tmp0 = tl.load_tensor_descriptor(tl.make_tensor_descriptor(base=in_ptr1, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]), [rm_desc, rk_desc])
            tmp1 = tmp0.to(tl.float16)
            a = tmp1.broadcast_to(xindex.shape)
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
        assert_size_stride(arg1_1, (256, 1024), (1024, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float16)
            # Topologically Sorted Source Nodes: [to, matmul], Original ATen: [aten._to_copy, aten.mm]
            workspace_2 = empty_strided_cuda((33792, ), (1, ), torch.uint8)
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_mm_0.run(arg1_1, arg0_1, buf1, workspace_2, 132, 1, 1, stream=raw_stream0)
            del workspace_2
            del arg0_1
            del arg1_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.int8)
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
