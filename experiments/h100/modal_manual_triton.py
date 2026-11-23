import json
import random
import statistics

import modal


app = modal.App("tcc-manual-triton-fusion")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")
vol = modal.Volume.from_name("tcc-tier2-out", create_if_missing=True)


@app.function(gpu="H100", image=image, volumes={"/out": vol}, timeout=1800)
def compare_manual_fusion():
    import os

    import torch
    import torch._dynamo
    import torch._inductor.config as cfg
    import torch._inductor.kernel.mm as mmmod
    import triton as triton_module
    import triton.language as tl_module
    from torch._higher_order_ops import InvokeQuant

    # Os kernels JIT resolvem `tl` no namespace global do modulo.
    globals()["tl"] = tl_module

    @triton_module.jit
    def manual_pointwise_prologue(
        x_ptr,
        scale_ptr,
        q_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        scale = tl.load(scale_ptr + offsets, mask=mask)
        q = (x * scale + (scale - 1.0)).to(tl.float16)
        tl.store(q_ptr + offsets, q, mask=mask)

    @triton_module.jit
    def manual_matmul(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            offsets_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr + offsets_m[:, None] * K + offsets_k[None, :],
                mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offsets_k[:, None] * N + offsets_n[None, :],
                mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(a, b)

        output = accumulator.to(tl.float16)
        tl.store(
            c_ptr + offsets_m[:, None] * N + offsets_n[None, :],
            output,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )

    @triton_module.jit
    def manual_fused_prologue_matmul(
        x_ptr,
        scale_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            offsets_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            x_offsets = offsets_m[:, None] * K + offsets_k[None, :]
            x_mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
            x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            scale = tl.load(scale_ptr + x_offsets, mask=x_mask, other=1.0)
            # O cast reproduz a materializacao fp16 do caminho separado.
            a = (x * scale + (scale - 1.0)).to(tl.float16)
            b = tl.load(
                b_ptr + offsets_k[:, None] * N + offsets_n[None, :],
                mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(a, b)

        output = accumulator.to(tl.float16)
        tl.store(
            c_ptr + offsets_m[:, None] * N + offsets_n[None, :],
            output,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )

    M = N = K = 256
    block_m = block_n = block_k = 32
    num_warps = 4
    warmup = 100
    rounds = 200
    calls_per_round = 100

    torch.manual_seed(0)
    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    scale = torch.randn((M, K), device="cuda", dtype=torch.float16)
    weight = torch.randn((K, N), device="cuda", dtype=torch.float16)
    intermediate = torch.empty_like(x)
    fused_output = torch.empty((M, N), device="cuda", dtype=torch.float16)
    split_output = torch.empty_like(fused_output)

    pointwise_grid = (triton_module.cdiv(M * K, 256),)
    matmul_grid = (
        triton_module.cdiv(M, block_m),
        triton_module.cdiv(N, block_n),
    )
    pointwise_meta = {"n_elements": M * K, "BLOCK_SIZE": 256}
    matmul_meta = {
        "M": M,
        "N": N,
        "K": K,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": num_warps,
    }

    # warmup() compila sem executar e devolve o CompiledKernel com os estagios de IR.
    pointwise_compiled = manual_pointwise_prologue.warmup(
        x, scale, intermediate, grid=pointwise_grid, **pointwise_meta
    )
    matmul_compiled = manual_matmul.warmup(
        intermediate,
        weight,
        split_output,
        grid=matmul_grid,
        **matmul_meta,
    )
    fused_compiled = manual_fused_prologue_matmul.warmup(
        x,
        scale,
        weight,
        fused_output,
        grid=matmul_grid,
        **matmul_meta,
    )

    def run_split():
        manual_pointwise_prologue[pointwise_grid](
            x, scale, intermediate, **pointwise_meta
        )
        manual_matmul[matmul_grid](
            intermediate, weight, split_output, **matmul_meta
        )

    def run_fused():
        manual_fused_prologue_matmul[matmul_grid](
            x, scale, weight, fused_output, **matmul_meta
        )

    invoke_quant = InvokeQuant()

    def automatic_quant_prologue(x, scale):
        return torch.mul(x, scale) + (scale - 1)

    # Code objects distintos evitam que variantes de configuracao compartilhem
    # indevidamente a mesma entrada do cache Dynamo.
    def automatic_std_fused(x, scale, weight):
        q = invoke_quant(
            automatic_quant_prologue,
            x,
            scale,
            scheme="nf4",
            quant_options=invoke_quant,
        )
        return q @ weight

    def automatic_std_split(x, scale, weight):
        return automatic_quant_prologue(x, scale) @ weight

    def automatic_tma_gap(x, scale, weight):
        q = invoke_quant(
            automatic_quant_prologue,
            x,
            scale,
            scheme="nf4",
            quant_options=invoke_quant,
        )
        return q @ weight

    def force_tma_only(enable):
        if enable:
            if not hasattr(mmmod.mm_template, "_manual_compare_orig"):
                mmmod.mm_template._manual_compare_orig = (
                    mmmod.mm_template.maybe_append_choice
                )
            mmmod.mm_template.maybe_append_choice = lambda *args, **kwargs: None
        elif hasattr(mmmod.mm_template, "_manual_compare_orig"):
            mmmod.mm_template.maybe_append_choice = (
                mmmod.mm_template._manual_compare_orig
            )

    def activate_automatic_case(tma, prologue_fusion):
        cfg.prologue_fusion = prologue_fusion
        cfg.triton.enable_persistent_tma_matmul = tma

    def compile_automatic_case(fn, tma, prologue_fusion):
        cfg.max_autotune = True
        cfg.max_autotune_gemm = True
        cfg.max_autotune_gemm_backends = "TRITON"
        cfg.epilogue_fusion = True
        cfg.force_disable_caches = True
        activate_automatic_case(tma, prologue_fusion)
        force_tma_only(tma)
        try:
            compiled = torch.compile(fn)
            output = compiled(x, scale, weight)
            torch.cuda.synchronize()
            return compiled, output
        finally:
            force_tma_only(False)

    automatic_specs = {
        "AUTO_STD_FUSED": (automatic_std_fused, False, True),
        "AUTO_STD_SPLIT": (automatic_std_split, False, False),
        "AUTO_TMA_GAP": (automatic_tma_gap, True, True),
    }
    torch._dynamo.reset()
    automatic_compiled = {}
    automatic_outputs = {}
    for name, (fn, tma, fusion) in automatic_specs.items():
        automatic_compiled[name], automatic_outputs[name] = compile_automatic_case(
            fn, tma, fusion
        )

    run_split()
    run_fused()
    torch.cuda.synchronize()
    reference = (x * scale + (scale - 1)) @ weight
    atol = 1e-1
    rtol = 1e-2
    correctness = {
        "split_allclose": bool(
            torch.allclose(split_output, reference, atol=atol, rtol=rtol)
        ),
        "fused_allclose": bool(
            torch.allclose(fused_output, reference, atol=atol, rtol=rtol)
        ),
        "fused_vs_split_allclose": bool(
            torch.allclose(fused_output, split_output, atol=1e-3, rtol=1e-3)
        ),
        "fused_vs_split_exact": bool(torch.equal(fused_output, split_output)),
        "split_max_abs_error": float((split_output - reference).abs().max()),
        "fused_max_abs_error": float((fused_output - reference).abs().max()),
        "reference_atol": atol,
        "reference_rtol": rtol,
    }
    for name, output in automatic_outputs.items():
        correctness[f"{name.lower()}_allclose"] = bool(
            torch.allclose(output, reference, atol=atol, rtol=rtol)
        )
        correctness[f"{name.lower()}_max_abs_error"] = float(
            (output - reference).abs().max()
        )
    if not all(
        correctness[key]
        for key in (
            "split_allclose",
            "fused_allclose",
            "fused_vs_split_allclose",
            "auto_std_fused_allclose",
            "auto_std_split_allclose",
            "auto_tma_gap_allclose",
        )
    ):
        raise RuntimeError(f"Falha de correcao numerica: {correctness}")

    runners = {
        "MANUAL_FUSED": run_fused,
        "MANUAL_SPLIT": run_split,
        "AUTO_STD_FUSED": lambda: automatic_compiled["AUTO_STD_FUSED"](
            x, scale, weight
        ),
        "AUTO_STD_SPLIT": lambda: automatic_compiled["AUTO_STD_SPLIT"](
            x, scale, weight
        ),
        "AUTO_TMA_GAP": lambda: automatic_compiled["AUTO_TMA_GAP"](
            x, scale, weight
        ),
    }

    def activate_runner(name):
        if name in automatic_specs:
            _, tma, fusion = automatic_specs[name]
            activate_automatic_case(tma, fusion)

    for name, runner in runners.items():
        activate_runner(name)
        for _ in range(warmup):
            runner()
    torch.cuda.synchronize()

    samples_us = {name: [] for name in runners}
    names = list(runners)
    rng = random.Random(0)
    for _ in range(rounds):
        rng.shuffle(names)
        for name in names:
            activate_runner(name)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(calls_per_round):
                runners[name]()
            end.record()
            end.synchronize()
            samples_us[name].append(
                start.elapsed_time(end) * 1000 / calls_per_round
            )

    profiler = {}
    for name, runner in runners.items():
        activate_runner(name)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(100):
                runner()
            torch.cuda.synchronize()
        events = [
            event
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and ("manual_" in event.name or "triton_" in event.name)
        ]
        profiler[name] = {
            "kernel_events": len(events),
            "kernels_per_call": len(events) / 100,
            "gpu_kernel_us_per_call": sum(
                event.time_range.elapsed_us() for event in events
            )
            / 100,
            "kernel_names": sorted({event.name for event in events}),
        }

    expected_kernels = {
        "MANUAL_FUSED": 1.0,
        "MANUAL_SPLIT": 2.0,
        "AUTO_STD_FUSED": 1.0,
        "AUTO_STD_SPLIT": 2.0,
        "AUTO_TMA_GAP": 2.0,
    }
    for name, expected in expected_kernels.items():
        actual = profiler[name]["kernels_per_call"]
        if actual != expected:
            raise RuntimeError(
                f"Estrutura invalida em {name}: {actual} kernels, esperado {expected}"
            )

    stats = {}
    for name, values in samples_us.items():
        stats[name] = {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "stdev_us": statistics.stdev(values),
            "p25_us": statistics.quantiles(values, n=4)[0],
            "p75_us": statistics.quantiles(values, n=4)[2],
        }

    ir_dir = "/out/manual_triton_ir"
    os.makedirs(ir_dir, exist_ok=True)
    compiled_kernels = {
        "pointwise_prologue": pointwise_compiled,
        "matmul": matmul_compiled,
        "fused_prologue_matmul": fused_compiled,
    }
    ir_manifest = {}
    for kernel_name, compiled in compiled_kernels.items():
        ir_manifest[kernel_name] = []
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            source = compiled.asm.get(stage)
            if source is None:
                continue
            path = os.path.join(ir_dir, f"{kernel_name}.{stage}")
            with open(path, "w") as output_file:
                output_file.write(str(source))
            ir_manifest[kernel_name].append(stage)

    fused_median = stats["MANUAL_FUSED"]["median_us"]
    split_median = stats["MANUAL_SPLIT"]["median_us"]
    result = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton_module.__version__,
        },
        "method": {
            "shape": [M, N, K],
            "dtype": "torch.float16",
            "blocks": [block_m, block_n, block_k],
            "warmup": warmup,
            "rounds": rounds,
            "calls_per_round": calls_per_round,
        },
        "correctness": correctness,
        "stats": stats,
        "profiler": profiler,
        "comparison": {
            "manual_fusion_speedup_x": split_median / fused_median,
            "manual_fusion_reduction_pct": 100 * (1 - fused_median / split_median),
            "automatic_std_fusion_speedup_x": stats["AUTO_STD_SPLIT"][
                "median_us"
            ]
            / stats["AUTO_STD_FUSED"]["median_us"],
            "automatic_std_fusion_reduction_pct": 100
            * (
                1
                - stats["AUTO_STD_FUSED"]["median_us"]
                / stats["AUTO_STD_SPLIT"]["median_us"]
            ),
            "manual_vs_automatic_fused_gpu_ratio_x": profiler["AUTO_STD_FUSED"][
                "gpu_kernel_us_per_call"
            ]
            / profiler["MANUAL_FUSED"]["gpu_kernel_us_per_call"],
            "intermediate_bytes": M * K * 2,
            "extra_intermediate_traffic_bytes": M * K * 2 * 2,
        },
        "ir_manifest": ir_manifest,
    }
    with open("/out/manual_triton_results.json", "w") as output_file:
        json.dump(result, output_file, indent=2)
    with open("/out/manual_vs_auto_results.json", "w") as output_file:
        json.dump(result, output_file, indent=2)
    vol.commit()
    return result


@app.local_entrypoint()
def main():
    result = compare_manual_fusion.remote()
    print(json.dumps(result, indent=2))
