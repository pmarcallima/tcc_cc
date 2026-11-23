# ============================================================================
# TIER 1 — Contraste de prologue fusion no template PADRAO (Colab T4, grátis)
# ----------------------------------------------------------------------------
# PROBE = cast de operando: x @ w_int8.to(fp16)
#   O prologue é o CAST int8->fp16 do operando B do matmul.
#   Esse é o caso que a heurística do Inductor (scheduler.py:5277) deixa fundir:
#   ler int8 (pouco) é mais barato que materializar um buffer fp16 (2x).
#   (Um simples x+1 fp16->fp16 NÃO funde: benchmark decide "generally slower".)
#
# Contraste:
#   prologue_fusion=True  -> cast absorvido no kernel do matmul (sem buffer fp16)
#   prologue_fusion=False -> cast em kernel triton_poi_ próprio + buffer fp16 DRAM
#
# T4 = CC 7.5 (sem TMA). Aqui só validamos que o MECANISMO de prologue fusion
# existe e funde no template padrão. O gap TMA é o Tier 2a (H100).
#
# Rodar (upload deste arquivo no Colab, evitar copiar/colar):
#   !python notebook_tier1_t4.py
# ============================================================================
import os, re, glob, shutil
import torch
import torch._inductor.config as cfg
import torch._inductor.utils as iu
import torch._dynamo

assert torch.cuda.is_available(), "Precisa de GPU CUDA (T4 no Colab)."
CC = torch.cuda.get_device_capability()
print(f"GPU: {torch.cuda.get_device_name(0)}  compute capability={CC}\n")

# T4 = 40 SMs; Inductor exige >=68 (is_big_gpu) p/ template Triton de GEMM. Sem
# este override cai em cuBLAS extern e nenhum prologue funde. Forçamos p/ a demo:
iu.is_big_gpu = lambda *a, **k: True

TRACE_ROOT = "/content/tcc_trace" if os.path.isdir("/content") else "./tcc_trace"

def make_probe():
    def f(x, w_i8):
        return x @ w_i8.to(torch.float16)   # prologue = cast int8->fp16 do operando B
    return f

# ---------------------------------------------------------------------------
def analyze(path):
    src = open(path).read()
    poi = re.findall(r"def (triton_poi_\w+)\(", src)      # kernels pointwise avulsos
    red = re.findall(r"def (triton_red_\w+)\(", src)
    tem = re.findall(r"def (triton_tem_\w+)\(", src)      # kernel(s) do template matmul
    extern = len(re.findall(r"extern_kernels\.(mm|addmm|bmm)", src))
    bufs = re.findall(r"(buf\d+) = empty_strided", src)
    # o cast fundido aparece no NOME do kernel template (_to_copy / convert)
    cast_in_template = bool(re.search(r"triton_tem_\w*(to_copy|convert)\w*", src))
    cast_standalone = [k for k in poi if re.search(r"(to_copy|convert)", k)]
    return {"path": path, "src": src, "poi": poi, "red": red, "tem": tem,
            "n_poi": len(poi), "n_tem": len(tem), "extern_mm": extern,
            "n_bufs": len(bufs), "cast_in_template": cast_in_template,
            "cast_standalone": cast_standalone}

def find_oc(run_dir):
    hits = glob.glob(os.path.join(run_dir, "**", "output_code.py"), recursive=True)
    return sorted(hits, key=os.path.getmtime)[-1] if hits else None

def run_case(name, prologue_fusion):
    torch._dynamo.reset()
    run_dir = os.path.join(TRACE_ROOT, name)
    shutil.rmtree(run_dir, ignore_errors=True); os.makedirs(run_dir, exist_ok=True)
    cfg.max_autotune = True
    cfg.max_autotune_gemm = True
    cfg.max_autotune_gemm_backends = "TRITON"
    cfg.prologue_fusion = prologue_fusion
    cfg.epilogue_fusion = True
    cfg.force_disable_caches = True
    cfg.trace.enabled = True
    cfg.trace.output_code = True
    cfg.trace.debug_dir = run_dir

    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    w_i8 = torch.randint(-4, 5, (2048, 2048), device="cuda", dtype=torch.int8)
    f = torch.compile(make_probe())
    out = f(x, w_i8); torch.cuda.synchronize()
    ok = torch.allclose(out, x @ w_i8.to(torch.float16), atol=1e-2, rtol=1e-2)

    oc = find_oc(run_dir)
    if oc is None:
        print(f"[{name}] output_code.py não encontrado"); return None
    r = analyze(oc); r["allclose"] = ok; r["name"] = name
    return r

print("=== Caso A: prologue_fusion=True  (esperado: cast funde no template) ===")
A = run_case("prologue_on", True)
print("=== Caso B: prologue_fusion=False (esperado: cast em kernel separado) ===")
B = run_case("prologue_off", False)

def fmt(r):
    if r is None: return "—"
    return (f"poi={r['n_poi']}{r['poi']} tem={r['n_tem']}{r['tem']} "
            f"extern_mm={r['extern_mm']} bufs={r['n_bufs']} "
            f"cast_no_template={r['cast_in_template']} "
            f"cast_avulso={r['cast_standalone']} allclose={r['allclose']}")

print("\n" + "="*80)
print("RESULTADO TIER 1 (determinístico) — probe: x @ w_int8.to(fp16)")
print("="*80)
print(f"prologue_fusion=True : {fmt(A)}")
print(f"prologue_fusion=False: {fmt(B)}")
print("-"*80)
if A and B:
    d_poi = B["n_poi"] - A["n_poi"]
    d_buf = B["n_bufs"] - A["n_bufs"]
    print(f"Δ kernels pointwise (off-on) = {d_poi}   (esperado >0)")
    print(f"Δ buffers           (off-on) = {d_buf}   (esperado >0)")
    fused = (A["cast_in_template"] and not A["cast_standalone"]
             and bool(B["cast_standalone"]))
    print("\nPROLOGUE FUSION DEMONSTRADA ✅" if fused
          else "\nNão fundiu como esperado — ver output_code (rodar diag).")
    print(f"\noutput_code ON : {A['path']}\noutput_code OFF: {B['path']}")
print("="*80)
