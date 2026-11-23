"""Patch experimental em memória para Torch 2.13/Triton 3.7.1.

Habilita o hook load_input de A no template persistent+TMA. As cargas do
subgrafo A são emitidas como descriptor loads, reutilizando o codegen de
aritmética do Inductor. Restrito a inputs fp16/int8 2D contíguos, mesma forma de A,
indexação identidade, tamanhos estáticos e tiles sem cauda M/K.
Não é um patch upstream ou suporte geral. Nenhum arquivo instalado é alterado.
"""

import contextlib

import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.simd import CantSplit
from torch._inductor.codegen.triton import IndexingOptions
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateKernel
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


SOURCE = '''{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
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
            {{load_input("A", "a", ("idx_m", "idx_k"), mask=None,
                         indent_width=12, index_shape=("BLOCK_M", "BLOCK_K"))}}
            b = tl.load_tensor_descriptor(b_desc, [rk_desc, rn_desc])
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
        idx_m = rm + tl.arange(0, BLOCK_M)[:, None]
        idx_n = rn + tl.arange(0, BLOCK_N)[None, :]
        mask = (idx_m < M) & (idx_n < N)
        {{store_output(("idx_m", "idx_n"), "acc", "mask", indent_width=8,
                        val_shape=("BLOCK_M", "BLOCK_N"))}}
'''


class ContractError(CantSplit, NotImplementedError):
    """Fora do contrato. Como CantSplit, o scheduler pula a escolha na fusão
    com benchmark em vez de abortar a compilação."""

    def __init__(self, message):
        # CantSplit.__init__ exige "remaining"; a mensagem basta aqui.
        Exception.__init__(self, message)

    def __str__(self):
        # CantSplit.__str__ usa expr/remaining, ausentes aqui.
        return str(self.args[0])


class RestrictedTMATemplateKernel(TritonTemplateKernel):
    @contextlib.contextmanager
    def set_subgraph_body(self, body_name):
        previous = getattr(self, "_in_a_load_subgraph", False)
        self._in_a_load_subgraph = body_name == "<LOAD_INPUT_A>"
        try:
            with super().set_subgraph_body(body_name):
                yield
        finally:
            self._in_a_load_subgraph = previous

    def load_input(self, input_name, output_name, indices, **kwargs):
        key = super().load_input(input_name, output_name, indices, **kwargs)
        if input_name != "A":
            raise ContractError("Somente prologue no operando A")
        # B é lido direto por descritor; um prólogo fundido em B seria ignorado.
        b = self.named_input_nodes.get("B")
        if b is not None and b.get_name() in self.prologue_fused_inputs:
            raise ContractError("Prologue em B não implementado")
        old_hook = self.render_hooks[key]
        node = self.named_input_nodes[input_name]

        def hook():
            if node.get_name() in self.prologue_fused_inputs:
                return old_hook()
            # Sem prologue: A continua sendo Q materializado ou entrada direta.
            line = f"{output_name} = tl.load_tensor_descriptor(tl.make_tensor_descriptor(base=A, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]), [rm_desc, rk_desc])"
            return line

        self.render_hooks[key] = hook
        return key

    def load(self, name, index):
        if not getattr(self, "_in_a_load_subgraph", False):
            return super().load(name, index)
        node = V.graph.get_buffer(name)
        layout = node.get_layout()
        expected_shape = self.named_input_nodes["A"].get_size()
        if (node.get_dtype() not in (torch.float16, torch.int8) or layout.size != expected_shape
                or layout.stride != [expected_shape[1], sympy.Integer(1)]
                or layout.offset != 0):
            raise ContractError(f"Carga fora do contrato TMA: {name}, {layout}")
        indexing = self.indexing(index, copy_shape=self.template_out_shape,
                                 override_mask=self.template_mask)
        # Indexação usa símbolos com assumptions próprias do codegen. Reutilizá-los
        # evita comparar Symbols de mesmo nome, mas identidades SymPy distintas.
        symbols = {str(symbol): symbol for symbol in index.free_symbols}
        # Estes nomes são o contrato explícito do hook de A em SOURCE; o hook
        # de saída usa idx_n e pode ter substituído template_indices na instância.
        identity = (symbols.get("idx_m", sympy.Symbol("idx_m")) * expected_shape[1]
                    + symbols.get("idx_k", sympy.Symbol("idx_k")))
        if not isinstance(indexing, IndexingOptions) or not (
            str(indexing.index) == "xindex" or sympy.simplify(index - identity) == 0
        ):
            raise ContractError(f"Indexação não identidade: {name}, {index}, {indexing}")
        pointer = self.args.input(name)
        line = f"tl.load_tensor_descriptor(tl.make_tensor_descriptor(base={pointer}, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]), [rm_desc, rk_desc])"
        dtype = node.get_dtype()
        if dtype == torch.float16 and config.triton.codegen_upcast_to_fp32:
            line += ".to(tl.float32)"
            dtype = torch.float32
        value = self.cse.generate(self.loads, line, dtype=dtype,
                                  shape=("BLOCK_M", "BLOCK_K"))
        value.mask_vars = OrderedSet()
        return value


@contextlib.contextmanager
def restricted_tma_template(enabled, declare_b=False):
    """declare_b=True declara A e B como entradas de prólogo, como o Triton
    comum, para que a escolha não seja descartada na fusão com benchmark.
    Prólogo em B continua não suportado e é rejeitado com ContractError."""
    import torch._inductor.kernel.mm as mm

    original = mm.persistent_tma_mm_template
    if not enabled:
        yield original
        return
    saved_registry = TritonTemplate.all_templates.pop(original.name)
    patched = TritonTemplate(name=original.name, grid=original.grid, source=SOURCE,
                             prologue_loads_all_inputs=declare_b)
    patched.kernel_type = RestrictedTMATemplateKernel
    mm.persistent_tma_mm_template = patched
    try:
        yield patched
    finally:
        mm.persistent_tma_mm_template = original
        TritonTemplate.all_templates[original.name] = saved_registry


@contextlib.contextmanager
def selectable_tma_template(enabled, decisions, declare_b=False):
    """Preserva candidatos nativos e usa extensão só na geometria suportada.

    Este filtro trata o matmul; validação/fallback de todo subgrafo de prologue
    continua sendo uma limitação separada.
    """
    import torch._inductor.kernel.mm as mm

    original = mm.persistent_tma_mm_template
    if not enabled:
        yield original
        return
    with restricted_tma_template(True, declare_b) as patched:
        generate = patched.generate

        def guarded_generate(*args, **kwargs):
            nodes = kwargs.get("input_nodes", args[0] if args else ())
            supported = len(nodes) == 2
            if supported:
                a, b = nodes
                ashape, bshape = a.get_size(), b.get_size()
                supported = (len(ashape) == len(bshape) == 2
                             and all(isinstance(s, sympy.Integer) for s in (*ashape, *bshape))
                             and a.get_dtype() == b.get_dtype() == torch.float16
                             and a.get_stride() == [ashape[1], 1]
                             and b.get_stride() == [bshape[1], 1]
                             and a.get_layout().offset == b.get_layout().offset == 0)
                if supported:
                    supported = (ashape[0] % kwargs["BLOCK_M"] == 0
                                 and ashape[1] % kwargs["BLOCK_K"] == 0
                                 and kwargs["BLOCK_K"] * 2 % 16 == 0)
            decisions.append({"extended": bool(supported),
                              "tile": [kwargs.get(key) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K")]})
            return (generate if supported else original.generate)(*args, **kwargs)

        patched.generate = guarded_generate
        yield patched


@contextlib.contextmanager
def fixed_tma_choice(template):
    """Controle diagnóstico: apenas TMA, um tile fixo, não seleção padrão."""
    import torch._inductor.kernel.mm as mm

    original_standard = mm.mm_template.maybe_append_choice
    original_append = template.maybe_append_choice
    mm.mm_template.maybe_append_choice = lambda *args, **kwargs: None
    seen_choices = []

    def append(choices, **kwargs):
        # Uma configuração por lista de choices; preservar metadados de TMA.
        if any(previous is choices for previous in seen_choices):
            return None
        seen_choices.append(choices)
        kwargs.update(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_stages=3, num_warps=4)
        if not kwargs.get("A_ROW_MAJOR") or not kwargs.get("B_ROW_MAJOR"):
            raise NotImplementedError("Somente A/B row-major")
        m, k = kwargs["input_nodes"][0].get_size()
        if not isinstance(m, sympy.Integer) or not isinstance(k, sympy.Integer) or m % 64 or k % 64:
            raise NotImplementedError("Somente M/K estáticos múltiplos de 64")
        return original_append(choices, **kwargs)

    template.maybe_append_choice = append
    try:
        yield
    finally:
        template.maybe_append_choice = original_append
        mm.mm_template.maybe_append_choice = original_standard
