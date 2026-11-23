# Contrato e fallback: oito casos — 03/11/2026

**Resultado: execuções corretas e protocolo conforme; cobertura 5 de 8.**

Comando: `.venv/bin/modal run experiments/integracao/modal_contrato_tma.py --cases all`.
App: https://modal.com/apps/pedromarcallima/main/ap-tXosi6V70ys9IoNpJTXMdq

Todos os casos rodaram na H100 com os controles original e extensão: 16 execuções,
três sementes cada, comparadas a eager (`atol=0.125`, `rtol=0.01`), todas corretas,
com erro absoluto máximo 0,0. A captura de cada controle usou `fail_on_recompile`.
`contract_verification.json` registra `all_executions_passed=true`,
`protocol_conformant=true` e `planned_coverage_complete=false`. Com
`--require-coverage`, o auditor retorna código 1 (cobertura incompleta), como o
protocolo prevê.

Em todos os controles o autotuner recebeu três escolhas (ATen, um Triton comum e
um TMA 64×128×64), com 20 configurações comuns e sete TMA puladas pelo limite.

## Cobertura

| Caso | Alvo | O que ocorreu | Estado |
|---|---|---|---|
| `positive` | filtro aceita | aceito; extensão no wrapper, 1 kernel | coberto |
| `tail_m` | filtro recusa `tail_m` | recusado; nativo | coberto |
| `tail_k` | filtro recusa `tail_k` | recusado; nativo | coberto |
| `dtype` | filtro recusa `dtype` | recusado; nativo | coberto |
| `prologue_b` | guarda recusa `prologue_b` | guarda rejeitou 1 tentativa; outra fusão válida | coberto |
| `noncontiguous` | filtro recusa `stride_a` | filtro aceitou; scheduler nativo recusou a fusão | não coberto |
| `offset` | filtro recusa `offset_a` | filtro aceitou; matmul puro, sem prólogo | não coberto |
| `broadcast` | guarda recusa `prologue_load_layout` | scheduler nativo recusou a fusão antes da guarda | não coberto |

### Por que três não foram cobertos

O filtro olha as entradas do nó de matmul, já depois de qualquer prólogo; isso vale
também para o `positive`. Os motivos, lidos em `<caso>/patched_fusion.log`:

- `noncontiguous`: `cannot fuse op0 with op1: memory deps did not match`. O
  `run.log` registra ainda `Layout conflict detected for buf0` (o template espera
  stride `[256, 1]`, o layout está fixado em `[1, 256]`) nas compilações desse caso.
  A relação entre os dois registros não foi investigada.
- `broadcast`: `prologue fusion that must be upcast to fp32 not profitable for low
  precision templates`, a mesma heurística nativa que já barrava a ReLU.
  `prologue_rejections` ficou vazio: a guarda de carga não foi chamada.
- `offset`: expressão sem prólogo (`x_fp16 @ w_fp16`); só `extern_kernels.mm` executa.
  O filtro leu offset 0. A causa não foi verificada no código do Inductor.

Portanto `stride_a`, `offset_a` e `prologue_load_layout` seguem sem evidência de
execução. Não se desligaram heurísticas nativas para forçar cobertura.

Em `dtype` e `prologue_b`, original e extensão terminaram com kernels diferentes
porque o autotuner escolheu de forma distinta. Não é comparação causal de desempenho.

## Proveniência

- `collection.json`: IDs, casos pedidos e concluídos, método.
- `<caso>/`: resultados, wrappers, logs de fusão, traces e snapshot do patch.
- `experiment_snapshot.py`: fonte do experimento usado.
- `execution_status.json`: app parado, zero tarefas. O código de saída local da
  coleta não foi registrado.
