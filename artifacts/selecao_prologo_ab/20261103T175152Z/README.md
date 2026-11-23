# Traces reais da seleção TMA — 03/11/2026

Esta é uma **nova coleta**, reproduzindo os três braços do experimento dos
commits `d46f489` e `db75391`. Os commits anteriores não preservavam uma timeline
CUDA: `trace.enabled` do Inductor salvava código gerado e diagnósticos de compilação.

Abra `trace_viewer.html` para navegar pelos 29 traces. Escolha um caso, uma das
cinco chamadas e o recorte da GPU ou a chamada completa. Os botões com os nomes
dos kernels mostram início, duração, stream e correlação. Os arquivos
`arm*/*_runtime_trace.json` também podem ser abertos em https://ui.perfetto.dev
para explorar todos os eventos originais e os fluxos CPU/GPU.

Exemplos:

- `arm2/pos_256_256_256/patched`: um kernel Triton comum com cast fundido.
- `arm2/pos_4096_4096_4096/patched`: cast separado seguido de matmul externo.
- `arm2/neg_b_1024_1024_1024/patched`: cast em B separado e matmul externo;
  o log de compilação registra seis rejeições de fusão por `ContractError`.
- `arm1/neg_b_1024_1024_1024/patched`: o template estendido aparece no wrapper,
  **sem fundir o prólogo**. O cast em B continua separado, antes do matmul TMA.
  A presença do template no wrapper não significa que o prólogo foi fundido.

## Verificação

`runtime_trace_audit.json` confirma, para 29 traces e 145 chamadas:

- quantidade, nomes Triton e ordem dos kernels compatíveis com o código gerado;
- todas as chamadas associadas aos lançamentos pela correlação CUDA;
- comparação numérica com a referência aprovada; os casos foram validados com
  sementes 0, 1 e 2, `atol=0.125`, `rtol=0.01`, e novamente após a captura;
- captura protegida com `torch.compiler.set_stance("fail_on_recompile")`.

Os kernels são contados pela categoria `kernel` do Chrome trace. As regiões
`gpu_user_annotation`, criadas por `record_function`, não contam como kernels.

Há cinco traces com eventos GPU anteriores à API CPU correlacionada, indicando
desalinhamento entre os relógios na captura. O visualizador avisa nesses casos;
a atribuição das chamadas usa correlação, em vez de inferir pelo timestamp.
Não use esses traces para medir atraso de lançamento CPU→GPU. As regiões CPU
incluem sincronização; durações sob profiler não substituem o benchmark.

O trace mostra o **programa final executado**. As tentativas de fusão rejeitadas
e os candidatos descartados são evidenciados pelos arquivos `*_fusion.log` e
pelos campos `fused_compiles`/`fused_benchmarks` dos `results.json`, não pela
timeline da execução final.

Nenhum dos casos positivos desta rodada usou o template estendido no wrapper.
As decisões de fusão variaram em relação à coleta histórica; esta execução não
substitui suas medições de desempenho.

## Ajustes necessários para capturar a versão correta

A primeira tentativa de captura, feita depois de compilar as duas versões,
detectou divergências entre o wrapper salvo e os kernels executados. O script
limpava o cache com `torch._dynamo.reset()` entre `patched` e `original`, o que
invalidava o runner anterior antes de voltar a executá-lo.

A coleta final limpa o cache uma vez por caso e captura cada versão antes da
próxima compilação, com as mesmas opções e extensão, impedindo recompilações.
As latências auxiliares históricas que reexecutavam runners após o reset merecem
revisão; os benchmarks internos de seleção e wrappers salvos são evidências
separadas. As tentativas preliminares não integram esta coleta final.

## Reprodução

Na raiz do repositório, com Modal autenticado:

```sh
.venv/bin/modal run experiments/integracao/modal_selecao_prologo_ab.py --profile-runtime
python3 scripts/analise/visualizar_trace_selecao.py artifacts/selecao_prologo_ab/NOVA_RODADA
```

O primeiro comando aloca H100 na Modal. O segundo só lê arquivos locais,
gera a auditoria e cria o visualizador sem servidor. `experiment_snapshot.py`
e `arm*/patch_snapshot.py` preservam os códigos usados. O ambiente e o UUID
de cada GPU estão nos `results.json`. Os braços usam containers separados.

`viewer_qa.json` registra a navegação testada nos 29 casos, alternância entre
uma e cinco chamadas e verificação em larguras de 390 e 728 px, sem erros de JS.
