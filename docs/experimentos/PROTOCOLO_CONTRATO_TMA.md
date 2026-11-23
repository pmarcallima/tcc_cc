# Contrato e fallback da extensão TMA — protocolo de 03/11/2026

**Estado: oito casos executados; cobertura 5 de 8.**
A [rodada completa](../../artifacts/contrato_tma/20261103T220051Z/contract_verification.json)
passou em correção (16 de 16 execuções) e em conformidade com o protocolo, com
limite de um candidato por template e três escolhas oferecidas por controle.
Foram cobertos `positive`, `tail_m`, `tail_k`, `dtype` e `prologue_b`.
`noncontiguous`, `offset` e `broadcast` **não foram cobertos**: o filtro recebeu
o candidato, e a fusão foi recusada pelo scheduler nativo antes das guardas
(`noncontiguous`, `broadcast`) ou não havia prólogo (`offset`).
Este protocolo continua a [prova controlada](../../artifacts/prova_fusao_tma/20261103T182123Z).
A pergunta agora é se os casos fora do domínio aceito conservam uma alternativa
correta, e qual etapa efetivamente recusou a extensão.

## Mudanças propostas para revisão

1. `src/inductor_tma_prologue_patch.py`: explicitar a primeira regra do filtro
   que falhou e registrar metadados, destino e resultado da geração do candidato.
   As condições de aceitação e a fonte Triton permanecem as mesmas. As exceções
   de prólogo recebem códigos de motivo, preservando suas mensagens e propagação.
2. `experiments/integracao/modal_contrato_tma.py`: uma coleta diagnóstica pequena,
   com os casos abaixo, controles original/extensão, correção e traces reais.
3. `scripts/auditoria/verificar_contrato_tma.py`: conferir os arquivos locais e
   reportar separadamente correção das execuções e cobertura dos alvos do protocolo.
   `test_contrato_tma.py` verifica roteamento, tratamento de falhas e interpretação
   da auditoria sem instalar PyTorch ou usar GPU.

## Casos definidos antes da coleta

Formas na ordem M/N/K. Exceto onde indicado, a expressão é
`x_int8.to(fp16) @ w_fp16`, com int8 entre −8 e 8 e peso `randn * 0.125`.

| Caso | Entrada/expressão | Alvo que precisa ser observado |
|---|---|---|
| `positive` | 256/256/256, contíguo | Filtro aceita e gera candidato estendido |
| `tail_m` | 257/256/256 | Filtro rejeita `tail_m` e gera candidato pelo template nativo |
| `tail_k` | 256/256/264 | Filtro rejeita `tail_k` e gera candidato pelo template nativo |
| `noncontiguous` | Entrada int8 transposta, 256/256/256 | Filtro rejeita `stride_a` e gera candidato nativo |
| `offset` | `x_fp16 @ w_fp16`, view contígua com offset de 8 elementos | Filtro rejeita `offset_a` e gera candidato nativo |
| `dtype` | `x_int8.to(bfloat16) @ w_bfloat16` | Filtro rejeita `dtype` e gera candidato nativo |
| `broadcast` | `(x_int8.to(fp16) + scale_1xK) @ w_fp16` | Guarda de carga do prólogo rejeita `prologue_load_layout` |
| `prologue_b` | `x_fp16 @ w_int8.to(fp16)` | Guarda de prólogo rejeita `prologue_b` |

O caso de offset exercita o operando do matmul, sem prólogo, para evitar que um
cast materialize um tensor novo com offset zero antes do filtro. Isso não cobre
uma carga com offset dentro de um prólogo fundido.

Esses são alvos, não resultados presumidos. O Inductor pode excluir TMA antes
do filtro, transformar o layout ou recusar a fusão antes da guarda de carga.
Nesse caso, registrar o comportamento e deixar o alvo como **não coberto**.
Não desligar heurísticas nativas para fabricar cobertura nesta rodada.

## Configuração e registros

- Torch 2.13.0, Triton 3.7.1, H100; registrar versões completas e UUID efetivo.
- Dois controles por caso: original e extensão com `declare_b=True`.
- Três sementes (0, 1, 2), comparação contra eager com `atol=0.125`, `rtol=0.01`.
- TMA habilitado como opção. ATen e Triton comum continuam disponíveis.
- Limitar o template Triton comum ao primeiro candidato e TMA a um candidato
  64×128×64, quatro warps, três estágios. Outros candidatos nativos que o
  compilador oferecer continuam disponíveis. Limite de escolhas de fusão: 8.
- A cota é por template e controle, que contém um único matmul. Não depende
  da identidade da lista temporária de escolhas. Uma primeira tentativa recusada
  não libera novas configurações; as alternativas nativas continuam disponíveis.
- Registrar tentativas, candidatos gerados, configurações puladas e a lista
  independente de escolhas efetivamente apresentada ao autotuner.
- Esse conjunto reduzido é um controle diagnóstico; não representa a seleção
  livre padrão e não serve para comparar desempenho.
- Registrar a elegibilidade TMA nativa, decisões do nosso filtro, exceções
  efetivamente levantadas pelas guardas, configuração, wrapper e log de fusão.
- Capturar três chamadas por controle dentro do contexto da compilação, com
  `fail_on_recompile`. Conferir as três saídas fora do profiler.
- Guardar snapshot do experimento e do patch, resultados por caso e manifesto
  incremental em `artifacts/contrato_tma/<timestamp>`.

## Auditoria e interpretação

A auditoria associa kernels a chamadas pela correlação CUDA e confere sua ordem
contra a sequência conjunta de chamadas Triton e externas no wrapper. Um matmul
externo pode executar vários kernels CUDA consecutivos, mas esse grupo deve
aparecer na posição correta: após o cast quando o wrapper executa cast → matmul.
Chamadas externas adjacentes não podem ser distinguidas só por essa sequência;
nesse caso, a auditoria falha em vez de presumir a associação.

`all_executions_passed` exige correção numérica, estrutura compatível com o
wrapper, chamadas correlacionadas, ausência de recompilação e coleta concluída.
Um caso negativo com prólogo não pode resultar em um único kernel estendido
fundido. O template estendido pode continuar aparecendo como matmul **separado**.

`protocol_conformant` confere os contadores, o limite de um candidato por
template, a configuração TMA e a correspondência com as escolhas realmente
oferecidas. Contadores ausentes ou candidatos adicionais impedem a aprovação
do protocolo, mesmo com saídas corretas. A CLI retorna erro nesse caso.

`planned_coverage_complete` exige execuções corretas, conformidade com o protocolo
e os oito alvos da tabela. A auditoria
marca separadamente:

- exclusão TMA nativa antes de nosso filtro;
- rejeição pelo filtro e geração delegada ao template nativo;
- rejeição de uma tentativa de fusão pela guarda de prólogo;
- caminho final identificado no wrapper e executado no trace.

Gerar um candidato nativo não demonstra que ele foi selecionado. A saída correta
de uma alternativa nativa demonstra continuidade naquele caso; não transforma
uma exclusão anterior ao filtro em evidência do ramo de delegação do patch.

Cobertura completa deste protocolo continua limitada a esses casos/configuração.
Ela não prova suporte geral a layouts, dtypes ou expressões arbitrárias.

## Sequência de execução após revisão do diff

Começar somente pelo controle positivo:

```sh
.venv/bin/modal run experiments/integracao/modal_contrato_tma.py --cases positive
python3 scripts/auditoria/verificar_contrato_tma.py artifacts/contrato_tma/RODADA
```

Após conferir a instrumentação, os oito casos podem ser coletados em sequência
com `--cases all`. A coleta para ao primeiro caso com falha e preserva os
arquivos já recebidos. A auditoria com `--require-coverage` também retorna erro
se algum dos oito alvos não foi exercitado.

Cada chamada remota tem timeout de 120 segundos e zero retries. O cliente espera
até 180 segundos pelo resultado e solicita cancelamento com término do container
em timeout, interrupção ou falha ao persistir o manifesto após iniciar a chamada.
Esses limites não incluem a preparação inicial da
imagem antes do entrypoint local. `collection.json` registra o ID do app e da
chamada ativa; uma falha no pedido de cancelamento também fica registrada.

Testes locais, sem iniciar app ou alocar GPU:

```sh
python3 -m unittest discover -s scripts/auditoria -p 'test_contrato_tma.py' -v
```

## Resultado da rodada completa (03/11/2026)

| Caso | O que o filtro viu em A | Decisão / guarda | Estado |
|---|---|---|---|
| `positive` | fp16 256×256, stride 256,1, offset 0 | aceita; extensão no wrapper | coberto |
| `tail_m` | M = 257 | recusa `tail_m`; nativo | coberto |
| `tail_k` | K = 264 | recusa `tail_k`; nativo | coberto |
| `dtype` | bf16 | recusa `dtype`; nativo | coberto |
| `prologue_b` | fp16, cast em B | aceita; guarda `prologue_b` rejeitou uma tentativa | coberto |
| `noncontiguous` | stride 256,1, offset 0 | aceita | **não coberto** |
| `offset` | offset 0 | aceita | **não coberto** |
| `broadcast` | stride 256,1 | aceita; nenhuma guarda disparou | **não coberto** |

O filtro olha as entradas do nó de matmul, já depois de qualquer prólogo; isso
vale também para o `positive`. Por isso um stride ou broadcast colocado antes do
cast não chega ao `stride_a`. Nos três casos não cobertos, a explicação lida nos
`patched_fusion.log` é distinta:

- `noncontiguous`: o scheduler nativo recusou a fusão do cast (`memory deps did
  not match`) e o filtro aceitou o candidato. O log do Modal também registra
  `Layout conflict detected for buf0` (template espera stride `[256, 1]`, layout
  fixado em `[1, 256]`); a relação entre os dois registros não foi investigada.
- `broadcast`: o scheduler nativo recusou a fusão (`prologue fusion that must be
  upcast to fp32 not profitable for low precision templates`), a mesma heurística
  que já barrava ReLU e a fórmula sem marcação. `prologue_rejections` ficou vazio:
  a guarda `prologue_load_layout` nunca foi chamada.
- `offset`: expressão sem prólogo, então nenhuma guarda de prólogo se aplica. O
  filtro leu offset zero; a causa não foi verificada no código do Inductor.

Esses ramos (`stride_a`, `offset_a`, `prologue_load_layout`) seguem sem evidência
de execução. Decidiu-se manter a cobertura em 5 de 8, sem desligar heurísticas
nativas para forçar os demais.

Em `dtype` e `prologue_b`, original e extensão terminaram com kernels diferentes
porque o autotuner escolheu de forma distinta; isso não é comparação causal.
Gerar um candidato nativo não demonstra que ele foi selecionado: a continuidade
é demonstrada pela saída correta do wrapper final.
