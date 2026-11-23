# Fusão de prólogos no TorchInductor

Código e dados experimentais do TCC de Pedro M. Lima. O trabalho investiga como o TorchInductor pode converter ou transformar uma matriz dentro do mesmo kernel que a multiplica, mantendo as cargas por Tensor Memory Accelerator (TMA) na NVIDIA H100.

[Ler o artigo completo (PDF)](docs/artigo.pdf)

## O que cada experimento mostra

| Experimento | Pergunta e resultado | Evidência |
|---|---|---|
| [Protótipo manual](src/tma_prologue_prototype.py) | É possível transformar a entrada e multiplicar as matrizes no mesmo kernel usando TMA? O protótipo fez isso na H100. Nos seis formatos testados, as saídas fundida e não fundida coincidiram e passaram na comparação com a referência. | [Resultados por formato](artifacts/tma_prototype/summary.csv) |
| [Extensão do TorchInductor](src/inductor_tma_prologue_patch.py) | O compilador consegue gerar a fusão para `X.to(torch.float16) @ W`? Com o template TMA fixado, o registro de execução mostrou dois kernels com a fusão desligada e um com ela ligada. O código gerado mostra o valor convertido chegando ao produto matricial. | [Prova da fusão](artifacts/prova_fusao_tma/20261103T182123Z/README.md) |
| [Quatro caminhos manuais](src/four_path_benchmark.py) | Quanto muda o tempo de execução ao fundir o prólogo? A comparação é feita dentro de cada família de cargas (convencionais ou TMA). O efeito variou com a expressão, o formato e a configuração. Ao comparar famílias, também podem mudar as instruções que o Triton gera. | [Tabela de tempos](artifacts/four_paths/20261103T131925406353Z/summary.csv) |
| [Seleção pelo compilador](experiments/integracao/modal_inductor_free_selection.py) | Qual implementação o Inductor escolhe quando pode comparar alternativas? Com `max-autotune`, nos dois formatos de cast avaliados, escolheu o Triton comum já fundido. | [Resultados da seleção](artifacts/free_tma/results.json) |
| [Limites da extensão](experiments/integracao/modal_contrato_tma.py) | O que acontece com entradas fora do domínio suportado? As 16 execuções produziram saídas corretas. Cinco dos oito casos atingiram a verificação interna prevista. A cobertura de entrada não contígua, deslocamento e broadcast segue em aberto. | [Cobertura dos oito casos](artifacts/contrato_tma/20261103T220051Z/README.md) |

Para conferir a fusão do cast usando a coleta salva na H100, execute na raiz do repositório:

```sh
python3 scripts/analise/provar_fusao_tma.py artifacts/prova_fusao_tma/20261103T182123Z
```

O verificador exige uma ocorrência de cada formato, `(256, 256, 256)` e
`(2048, 1024, 256)`, e confere os três controles de cada um. O relatório
`proof_verification.json` mostra quais verificações passaram. Após uma auditoria
bem-sucedida, `proof_viewer.html` permite examinar as chamadas registradas.
Ambos ficam na pasta da coleta.

## Reproduzir a análise local

Na raiz do clone, com Python 3 e sua biblioteca padrão, é possível verificar as evidências registradas na GPU e recalcular as tabelas de desempenho.

```sh
python3 -m unittest discover -s scripts/auditoria
python3 scripts/auditoria/verificar_integracao_tma.py
python3 scripts/auditoria/verificar_prologues_comuns.py
python3 scripts/auditoria/verificar_selecao_livre.py
python3 scripts/auditoria/verificar_contrato_tma.py artifacts/contrato_tma/20261103T220051Z
python3 scripts/analise/analisar_quatro_caminhos.py artifacts/four_paths/20261103T131925406353Z --latex-dir /tmp/tcc-cc-tabelas
```

Os arquivos em [src](src) contêm as implementações. Os [experimentos](experiments) geram as coletas e os [protocolos](docs/experimentos) descrevem as condições de comparação. As coletas em [artifacts](artifacts) preservam as medições, o código gerado e os registros de execução usados pelos verificadores. Os [trechos comentados do PyTorch](references/evidencia) identificam o ponto em que o template original deixa de aceitar o prólogo.

A [execução principal da seleção instrumentada](artifacts/selecao_prologo_ab/20261103T170039Z/arm2/results.json) registra os tempos internos do compilador. A [coleta complementar](artifacts/selecao_prologo_ab/20261103T175152Z) inclui os traces. Os snapshots nas pastas das coletas correspondem às versões usadas em cada execução.

## Repetir a prova controlada em GPU

O comando abaixo usa uma H100 paga em uma conta Modal autenticada. O próprio experimento prepara a imagem com PyTorch e Triton. Os comandos de auditoria acima usam apenas os dados locais.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-benchmarks.txt
.venv/bin/modal setup
.venv/bin/modal run experiments/integracao/modal_inductor_plain_prologues.py --suite cast --proof-trace
```
