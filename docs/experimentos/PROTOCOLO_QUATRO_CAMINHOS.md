# Comparação na H100: cargas comuns/TMA e prólogo separado/fundido

Protocolo definido em 03/11/2026 antes da nova coleta. Coleta concluída e auditada
em três sessões; resultados em [tabela de resultados](../../artifacts/four_paths/20261103T131925406353Z/summary.csv).

## Perguntas e comparadores

1. Quanto a fusão altera a latência em cada família de cargas?
2. Essa conclusão muda quando cada caminho seleciona sua configuração?
3. Com M/K fixos, ampliar N muda o custo relativo da fusão, consistente com
   a repetição do prólogo para tiles de colunas?

Os quatro caminhos são `std_split`, `std_fused`, `tma_split`, `tma_fused`.
Compartilham um fonte de matmul persistente, atribuição de tiles e contrato
numérico. O separado inclui prólogo + matmul; o fundido inclui ambos num kernel.
Ambas as famílias têm grid limitado ao mínimo entre tiles e SMs.

Isso é um controle manual. Não mede a decisão automática do Inductor, nem isola
TMA como única diferença de instruções: o lowering e o caminho de operandos podem
mudar junto. Não constitui comparação entre GPUs.

## Domínio pré-definido

- Operações: cast `int8 → fp16`; fórmula `x*scale+(scale-1)`.
- Fórmula: aritmética fp32, sem FMA, cast único fp16; dot acumula fp32 e grava fp16.
- Referência: Q com esse contrato e matmul fp32, TF32 desligado, saída fp16.
- Entradas contíguas; peso fp16 `randn*0.125`, escala fp16 `randn*0.25+1`;
  cast com inteiros entre -8 e 8, inclusive. Mesmas entradas nos quatro caminhos.
- Formatos M/N/K: 256³, 1024³, 2048³, 4096³; mais (1024,256,1024),
  (1024,512,1024) e (1024,4096,1024). O 1024³ completa a varredura de N.
- Quatro configurações BM/BN/BK/estágios: 64/128/64/2 (controle fixo),
  64/128/64/3, 128/128/64/3, 64/256/64/3; quatro warps em todas.
- Todas as formas são divisíveis pelos blocos; cobertura de caudas não é o objeto.

## Execução e evidências

1. Smoke: dois prólogos no formato 256³ e na configuração fixa.
2. Coleta completa: três sessões sequenciais, cada uma em container de uso único.
   Registrar UUID da GPU; containers distintos podem usar a mesma GPU física.
3. Descoberta: semente 0; três medidas CUDA Graphs, 15 ms cada, ordem aleatória.
   Selecionar o menor tempo mediano por caminho, sem procurar apenas casos favoráveis.
4. Reavaliar controle fixo e finalistas com semente 99: nove medidas intercaladas,
   30 ms cada. Validar finalistas também nas sementes 1 e 2; descoberta já valida 0.
5. Correção: `atol=0.125`, `rtol=0.01`, erro máximo e relativo L2; conferir Q exato
   nos caminhos separados. Falha numérica invalida a sessão; não remover o caso.
6. Preservar TTIR/TTGIR/PTX de todos os candidatos válidos, registradores, spills,
   shared, grid e número de tiles. Exigir instrução TMA nos caminhos TMA e ausência
   dela nos comuns. Não identificar TMA apenas por flags ou WGMMA.
7. Contar eventos do profiler em cinco chamadas de cada finalista: dez eventos
   nos separados e cinco nos fundidos. Perfilamento fica fora da medição principal.
8. Arquivar fontes com SHA-256, protocolo, resultados e IR em diretório exclusivo.
   Não sobrescrever coletas antigas. Erros de compilação/recursos ficam registrados;
   falha de qualquer caminho do controle fixo impede a comparação fatorial completa.

## Análise e limites

Apresentar por sessão as medianas e amostras; para síntese entre sessões, usar
mediana e faixa das medianas, sem tratá-las como intervalo de confiança.
Calcular redução de latência `100*(1-t_f/t_s)` com o par da mesma sessão.
Separar configuração fixa de vencedores da busca limitada; não chamar de ótimo.
Não agrupar réplicas CUDA Graphs como se fossem GPUs ou sessões independentes.

O controle de N mantém o tamanho de Q, mas também altera FLOPs, tiles, reuso e
pressão nos caches. Logo, uma tendência não isola causalmente recomputação.
Cast e fórmula diferem também em dtype e quantidade de entradas; seu contraste
não é uma ablação exclusiva do custo aritmético do prólogo.
Contadores de recursos não demonstram, isoladamente, causa de regressão.
Descritores, buffers e compilação ficam fora do tempo: resultado é replay GPU,
não latência Python. O harness pré-aloca Q para os controles separados; os kernels
fundidos não o acessam. Não se mede redução de alocação ou pico de memória.
Clocks não fixados; não há controle de interferência externa.
Sem coleta NCU nova, não concluir redução efetiva de DRAM, ocupação ou atividade
tensor a partir dos metadados. Não há especialização explícita de warps.

## Reprodução

Cliente local utilizado nesta preparação: Modal 1.6.3, Python 3.14. O requisito
histórico `modal==1.5.4` não estava disponível no índice consultado. Isso não altera
as imagens históricas; a nova imagem mantém Torch 2.13.0/Triton 3.7.1.

```sh
.venv/bin/modal token new
.venv/bin/modal run experiments/tma/modal_four_paths.py --smoke --sessions 1
.venv/bin/modal run experiments/tma/modal_four_paths.py --sessions 3
.venv/bin/python scripts/analise/analisar_quatro_caminhos.py artifacts/four_paths/EXECUCAO
```

Executar a coleta completa somente após sucesso do smoke. A GPU Modal é paga;
o protocolo limita a três sessões e timeout de 1.500 s por sessão. Tempos internos
não são estimativa ou comprovação de custo financeiro. O resumo rejeita smoke,
sessões incompletas e evidências inconsistentes para não promover diagnóstico a resultado.
