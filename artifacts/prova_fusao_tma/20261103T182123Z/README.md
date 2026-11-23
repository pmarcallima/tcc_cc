# Prova controlada de fusão TMA

[Verificador da coleta](../../../scripts/analise/provar_fusao_tma.py).

Abra `proof_viewer.html` para comparar os três controles dos dois formatos.
Cada arquivo `*_runtime_trace.json` também abre no Perfetto. A linha do tempo
mostra eventos reais; cada linha é uma execução distinta alinhada ao primeiro
kernel. Os tempos sob profiler não são um benchmark de desempenho.

## Resultado

- Original: cast separado e matmul TMA, dois kernels por chamada.
- Extensão com fusão desligada: cast separado e matmul TMA, dois kernels.
- Extensão com fusão ligada: cast dentro do matmul TMA, um kernel.
- Seis controles auditados, 30 chamadas capturadas e todas as saídas conferidas.
- Dezoito comparações contra eager com três sementes, todas corretas.
- TTIR liga carga int8 por descritor, conversão e dot; PTX confirma TMA/WGMMA.

`proof_verification.json` contém os testes e caminhos exatos de cada evidência.
`results.json` contém os resultados da coleta e o UUID da GPU.
`proof_ir/` preserva dumps por formato e controle; `ir/` contém todos os dumps
exportados pelo job. `experiment_snapshot.py` e `patch_snapshot.py` preservam
os fontes usados. Os seis wrappers e seus logs de fusão ficam nesta pasta.

## Auditoria local

```sh
python3 scripts/analise/provar_fusao_tma.py artifacts/prova_fusao_tma/20261103T182123Z
```

A rotina falha se qualquer controle divergir em estrutura, decisão do scheduler,
correção, nomes/ordem no trace ou presença das instruções TMA/WGMMA. Para os
fundidos, também verifica o fluxo de conversão para o dot e a ausência de Q.

A nova opção de coleta exigiu corrigir dois problemas no script histórico:
resolução do caminho quando importado em `/root` no container e serialização de
`TorchVersion` para leitura local sem instalar torch. As tentativas preliminares
não integram esta pasta. A implementação do kernel não foi alterada.
