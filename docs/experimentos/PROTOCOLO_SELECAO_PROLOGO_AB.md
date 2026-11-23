# Seleção livre: a extensão TMA concorre na fusão do prólogo?

Protocolo definido em 03/11/2026, antes da coleta. Motivação na
triagem da aplicação.

## Pergunta

Na seleção livre (max-autotune, TMA persistente habilitado como opção), a extensão
TMA chega a ser medida como kernel fundido? Se for medida, ganha ou perde?

**Hipótese a testar.** A extensão declara só A como entrada de prólogo
(`allowed_prologue_inps = {A}`). O Triton comum declara A e B. Na fusão com
benchmark, o scheduler descarta a escolha cujo conjunto difere da união
(`choice_supports_fusion`, `scheduler.py:5429-5444` das fontes preservadas).
Se for assim, a extensão nunca é medida fundida, e a afirmação de que o Triton
comum "venceu os candidatos TMA" está imprecisa.

Outros filtros do mesmo laço (`scheduler.py:5518-5528`) também podem excluir uma
escolha. O laço percorre as escolhas em ordem de tempo sem fusão e para quando
`tempo ≥ ms1 + ms2`, ou quando o número de escolhas aceitas passa de
`max_epilogue_benchmarked_choices`. Os registros precisam separar essas causas.

## Braços

0. **Patch atual, só instrumentado.** Formas 256³ e (2048,1024,256), as mesmas da
   rodada `artifacts/free_tma`. Confirma ou refuta a hipótese.
1. **Patch declara {A, B}, com guardas.** Usa as 7 formas do
   [protocolo dos quatro caminhos](PROTOCOLO_QUATRO_CAMINHOS.md): 256³, 1024³,
   2048³, 4096³, (1024,256,1024), (1024,512,1024) e (1024,4096,1024).
   - Prólogo em B: o template não o implementa e levanta `CantSplit`.
   - Carga fora do contrato em A: passa a levantar `CantSplit`, que o scheduler
     trata como "pular esta escolha".
2. **Diagnóstico fora do padrão.** Igual ao braço 1, mas com
   `max_epilogue_benchmarked_choices` elevado. Esse braço serve só para
   interpretar o efeito do limite e não descreve o comportamento padrão.

**Casos negativos**, no braço 1:
- `x @ w_int8.to(fp16)`: prólogo em B.
- `x_t.t().to(fp16) @ w`: indexação de A fora da identidade.

O esperado é que a compilação não aborte e que o resultado seja correto.

Prólogo dos casos positivos: cast `int8 → fp16`, sem InvokeQuant, com entradas
iguais às da rodada anterior (inteiros de -8 a 8; peso `randn*0.125`).

## Registros

Para cada caso:
- as escolhas e seus tempos sem fusão;
- os conjuntos `allowed_prologue_inps` de cada escolha e da união;
- quais escolhas foram compiladas e medidas fundidas, com os respectivos tempos;
- o valor de `max_epilogue_benchmarked_choices`;
- o kernel final no wrapper;
- a correção (`atol=0.125`, `rtol=0.01`, sementes 0, 1 e 2);
- as exceções, com o traceback.

Uma exceção em um caso não interrompe os demais.

## Critério de interpretação, definido antes da coleta

- **Braço 0: TMA com `{A}` diferente da união `{A,B}` e nunca medido fundido.**
  A hipótese é confirmada. Corrigir a frase e a seção de seleção livre do artigo.
- **Braço 1: TMA medido fundido e mais lento.** "Concorreu e perdeu" passa a ser
  um resultado medido.
- **Braço 1: TMA medido fundido e escolhido.** A extensão é usada sob seleção
  real. Reportar com destaque e com o kernel no wrapper como evidência.
- **Braço 1: TMA ainda não medido.** Reportar qual filtro o excluiu (tempo sem
  fusão ou limite de escolhas). Isso é um segundo limite da etapa de seleção.

Uma sessão, em uma H100. O objeto é o comportamento de seleção; latência é registrada
mas não é usada para conclusões fora do que está descrito acima. Não acrescentar
formas além das listadas depois de ver resultados.
