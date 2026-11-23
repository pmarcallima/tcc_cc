# Evidência de fonte — gap de *prologue fusion* no template TMA

Pacote Tier 0 do plano: prova, **direto do código do PyTorch** (sem GPU), que a
fusão de prologue (op elementwise antes do matmul: `x+1`, cast, dequant) funde no
template Triton padrão mas **não** funde no template persistent+TMA usado em
Hopper (H100) / Blackwell.

Versão inspecionada: PyTorch instalado em
`.venv/lib/python3.14/site-packages/torch/_inductor/`.

## A cadeia do argumento (ler nesta ordem)

| Arquivo | Mostra |
|---|---|
| `01_template_padrao.txt` | template padrão carrega input via macro `load_input` → ponto de injeção do prologue |
| `02_template_tma.txt` | template TMA carrega via `_experimental_descriptor_load` (hardware) → **sem** ponto de injeção |
| `03_scheduler_todo.txt` | o TODO oficial + o `continue` que pula a fusão na choice TMA |
| `04_mm_templates.txt` | padrão define `prologue_loads_all_inputs=True`; TMA não |
| `05_load_input_hook.txt` | `load_input` abre o subgrafo `<LOAD_INPUT_...>` onde o prologue é gerado |
| `06_tma_hardware_gate.txt` | TMA exige compute capability ≥ 9.0 → só Hopper+ (por isso precisa de H100) |

## Resumo em uma frase

Prologue fusion = injetar o op elementwise dentro da carga do operando. O template
padrão carrega com uma macro do Inductor (`load_input`) que abre um subgrafo
controlável → o op entra ali. O template TMA carrega com uma instrução de hardware
(TMA: copia global→shared direto, sem `tl.load` por thread) → não há subgrafo, não
há onde injetar → o op fica em kernel separado materializando buffer na DRAM. Os
próprios devs do PyTorch marcam isso com um TODO em `scheduler.py:4161`.
