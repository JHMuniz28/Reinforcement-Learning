# Scripts — Dyna-Q no labirinto (Sutton & Barto, Exemplo 8.1)

## Instalação

```bash
pip install -r requirements.txt
```

## Comando exato

Experimento completo (10 sementes, 50 episódios, n = 0, 5, 50). Leva poucos segundos em CPU:

```bash
python dyna_q.py
```

Modo reduzido (5 sementes, 30 episódios):

```bash
python dyna_q.py --rapido
```

A figura é salva em `figuras/dyna_q_curvas.png` (mude com `--saida`).

## Parâmetros principais

| Opção | Padrão | Significado |
|---|---|---|
| `--n` | `0 5 50` | passos de planejamento por passo real (`0` = Q-learning puro) |
| `--alpha` | `0.1` | taxa de aprendizado $\alpha$ |
| `--epsilon` | `0.1` | exploração $\varepsilon$ |
| `--gamma` | `0.95` | desconto $\gamma$ |
| `--episodios` | `50` | episódios por execução |
| `--sementes` | `10` | número de sementes (o mínimo para experimentos tabulares é 5) |
| `--semente-base` | `0` | primeira semente; as demais são consecutivas |

## O que esperar de cada execução

Com os valores padrão, a saída deve ser parecida com esta (variam um pouco com a semente):

```
n=  0: episódio 2 ->   605.0 passos | últimos 10 episódios ->   17.5 passos
n=  5: episódio 2 ->   186.3 passos | últimos 10 episódios ->   16.4 passos
n= 50: episódio 2 ->    40.0 passos | últimos 10 episódios ->   17.5 passos
```

- No primeiro episódio todos andam ao acaso (centenas de passos).
- No segundo episódio, quanto maior o $n$, menos passos: com $n = 50$ o agente já está perto do caminho curto.
- Depois de uns 30 episódios as três curvas se encontram perto de 17 passos. O caminho mais curto do labirinto tem 14 passos, e os passos a mais vêm da exploração $\varepsilon = 0{,}1$.
- A curva de $n = 0$ (Q-learning) tem faixa de variação muito maior entre sementes.

Se a curva de $n = 50$ não cair para algo próximo de 40 passos no episódio 2, algo está errado.
