# Dyna-Q: planejamento com um modelo aprendido

> **RASCUNHO (avanço).** Seções marcadas com `[A FAZER]` ainda precisam ser escritas ou revisadas. O objetivo final é entre 1500 e 3000 palavras.

## 1. Motivação

Em MDPs, a aula 03 mostrou que, se conhecemos a dinâmica $p(s', r \mid s, a)$, dá para calcular valores por varreduras (iteração de valor). O problema é que, na prática, quase nunca conhecemos $p$. O Q-learning resolve isso aprendendo só pela experiência, sem modelo, mas ele tem um custo: cada transição real $(S_t, A_t, R_{t+1}, S_{t+1})$ serve para **uma** atualização de $Q$ e depois é descartada.

Isso é caro quando interagir com o ambiente custa tempo ou desgaste. A aula 01 já aponta a saída: "com modelo dá para planejar; sem modelo, resta tentativa e erro". O Dyna-Q junta as duas coisas. O agente aprende pela experiência real **e** guarda um modelo do que viu, usando esse modelo para fazer atualizações extras de $Q$ sem tocar o ambiente.

No labirinto deste material, o efeito é visível: no segundo episódio, o Q-learning ainda anda em média cerca de 600 passos, enquanto o Dyna-Q com $n = 50$ já chega à meta em cerca de 40.

## 2. Onde isto se encaixa

`[A FAZER: revisar e ampliar com frases de justificativa]`

1. **Classificação.**
   - Tabular (um $Q[s, a]$ por par estado-ação).
   - *Model-based* no sentido do planejamento, mas o modelo é **aprendido**; a atualização de $Q$ em si é *model-free*.
   - *Off-policy*: o alvo usa $\max_a$, e o comportamento é $\varepsilon$-guloso.
   - Controle (estima $q_*$ para achar a política), não só predição.
   - Tarefa episódica.
2. **Genealogia.** Descende do Q-learning (aula 05 / cap. 6). A regra de atualização de $Q$ é a mesma; o que muda é de **onde vêm as transições** usadas.

   Q-learning:
   $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \big]$$

   Dyna-Q: a mesma regra, aplicada $n$ vezes a mais por passo real a pares $(s, a)$ sorteados do modelo, com $(r, s')$ lidos do modelo em vez do ambiente:
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big]$$

   Com $n = 0$ o Dyna-Q **é** o Q-learning.
3. **Símbolos.** Seguem o glossário da disciplina. Os que o glossário ainda não traz:

   | Símbolo | No código | Significado |
   |---|---|---|
   | $\alpha$ | `alpha` | taxa de aprendizado (passo constante, aula 02) |
   | $\varepsilon$ | `epsilon` | probabilidade de explorar (aula 02) |
   | $n$ | `n` | passos de planejamento por passo real |
   | $\text{Model}(s, a)$ | `modelo[(s, a)]` | último $(r, s')$ observado; aproximação de $p(s', r \mid s, a)$ |

4. **Hipóteses.**
   - O modelo guardado só é correto se o ambiente for **determinístico** (a última transição observada vale para sempre). É o caso do labirinto.
   - Estados e ações finitos e pequenos, para caber em tabela.
   - $\gamma < 1$ ou tarefa episódica (aqui, episódica com $\gamma = 0{,}95$).
   - Se o ambiente **mudar**, o modelo fica desatualizado e o planejamento passa a reforçar informação falsa (ver Limites).

## 3. O método

`[A FAZER: escrever a ideia em palavras e depois a derivação passo a passo]`

A ideia em palavras: depois de cada passo real, o agente (i) atualiza $Q$ com a transição que acabou de ver, (ii) anota essa transição no modelo e (iii) repete $n$ vezes: sorteia um par $(s, a)$ que já visitou, consulta o modelo para saber o que aconteceu naquela vez e atualiza $Q$ como se tivesse acabado de acontecer. As atualizações (iii) são "experiência imaginada", barata porque não toca o ambiente.

## 4. Pseudocódigo

```
Dyna-Q tabular (n passos de planejamento)

Parâmetros: α ∈ (0, 1], ε > 0, γ ∈ [0, 1], n ≥ 0
Inicialize Q[s, a] = 0 para todo s, a
Inicialize Model(s, a) como vazio

Repita para cada episódio:
    S ← estado inicial
    Repita até S ser terminal:
        A ← ação ε-gulosa em relação a Q[S, ·]
        Execute A; observe R e S'
        Q[S, A] ← Q[S, A] + α [ R + γ max_a Q[S', a] − Q[S, A] ]    (aprendizado direto)
        Model(S, A) ← (R, S')                                        (atualiza o modelo)
        Repita n vezes:                                              (planejamento)
            s ← par de estado já visitado, sorteado
            a ← ação já tomada em s, sorteada
            (r, s') ← Model(s, a)
            Q[s, a] ← Q[s, a] + α [ r + γ max_a' Q[s', a'] − Q[s, a] ]
        S ← S'
```

Em estados terminais, o termo $\max_a Q$ vale 0.

## 5. Implementação

`[A FAZER: explicar as decisões que o pseudocódigo esconde]`

Arquivo: `dyna_q.py`. Detalhes a descrever:

- Labirinto 6×9 do livro, com paredes e meta fixos; recompensa 0 em todo passo e +1 ao chegar na meta; $Q$ é um array `Q[s, a]` de forma $(54, 4)$.
- O modelo é um dicionário `(s, a) → (R_{t+1}, S_{t+1}, terminou)`. Como o ambiente é determinístico, guardar só a última transição basta.
- Desempate aleatório no `argmax` (sem isso, com $Q$ zerado o agente sempre escolheria a mesma ação).
- Hiperparâmetros: $\alpha = 0{,}1$, $\varepsilon = 0{,}1$, $\gamma = 0{,}95$, $n \in \{0, 5, 50\}$.
- `[A FAZER: o que quebra se for mudado, por exemplo $\varepsilon = 0$, $\alpha$ grande, $n$ muito alto]`

## 6. Experimento

`[A FAZER: inserir figura final e discutir]`

Comando: `python dyna_q.py` (ver `scripts/README.md`). Figura: `figuras/dyna_q_curvas.png`.

Resultado das 10 sementes (média):

| $n$ | passos no episódio 2 | passos nos últimos 10 episódios |
|---|---|---|
| 0 (Q-learning, linha de base) | ≈ 605 | ≈ 17,5 |
| 5 | ≈ 186 | ≈ 16,4 |
| 50 | ≈ 40 | ≈ 17,5 |

O que a figura mostra: quanto maior $n$, mais rápido a curva cai, e todas convergem para o mesmo patamar (~17 passos; o caminho mais curto tem 14, e os passos extras vêm da exploração $\varepsilon = 0{,}1$).

`[A FAZER: explicar POR QUE esse é o resultado esperado. Rascunho da explicação: no Q-learning, a informação da meta (+1) anda um estado para trás por episódio. No Dyna-Q, o planejamento repete as transições já vistas e propaga a informação por vários estados de uma vez.]`

## 7. Limites

`[A FAZER]`

- Modelo guardado supõe ambiente determinístico e estacionário; se o labirinto mudar, o modelo mente (próximo tema: Dyna-Q+, tema A8).
- O planejamento sorteia pares uniformemente, inclusive os que não importam; priorizar as atualizações (*prioritized sweeping*, tema A9) usa melhor o mesmo orçamento.
- Só funciona em tabela; com estados grandes é preciso aproximação de função.

## 8. Referências

- SUTTON, R. S.; BARTO, A. G. *Reinforcement Learning: An Introduction*. 2. ed. MIT Press, 2018. Cap. 8, seções 8.1 a 8.3 (labirinto do Exemplo 8.1 e Figura 8.2).
- Notas e slides das aulas 01 a 03 da disciplina PPGI071 (glossário e notação).
- `[A FAZER: conferir se há outras fontes usadas]`
