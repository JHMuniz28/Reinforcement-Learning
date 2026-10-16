"""Dyna-Q no labirinto do Sutton & Barto (2018), Exemplo 8.1.

Notação da disciplina (PPGI071):
    S_t, A_t, R_{t+1}  -> estado, ação, recompensa (a recompensa de A_t é R_{t+1})
    Q[s, a]            -> estimativa de q(s, a)
    gamma, alpha, epsilon
    n                  -> passos de planejamento por passo real
    modelo[(s, a)]     -> (R_{t+1}, S_{t+1}) já observados (modelo aprendido de p(s', r | s, a))

Com n = 0 o algoritmo é Q-learning puro (linha de base).
"""
import argparse
import os

import numpy as np

# Ações: 0 = cima, 1 = baixo, 2 = esquerda, 3 = direita
ACOES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACOES = len(ACOES)


class Labirinto:
    """Grade 6x9 do livro. Recompensa 0 em todo passo e +1 ao chegar na meta."""

    LINHAS, COLUNAS = 6, 9
    PAREDES = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
    INICIO = (2, 0)
    META = (0, 8)

    def __init__(self):
        self.n_estados = self.LINHAS * self.COLUNAS

    def codifica(self, pos):
        return pos[0] * self.COLUNAS + pos[1]

    def reset(self):
        self.pos = self.INICIO
        return self.codifica(self.pos)

    def passo(self, s, a):
        """Dinâmica determinística: devolve (R_{t+1}, S_{t+1}, terminou)."""
        linha, coluna = divmod(s, self.COLUNAS)
        dl, dc = ACOES[a]
        nova = (linha + dl, coluna + dc)
        dentro = 0 <= nova[0] < self.LINHAS and 0 <= nova[1] < self.COLUNAS
        if not dentro or nova in self.PAREDES:
            nova = (linha, coluna)  # bateu: fica parado
        terminou = nova == self.META
        return (1.0 if terminou else 0.0), self.codifica(nova), terminou


def escolhe_acao(Q, s, epsilon, rng):
    """Política epsilon-gulosa; desempate aleatório entre os máximos."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACOES))
    melhores = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(melhores))


def dyna_q(n, alpha, epsilon, gamma, episodios, seed, max_passos=5000):
    """Treina um agente Dyna-Q. Devolve o nº de passos de cada episódio."""
    rng = np.random.default_rng(seed)
    amb = Labirinto()
    Q = np.zeros((amb.n_estados, N_ACOES))
    modelo = {}  # (s, a) -> (R_{t+1}, S_{t+1}, terminou)
    passos_por_episodio = []

    for _ in range(episodios):
        S_t = amb.reset()
        passos = 0
        terminou = False
        while not terminou and passos < max_passos:
            A_t = escolhe_acao(Q, S_t, epsilon, rng)
            R_t1, S_t1, terminou = amb.passo(S_t, A_t)

            # (d) aprendizado direto: regra do Q-learning com a transição real
            alvo = R_t1 + (0.0 if terminou else gamma * Q[S_t1].max())
            Q[S_t, A_t] += alpha * (alvo - Q[S_t, A_t])

            # (e) atualiza o modelo com o que foi observado
            modelo[(S_t, A_t)] = (R_t1, S_t1, terminou)

            # (f) planejamento: n atualizações com transições simuladas
            if n > 0:
                chaves = list(modelo.keys())
                for idx in rng.integers(len(chaves), size=n):
                    s, a = chaves[idx]
                    r, s_prox, fim = modelo[(s, a)]
                    alvo_sim = r + (0.0 if fim else gamma * Q[s_prox].max())
                    Q[s, a] += alpha * (alvo_sim - Q[s, a])

            S_t = S_t1
            passos += 1
        passos_por_episodio.append(passos)
    return passos_por_episodio


def main():
    p = argparse.ArgumentParser(
        description="Dyna-Q no labirinto do Sutton & Barto (Ex. 8.1): compara valores de n.")
    p.add_argument("--n", type=int, nargs="+", default=[0, 5, 50],
                   help="passos de planejamento por passo real (0 = Q-learning puro)")
    p.add_argument("--alpha", type=float, default=0.1, help="taxa de aprendizado")
    p.add_argument("--epsilon", type=float, default=0.1, help="exploração da política epsilon-gulosa")
    p.add_argument("--gamma", type=float, default=0.95, help="fator de desconto")
    p.add_argument("--episodios", type=int, default=50, help="episódios por execução")
    p.add_argument("--sementes", type=int, default=10, help="nº de sementes (mínimo recomendado: 5)")
    p.add_argument("--semente-base", type=int, default=0, help="primeira semente; as demais são consecutivas")
    p.add_argument("--rapido", action="store_true", help="modo reduzido: 5 sementes e 30 episódios")
    p.add_argument("--saida", default="figuras/dyna_q_curvas.png", help="caminho da figura gerada")
    # Passando uma lista vazia para evitar ler os parâmetros do kernel do Colab
    args = p.parse_args([])

    if args.rapido:
        args.sementes, args.episodios = 5, 30

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, args.episodios + 1)
    for n in args.n:
        runs = np.array([
            dyna_q(n, args.alpha, args.epsilon, args.gamma, args.episodios, args.semente_base + k)
            for k in range(args.sementes)
        ])
        media, dp = runs.mean(axis=0), runs.std(axis=0)
        rotulo = f"n = {n}" + (" (Q-learning)" if n == 0 else "")
        ax.plot(x, media, label=rotulo)
        ax.fill_between(x, media - dp, media + dp, alpha=0.2)
        print(f"n={n:>3}: episódio 2 -> {media[1]:7.1f} passos | "
              f"últimos 10 episódios -> {media[-10:].mean():6.1f} passos")

    ax.set_xlabel("episódio")
    ax.set_ylabel("passos até a meta")
    ax.set_ylim(0, 800)
    ax.set_title(f"Dyna-Q no labirinto (média ± desvio, {args.sementes} sementes)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    fig.savefig(args.saida, dpi=150)
    print(f"figura salva em {args.saida}")


if __name__ == "__main__":
    main()
