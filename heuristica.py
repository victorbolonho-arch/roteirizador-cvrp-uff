"""
GRASP para o problema real: frota heterogênea, vários depósitos, janelas de
tempo e matriz assimétrica.

É a mesma ideia do trabalho 4 — construção gulosa aleatorizada seguida de busca
local, repetida, guardando a melhor —, adaptada ao problema que o aplicativo
resolve:

  CONSTRUÇÃO  inserção mais barata aleatorizada. A cada passo calcula-se, para
              cada cliente ainda sem rota, a inserção viável mais barata (numa
              rota existente ou abrindo um veículo novo), e sorteia-se entre os
              α melhores clientes. É a RCL por cardinalidade, que a calibração
              do trabalho 4 mostrou ser a melhor escolha no CVRP (α = 3).
              Trocou-se o algoritmo das economias pela inserção porque ela lida
              naturalmente com janelas de tempo e com veículos diferentes: cada
              tentativa é só uma avaliação de rota.

  BUSCA LOCAL os movimentos do trabalho 3 — 2-opt intra-rota, troca(0,1) e
              troca(1,1) — mais a troca do tipo de veículo de uma rota. Como a
              matriz é assimétrica e há janelas, cada candidato é conferido pela
              avaliação completa da rota (`modelo.avaliar_rota`), em vez das
              fórmulas de ganho incremental do caso simétrico.
"""

from __future__ import annotations

import random
import time

from modelo import Problema, Rota, Solucao, avaliar_rota

ALFA = 3


def _custo(p: Problema, tipo: int, seq: list[int]) -> float | None:
    return avaliar_rota(p, tipo, seq)


# ---------------------------------------------------------------------------
# Construção
# ---------------------------------------------------------------------------

def _melhor_insercao(p, tipo, seq, custo_atual, c):
    """(acréscimo de custo, posição) da melhor inserção viável de c na rota."""
    melhor = None
    for pos in range(len(seq) + 1):
        novo = _custo(p, tipo, seq[:pos] + [c] + seq[pos:])
        if novo is not None:
            delta = novo - custo_atual
            if melhor is None or delta < melhor[0]:
                melhor = (delta, pos)
    return melhor


def construir(p: Problema, rng: random.Random, alfa: int = ALFA):
    """Constrói uma solução por inserção mais barata aleatorizada."""
    rotas: list[list] = []                 # [tipo, sequência, custo]
    uso = [0] * len(p.tipos)
    pendentes = set(p.clientes)

    # Melhor inserção de cada cliente em cada rota, guardada para não ser
    # recalculada: quando uma rota muda, só a coluna dela é refeita.
    cache: dict[tuple[int, int], tuple[float, int] | None] = {}

    def abrir_veiculo(c):
        """Custo de atender c sozinho no tipo de veículo mais barato disponível."""
        melhor = None
        for k, t in enumerate(p.tipos):
            if uso[k] >= t.quantidade:
                continue
            custo = _custo(p, k, [c])
            if custo is not None and (melhor is None or custo < melhor[0]):
                melhor = (custo, k)
        return melhor

    while pendentes:
        opcoes = []                        # (acréscimo, cliente, ação)
        for c in pendentes:
            melhor = None
            for r, (tipo, seq, custo) in enumerate(rotas):
                chave = (c, r)
                if chave not in cache:
                    cache[chave] = _melhor_insercao(p, tipo, seq, custo, c)
                ins = cache[chave]
                if ins is not None and (melhor is None or ins[0] < melhor[0]):
                    melhor = (ins[0], ("inserir", r, ins[1]))
            novo = abrir_veiculo(c)
            if novo is not None and (melhor is None or novo[0] < melhor[0]):
                melhor = (novo[0], ("abrir", novo[1]))
            if melhor is not None:
                opcoes.append((melhor[0], c, melhor[1]))

        if not opcoes:
            break                          # sobrou cliente que nenhum veículo atende

        opcoes.sort(key=lambda o: o[0])
        _, c, acao = opcoes[rng.randrange(min(alfa, len(opcoes)))]

        if acao[0] == "inserir":
            _, r, pos = acao
            tipo, seq, _ = rotas[r]
            seq = seq[:pos] + [c] + seq[pos:]
            rotas[r] = [tipo, seq, _custo(p, tipo, seq)]
            for chave in [k for k in cache if k[1] == r]:
                del cache[chave]
        else:
            tipo = acao[1]
            uso[tipo] += 1
            rotas.append([tipo, [c], _custo(p, tipo, [c])])
        pendentes.discard(c)

    return [[t, s] for t, s, _ in rotas], sorted(pendentes)


# ---------------------------------------------------------------------------
# Busca local
# ---------------------------------------------------------------------------

def _dois_opt(p, rotas) -> bool:
    """Inverte trechos dentro de cada rota enquanto isso baixar o custo."""
    melhorou = False
    for r in rotas:
        tipo, seq = r
        custo = _custo(p, tipo, seq)
        achou = True
        while achou:
            achou = False
            for i in range(len(seq) - 1):
                for j in range(i + 1, len(seq)):
                    nova = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    c = _custo(p, tipo, nova)
                    if c is not None and c < custo - 1e-9:
                        seq, custo, achou, melhorou = nova, c, True, True
                        break
                if achou:
                    break
        r[1] = seq
    return melhorou


def _troca_0_1(p, rotas) -> bool:
    """Tira um cliente de uma rota e o põe na melhor posição de outra (ou da mesma).

    Se a rota de origem esvazia, o custo fixo do veículo deixa de ser pago —
    é assim que a busca local consegue reduzir o tamanho da frota.
    """
    melhorou = False
    custos = [_custo(p, t, s) for t, s in rotas]
    for a in range(len(rotas)):
        i = 0
        while i < len(rotas[a][1]):
            tipo_a, seq_a = rotas[a]
            c = seq_a[i]
            sem = seq_a[:i] + seq_a[i + 1:]
            custo_sem = _custo(p, tipo_a, sem)
            if custo_sem is None:
                i += 1
                continue
            ganho = custos[a] - custo_sem
            melhor = None
            for b, (tipo_b, seq_b) in enumerate(rotas):
                base = seq_b if b != a else sem
                custo_base = custos[b] if b != a else custo_sem
                for pos in range(len(base) + 1):
                    if b == a and pos == i:
                        continue
                    novo = _custo(p, tipo_b, base[:pos] + [c] + base[pos:])
                    if novo is None:
                        continue
                    # Na mesma rota compara-se direto com o custo atual dela;
                    # em outra, soma-se o que a origem economiza sem o cliente.
                    delta = novo - custos[a] if b == a else (novo - custo_base) - ganho
                    if delta < -1e-9 and (melhor is None or delta < melhor[0]):
                        melhor = (delta, b, pos)
            if melhor is None:
                i += 1
                continue
            _, b, pos = melhor
            if b == a:
                rotas[a][1] = sem[:pos] + [c] + sem[pos:]
            else:
                rotas[a][1] = sem
                rotas[b][1] = rotas[b][1][:pos] + [c] + rotas[b][1][pos:]
                custos[b] = _custo(p, rotas[b][0], rotas[b][1])
            custos[a] = _custo(p, rotas[a][0], rotas[a][1])
            melhorou = True
    return melhorou


def _troca_1_1(p, rotas) -> bool:
    """Duas rotas trocam um cliente cada, na mesma posição."""
    melhorou = False
    custos = [_custo(p, t, s) for t, s in rotas]
    for a in range(len(rotas)):
        for b in range(a + 1, len(rotas)):
            achou = True
            while achou:
                achou = False
                ta, sa = rotas[a]
                tb, sb = rotas[b]
                for i in range(len(sa)):
                    for j in range(len(sb)):
                        na = sa[:i] + [sb[j]] + sa[i + 1:]
                        nb = sb[:j] + [sa[i]] + sb[j + 1:]
                        ca = _custo(p, ta, na)
                        if ca is None:
                            continue
                        cb = _custo(p, tb, nb)
                        if cb is None:
                            continue
                        if ca + cb < custos[a] + custos[b] - 1e-9:
                            rotas[a][1], rotas[b][1] = na, nb
                            custos[a], custos[b] = ca, cb
                            achou = melhorou = True
                            break
                    if achou:
                        break
    return melhorou


def _trocar_veiculo(p, rotas) -> bool:
    """Passa uma rota para um tipo de veículo mais barato que ainda esteja livre."""
    melhorou = False
    uso = [0] * len(p.tipos)
    for t, s in rotas:
        if s:
            uso[t] += 1
    for r in rotas:
        tipo, seq = r
        if not seq:
            continue
        atual = _custo(p, tipo, seq)
        for k, t in enumerate(p.tipos):
            if k == tipo or uso[k] >= t.quantidade:
                continue
            novo = _custo(p, k, seq)
            if novo is not None and novo < atual - 1e-9:
                uso[tipo] -= 1
                uso[k] += 1
                r[0], tipo, atual = k, k, novo
                melhorou = True
    return melhorou


def busca_local(p: Problema, rotas) -> None:
    """Cicla entre os movimentos até nenhum deles melhorar (VND)."""
    for _ in range(50):
        melhorou = _dois_opt(p, rotas)
        melhorou |= _troca_0_1(p, rotas)
        rotas[:] = [r for r in rotas if r[1]]
        melhorou |= _troca_1_1(p, rotas)
        melhorou |= _trocar_veiculo(p, rotas)
        if not melhorou:
            break


# ---------------------------------------------------------------------------
# GRASP
# ---------------------------------------------------------------------------

def grasp(p: Problema, tempo_limite: float = 10.0, max_iteracoes: int = 500,
          alfa: int = ALFA, semente: int = 0) -> Solucao:
    """Repete construção + busca local e devolve a melhor solução vista."""
    rng = random.Random(semente)
    inicio = time.perf_counter()
    melhor_custo, melhor_rotas, melhor_falta = None, None, None
    iteracoes = 0

    while iteracoes < max_iteracoes and time.perf_counter() - inicio < tempo_limite:
        # A primeira iteração é gulosa pura (α = 1), como referência.
        rotas, falta = construir(p, rng, alfa=1 if iteracoes == 0 else alfa)
        busca_local(p, rotas)
        custo = sum(_custo(p, t, s) for t, s in rotas)
        iteracoes += 1

        # Atender mais clientes vale mais que custar menos.
        chave = (len(falta), custo)
        if melhor_custo is None or chave < (len(melhor_falta), melhor_custo):
            melhor_custo, melhor_rotas, melhor_falta = custo, [r[:] for r in rotas], falta

    sol = Solucao(
        rotas=[avaliar_rota(p, t, s, detalhar=True) for t, s in melhor_rotas],
        metodo="GRASP", tempo_s=time.perf_counter() - inicio,
        nao_atendidos=melhor_falta, iteracoes=iteracoes,
    )
    sol.status = (f"{iteracoes} iterações" if not melhor_falta else
                  f"{len(melhor_falta)} cliente(s) sem veículo que os atenda")
    return sol
