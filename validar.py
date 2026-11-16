"""
Confere que o roteirizador está correto.

O teste central compara o VRPSolverEasy com FORÇA BRUTA: em instâncias
pequenas dá para enumerar todas as formas de dividir os clientes em rotas e
todas as ordens de visita, e o ótimo verdadeiro tem que bater com o que o
solver diz ser ótimo. Isso é feito com matriz assimétrica, janelas de tempo e
rota aberta — justamente as partes em que a modelagem teve que contornar
comportamentos da biblioteca (ver solver.py).

Uso:
    python validar.py            # testes de modelo, heurística e solver
    python validar.py --app      # inclui o fluxo completo da interface
"""

from __future__ import annotations

import itertools
import random
import sys
from functools import lru_cache

import numpy as np

import heuristica
import solver
from modelo import HORIZONTE, Problema, TipoVeiculo, avaliar_rota, verificar

falhas: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    print(f"  [{'ok  ' if condicao else 'FALHA'}] {descricao}")
    if not condicao:
        falhas.append(descricao)


def instancia(n_clientes: int, semente: int, assimetrica=True, janelas=False,
              retornar=True, tipos=None) -> Problema:
    rng = np.random.default_rng(semente)
    n = n_clientes + 1
    xy = rng.uniform(0, 10, size=(n, 2))
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    if assimetrica:
        dist = dist * (1 + 0.3 * rng.random((n, n)))      # ida != volta
        np.fill_diagonal(dist, 0)
    tempo = dist * 3.0
    jini, jfim = np.zeros(n), np.full(n, HORIZONTE)
    if janelas:
        for c in range(1, n):
            jini[c] = rng.choice([0, 30])
            jfim[c] = jini[c] + rng.choice([40, 80, 200])
    return Problema(
        nomes=[f"p{i}" for i in range(n)], depositos=[0], clientes=list(range(1, n)),
        demanda=np.array([0] + list(rng.integers(1, 5, n_clientes))),
        servico=np.full(n, 2.0), janela_ini=jini, janela_fim=jfim, dist=dist, tempo=tempo,
        tipos=tipos or [TipoVeiculo("V", 8, 3, 10.0, 1.0, 0.0, 0, 0.0, HORIZONTE)],
        retornar=retornar, usar_janelas=janelas)


def otimo_forca_bruta(p: Problema) -> float:
    """Ótimo exato por enumeração completa (só para instâncias minúsculas).

    Para cada subconjunto de clientes, a melhor rota é a melhor ordem entre
    todas as permutações; depois, programação dinâmica escolhe a partição em
    rotas mais barata respeitando o número de veículos. Um tipo de veículo só.
    """
    clientes = p.clientes
    k = len(clientes)
    melhor_rota = {}
    for mascara in range(1, 1 << k):
        membros = [clientes[i] for i in range(k) if mascara >> i & 1]
        custos = [avaliar_rota(p, 0, list(perm)) for perm in itertools.permutations(membros)]
        viaveis = [c for c in custos if c is not None]
        melhor_rota[mascara] = min(viaveis) if viaveis else None

    @lru_cache(maxsize=None)
    def resolver(restante: int, veiculos: int) -> float:
        if restante == 0:
            return 0.0
        if veiculos == 0:
            return float("inf")
        primeiro = restante & -restante          # o menor cliente fica sempre na 1ª rota
        melhor = float("inf")
        sub = restante
        while sub:
            if sub & primeiro and melhor_rota[sub] is not None:
                melhor = min(melhor, melhor_rota[sub] + resolver(restante & ~sub, veiculos - 1))
            sub = (sub - 1) & restante
        return melhor

    return resolver((1 << k) - 1, p.tipos[0].quantidade)


# ---------------------------------------------------------------------------

def testar_avaliacao():
    print("\n1. Avaliação de rota")
    p = instancia(4, 1, assimetrica=False)
    d = p.D
    esperado = 10.0 + d[0][1] + d[1][2] + d[2][0]
    checar(abs(avaliar_rota(p, 0, [1, 2]) - esperado) < 1e-9,
           "custo = fixo + distância da ida e da volta")
    p.retornar = False
    checar(abs(avaliar_rota(p, 0, [1, 2]) - (10.0 + d[0][1] + d[1][2])) < 1e-9,
           "rota aberta não cobra a volta ao depósito")
    p.retornar = True
    p.dem[1] = 100
    checar(avaliar_rota(p, 0, [1, 2]) is None, "rota acima da capacidade é inviável")

    p = instancia(3, 2, janelas=True)
    p.jini[1], p.jfim[1] = 0.0, 1.0                   # janela impossível de alcançar
    checar(avaliar_rota(p, 0, [1]) is None or p.T[0][1] <= 1.0,
           "chegar depois do fechamento da janela torna a rota inviável")
    p.jini[2], p.jfim[2] = 500.0, 600.0
    r = avaliar_rota(p, 0, [2], detalhar=True)
    checar(r is not None and r.inicio[0] == 500.0 and r.chegada[0] < 500.0,
           "chegar antes da abertura faz o veículo esperar")


def testar_conferencia_de_rua():
    print("\n0. Geocodificação: a rua achada tem que ser a digitada")
    import geo

    def lugar(rua, nome=""):
        return {"rua": rua, "nome": nome}

    checar(geo._rua_confere("Rua Pio Correa 80", lugar("Rua Pio Corrêa")),
           "aceita a mesma rua escrita sem acento")
    checar(not geo._rua_confere("Rua Pio Correa 80", lugar("Rua Vilar Correa")),
           "rejeita rua só parecida (Pio Correa x Vilar Correa)")
    checar(not geo._rua_confere("São Clemente 244", lugar("", "Parque São Clemente")),
           "com número, um parque de mesmo nome não serve")
    checar(geo._rua_confere("R. Prof. Plínio Leite 10", lugar("Rua Professor Plínio Leite")),
           "aceita abreviações (Prof. = Professor)")
    checar(geo._menciona_cidade("Centro, Rio de Janeiro", "Rio de Janeiro")
           and not geo._menciona_cidade("Centro, Niterói", "Rio de Janeiro"),
           "reconhece a cidade escrita no endereço")


def testar_forca_bruta():
    print("\n2. VRPSolverEasy contra força bruta")
    casos = [
        ("simétrica", dict(assimetrica=False)),
        ("assimétrica (mão única)", dict(assimetrica=True)),
        ("com janelas de tempo", dict(janelas=True)),
        ("rota aberta", dict(retornar=False)),
    ]
    for nome, kw in casos:
        for semente in (3, 4):
            p = instancia(6, semente, **kw)
            verdade = otimo_forca_bruta(p)
            s = solver.resolver_exato(p, tempo_limite=30)
            checar(s.otima and abs(s.custo - verdade) < 1e-4,
                   f"{nome}, semente {semente}: solver {s.custo:.4f} = força bruta {verdade:.4f}")
            checar(not verificar(p, s), f"{nome}, semente {semente}: solução do solver viável")


def testar_grasp():
    print("\n3. GRASP")
    for nome, kw in (("simples", {}), ("janelas", dict(janelas=True)),
                     ("aberta", dict(retornar=False))):
        p = instancia(7, 11, **kw)
        g = heuristica.grasp(p, tempo_limite=2)
        verdade = otimo_forca_bruta(p)
        checar(not verificar(p, g), f"{nome}: solução do GRASP viável e completa")
        checar(g.custo >= verdade - 1e-6, f"{nome}: GRASP ({g.custo:.3f}) não fica abaixo do ótimo ({verdade:.3f})")
        checar(g.custo <= verdade * 1.10, f"{nome}: GRASP a menos de 10% do ótimo")

    # frota heterogênea: o GRASP tem que usar o tipo mais barato que der conta
    tipos = [TipoVeiculo("grande", 50, 1, 100.0, 1.0, 0.0, 0),
             TipoVeiculo("pequeno", 50, 1, 10.0, 1.0, 0.0, 0)]
    p = instancia(4, 5, tipos=tipos)
    g = heuristica.grasp(p, tempo_limite=1)
    checar(len(g.rotas) == 1 and p.tipos[g.rotas[0].tipo].nome == "pequeno",
           "com dois veículos iguais, escolhe o de custo fixo menor")


def testar_partida_e_prova():
    print("\n4. GRASP como solução de partida do solver")
    p = instancia(6, 21)
    g = heuristica.grasp(p, tempo_limite=2)
    s = solver.resolver_exato(p, tempo_limite=30, partida=g)
    verdade = otimo_forca_bruta(p)
    checar(s.otima, f"partindo do GRASP, o solver prova a otimalidade ({s.status})")
    checar(abs(s.custo - verdade) < 1e-4, "e o custo final é o ótimo da força bruta")
    checar(s.custo <= g.custo + 1e-9, "o solver nunca piora a solução de partida")


def ler_cvrplib(caminho) -> tuple[Problema, float]:
    """Lê uma instância .vrp da CVRPLIB (EUC_2D) e devolve o problema e o ótimo publicado.

    Distância arredondada como na CVRPLIB: nint(x) = floor(x + 0,5).
    """
    import re
    from pathlib import Path
    linhas = Path(caminho).read_text().splitlines()
    cabecalho = {l.split(":")[0].strip(): l.split(":", 1)[1].strip() for l in linhas if ":" in l}
    otimo = float(re.search(r"Optimal value:\s*(\d+)", cabecalho["COMMENT"]).group(1))
    caminhoes = int(re.search(r"-k(\d+)", cabecalho["NAME"]).group(1))
    secoes, atual = {"NODE_COORD_SECTION": [], "DEMAND_SECTION": []}, None
    for l in linhas:
        chave = l.strip()
        if chave in secoes or chave in ("DEPOT_SECTION", "EOF"):
            atual = chave if chave in secoes else None
        elif atual and chave:
            secoes[atual].append([float(x) for x in chave.split()[1:]])
    xy = np.array(secoes["NODE_COORD_SECTION"])
    n = len(xy)
    dist = np.floor(np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) + 0.5)
    p = Problema(
        nomes=[str(i + 1) for i in range(n)], depositos=[0], clientes=list(range(1, n)),
        demanda=np.array([int(d[0]) for d in secoes["DEMAND_SECTION"]]),
        servico=np.zeros(n), janela_ini=np.zeros(n), janela_fim=np.full(n, HORIZONTE),
        dist=dist, tempo=np.zeros((n, n)),
        tipos=[TipoVeiculo("caminhão", int(cabecalho["CAPACITY"]), caminhoes, 0, 1, 0, 0)],
        objetivo="distancia")
    return p, otimo


def testar_cvrplib():
    print("\n7. Instâncias da CVRPLIB (ótimo publicado)")
    from pathlib import Path
    for arquivo in sorted(Path(__file__).with_name("dados").joinpath("cvrplib").glob("*.vrp")):
        p, otimo = ler_cvrplib(arquivo)
        g = heuristica.grasp(p, tempo_limite=3)
        s = solver.resolver_exato(p, tempo_limite=60, partida=g)
        checar(s.otima and abs(s.custo - otimo) < 1e-6 and not verificar(p, s),
               f"{arquivo.stem}: solver prova {s.custo:.0f} = ótimo publicado {otimo:.0f} "
               f"({s.tempo_s:.1f} s; GRASP {g.custo:.0f})")


def testar_objetivo_e_json():
    print("\n8. Objetivo distância (CVRP clássico) e instância em JSON")
    import json
    import resolver_instancia
    tipos = [TipoVeiculo("caro", 8, 2, 500.0, 3.0, 9.0, 0),
             TipoVeiculo("barato", 8, 2, 1.0, 0.5, 0.0, 0)]
    p = instancia(6, 31, tipos=tipos)
    p.objetivo = "distancia"
    d = p.D
    checar(abs(avaliar_rota(p, 0, [1, 2]) - (d[0][1] + d[1][2] + d[2][0])) < 1e-9,
           "no objetivo distância, o valor da rota é só a distância (sem custo fixo)")
    # Mesma capacidade e mesmo depósito: só pela distância, os dois tipos valem
    # como um tipo só com os 4 veículos — que a força bruta sabe resolver.
    import copy
    unico = copy.copy(p)
    unico.tipos = [TipoVeiculo("qualquer", 8, 4, 0.0, 1.0, 0.0, 0)]
    verdade = otimo_forca_bruta(unico)
    s = solver.resolver_exato(p, tempo_limite=30)
    checar(s.otima and abs(s.custo - verdade) < 1e-4,
           f"solver = força bruta no objetivo distância ({s.custo:.4f} = {verdade:.4f})")

    copia = Problema.de_dict(json.loads(json.dumps(p.para_dict())))
    checar(np.allclose(copia.dist, p.dist) and copia.objetivo == p.objetivo
           and [t.custo_fixo for t in copia.tipos] == [t.custo_fixo for t in p.tipos],
           "a instância sai e volta do JSON sem perder nada")
    r = resolver_instancia.resolver(copia, tempo_exato=30, tempo_grasp=1)
    checar(abs(r.custo - s.custo) < 1e-4, "resolver a instância em JSON dá o mesmo ótimo")


def testar_frota_insuficiente():
    print("\n5. Frota insuficiente")
    p = instancia(6, 7)
    p.tipos[0].quantidade = 1
    p.tipos[0].capacidade = 5
    g = heuristica.grasp(p, tempo_limite=1)
    checar(len(g.nao_atendidos) > 0, "o GRASP aponta os clientes que ficaram sem veículo")
    checar(not verificar(p, g), "e o que ele atendeu continua viável")


def testar_app():
    print("\n6. Fluxo completo da interface")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].select("Rio de Janeiro, RJ").run()
    [b for b in at.sidebar.button if "Aplicar" in b.label][0].click().run()
    checar(at.session_state.cidade == "Rio de Janeiro, RJ", "'Aplicar cidade' troca a cidade padrão")
    [b for b in at.sidebar.button if "exemplo" in b.label][0].click().run()
    checar(at.session_state.cidade == "Niterói, RJ",
           "carregar o exemplo volta a cidade para Niterói (os endereços dele não a dizem)")
    [b for b in at.button if "Geocodificar" in b.label][0].click().run()
    checar(not at.exception and not at.error, "etapa 1: exemplo geocodificado sem erros")
    [b for b in at.button if "Confirmar pontos" in b.label][0].click().run()
    checar(not at.exception and at.title[0].value.startswith("Etapa 2"), "etapa 2: frota aberta")
    [b for b in at.button if "Calcular rotas" in b.label][0].click().run()
    checar(not at.exception and at.title[0].value.startswith("Etapa 3"), "etapa 3: rotas calculadas")
    checar(any("ótima" in s.value.lower() for s in at.success),
           "o exemplo termina com solução ótima provada")
    checar(at.metric[0].label == "Distância total",
           "o objetivo padrão é o CVRP clássico (menor distância)")
    km_classico = float(at.metric[0].value.split()[0])

    [b for b in at.button if b.label == "← 2. Frota"][0].click().run()
    at.radio[0].set_value("custo").run()
    [b for b in at.button if "Calcular rotas" in b.label][0].click().run()
    # O valor exato muda quando o OpenStreetMap é atualizado (a matriz vem do
    # OSRM na hora): confere-se a prova de otimalidade, não um número fixo.
    checar(not at.exception and at.metric[0].label == "Custo total"
           and any("ótima" in s.value.lower() for s in at.success),
           f"no objetivo custo, o exemplo também termina com ótimo provado ({at.metric[0].value})")
    checar(float(at.metric[1].value.split()[0]) >= km_classico - 1e-6,
           f"o plano de menor custo não roda menos km que o de menor distância "
           f"({at.metric[1].value} x {km_classico} km)")

    [b for b in at.button if b.label == "← 2. Frota"][0].click().run()
    checar(not at.exception and at.title[0].value.startswith("Etapa 2"),
           "da etapa 3 dá para voltar à frota")
    [b for b in at.sidebar.button if "3. Rotas" in b.label][0].click().run()
    checar(not at.exception and at.title[0].value.startswith("Etapa 3"),
           "e voltar às rotas pela barra lateral, sem recalcular")
    [b for b in at.button if b.label == "🏠 Novo roteiro"][0].click().run()
    checar(not at.exception and at.title[0].value.startswith("Etapa 1")
           and "resultado" not in at.session_state,
           "'Novo roteiro' volta à tela inicial e apaga o cálculo")


def main() -> int:
    random.seed(0)
    testar_conferencia_de_rua()
    testar_avaliacao()
    testar_forca_bruta()
    testar_grasp()
    testar_partida_e_prova()
    testar_frota_insuficiente()
    testar_cvrplib()
    testar_objetivo_e_json()
    if "--app" in sys.argv:
        testar_app()
    print("\n" + "-" * 60)
    if falhas:
        print(f"{len(falhas)} FALHA(S):")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("todos os testes passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
