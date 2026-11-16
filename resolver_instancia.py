"""
Resolve uma instância em JSON — o módulo de otimização sozinho.

O app gera a instância (pontos, matriz de distâncias, frota) no módulo
geográfico e permite baixá-la na etapa 3. Este script lê esse arquivo e
devolve a solução também em JSON, sem interface, sem endereços e sem internet.
Assim o solver pode ser testado e usado separado do resto, inclusive com
matrizes artificiais ou instâncias da CVRPLIB.

Uso:
    python resolver_instancia.py instancia.json
    python resolver_instancia.py instancia.json --tempo 120 --saida solucao.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import exportar
import heuristica
import solver
from modelo import Problema, melhor_solucao, verificar


def resolver(p: Problema, tempo_exato: float = 60, tempo_grasp: float = 5):
    """GRASP como partida + VRPSolverEasy; devolve a melhor das duas."""
    solucoes = {"GRASP": heuristica.grasp(p, tempo_limite=tempo_grasp)}
    solucoes["VRPSolverEasy"] = solver.resolver_exato(p, tempo_exato, partida=solucoes["GRASP"])
    return solucoes[melhor_solucao(solucoes)]


def main() -> int:
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("instancia", type=Path)
    argumentos.add_argument("--tempo", type=float, default=60, help="limite do VRPSolverEasy (s)")
    argumentos.add_argument("--saida", type=Path, help="arquivo JSON da solução")
    a = argumentos.parse_args()

    p = Problema.de_dict(json.loads(a.instancia.read_text(encoding="utf-8")))
    sol = resolver(p, a.tempo)
    erros = verificar(p, sol)
    resultado = exportar.solucao_dict(p, sol)

    print(f"{sol.metodo}: {sol.custo:.2f} {p.unidade} · {len(sol.rotas)} rotas · {sol.status}")
    for k, r in enumerate(resultado["rotas"], start=1):
        print(f"  rota {k} ({r['veiculo']}): {r['sequencia']}")
    if erros:
        print("ATENÇÃO:", "; ".join(erros))
    if a.saida:
        a.saida.write_bytes(exportar.json_bytes(resultado))
        print(f"solução gravada em {a.saida}")
    return 1 if erros else 0


if __name__ == "__main__":
    raise SystemExit(main())
