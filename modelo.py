"""
O problema de roteirização e a avaliação de uma rota.

Tudo o que o solver exato e a heurística precisam saber sobre a instância fica
em `Problema`, e a única regra de custo e viabilidade fica em `avaliar_rota`.
As duas abordagens são conferidas por essa mesma função — se o VRPSolverEasy e
o GRASP calculassem custo cada um do seu jeito, a comparação entre eles não
valeria nada.

Unidades: distância em km, tempo em minutos contados a partir da meia-noite
(08:00 = 480), dinheiro em reais.

Dois objetivos: "distancia" é o CVRP clássico (minimizar os km rodados; os
custos da frota não entram) e "custo" minimiza custo fixo + km + hora em R$.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HORIZONTE = 48 * 60.0       # "sem limite de horário": dois dias em minutos
FOLGA = 1e-6


@dataclass
class TipoVeiculo:
    nome: str
    capacidade: int
    quantidade: int
    custo_fixo: float          # por veículo usado (motorista, diária, depreciação)
    custo_km: float
    custo_hora: float          # sobre o tempo dirigindo
    deposito: int              # índice do ponto de onde sai
    jornada_ini: float = 0.0
    jornada_fim: float = HORIZONTE


@dataclass
class Problema:
    nomes: list[str]                  # rótulo de cada ponto
    depositos: list[int]              # índices dos depósitos
    clientes: list[int]               # índices dos clientes
    demanda: np.ndarray
    servico: np.ndarray               # minutos parado em cada cliente
    janela_ini: np.ndarray
    janela_fim: np.ndarray
    dist: np.ndarray                  # km, possivelmente assimétrica
    tempo: np.ndarray                 # minutos, possivelmente assimétrica
    tipos: list[TipoVeiculo]
    retornar: bool = True             # o veículo volta ao depósito no fim
    usar_janelas: bool = False
    objetivo: str = "custo"           # "distancia" (CVRP clássico) ou "custo"

    def __post_init__(self):
        # Listas de listas são bem mais rápidas que numpy para acessar um
        # elemento de cada vez, que é o que a heurística faz milhões de vezes.
        self.D = self.dist.tolist()
        self.T = self.tempo.tolist()
        self.dem = [int(x) for x in self.demanda]
        self.srv = [float(x) for x in self.servico]
        self.jini = [float(x) for x in self.janela_ini]
        self.jfim = [float(x) for x in self.janela_fim]

    @property
    def demanda_total(self) -> int:
        return int(sum(self.dem[c] for c in self.clientes))

    @property
    def capacidade_total(self) -> int:
        return int(sum(t.capacidade * t.quantidade for t in self.tipos))

    @property
    def unidade(self) -> str:
        """Unidade do valor da função objetivo."""
        return "km" if self.objetivo == "distancia" else "R$"

    # -- Instância em JSON ---------------------------------------------------
    #
    # A fronteira entre o módulo geográfico e o de otimização: tudo o que o
    # solver precisa (matriz, demandas, frota) num arquivo que dá para gerar,
    # guardar e resolver sem endereços nem internet (ver resolver_instancia.py).

    def para_dict(self, coords: list[tuple[float, float]] | None = None) -> dict:
        pontos = []
        for i, nome in enumerate(self.nomes):
            ponto = {"id": i, "nome": nome, "deposito": i in self.depositos,
                     "demanda": self.dem[i], "atendimento_min": self.srv[i]}
            if self.usar_janelas:
                ponto["janela_min"] = [self.jini[i], self.jfim[i]]
            if coords is not None:
                ponto["lat"], ponto["lon"] = coords[i]
            pontos.append(ponto)
        return {
            "objetivo": self.objetivo, "retornar_ao_deposito": self.retornar,
            "usar_janelas": self.usar_janelas, "pontos": pontos,
            "frota": [{"tipo": t.nome, "capacidade": t.capacidade, "quantidade": t.quantidade,
                       "custo_fixo": t.custo_fixo, "custo_km": t.custo_km,
                       "custo_hora": t.custo_hora, "deposito": t.deposito,
                       "jornada_min": [t.jornada_ini, t.jornada_fim]} for t in self.tipos],
            "distancia_km": self.dist.tolist(),
            "tempo_min": self.tempo.tolist(),
        }

    @classmethod
    def de_dict(cls, d: dict) -> "Problema":
        pontos = sorted(d["pontos"], key=lambda x: x["id"])
        n = len(pontos)
        janelas = [x.get("janela_min", [0.0, HORIZONTE]) for x in pontos]
        return cls(
            nomes=[x["nome"] for x in pontos],
            depositos=[x["id"] for x in pontos if x["deposito"]],
            clientes=[x["id"] for x in pontos if not x["deposito"]],
            demanda=np.array([x["demanda"] for x in pontos]),
            servico=np.array([x.get("atendimento_min", 0.0) for x in pontos], dtype=float),
            janela_ini=np.array([j[0] for j in janelas], dtype=float),
            janela_fim=np.array([j[1] for j in janelas], dtype=float),
            dist=np.array(d["distancia_km"], dtype=float).reshape(n, n),
            tempo=np.array(d.get("tempo_min", np.zeros((n, n))), dtype=float).reshape(n, n),
            tipos=[TipoVeiculo(t["tipo"], int(t["capacidade"]), int(t["quantidade"]),
                               float(t.get("custo_fixo", 0)), float(t.get("custo_km", 0)),
                               float(t.get("custo_hora", 0)), int(t["deposito"]),
                               *map(float, t.get("jornada_min", [0.0, HORIZONTE])))
                   for t in d["frota"]],
            retornar=bool(d.get("retornar_ao_deposito", True)),
            usar_janelas=bool(d.get("usar_janelas", False)),
            objetivo=d.get("objetivo", "custo"),
        )


@dataclass
class Rota:
    tipo: int
    clientes: list[int]
    custo: float = 0.0
    distancia: float = 0.0
    tempo_viagem: float = 0.0          # só dirigindo
    carga: int = 0
    chegada: list[float] = field(default_factory=list)   # chegada a cada cliente
    inicio: list[float] = field(default_factory=list)    # início do atendimento
    saida_deposito: float = 0.0
    fim: float = 0.0                   # volta ao depósito (ou fim do último atendimento)


@dataclass
class Solucao:
    rotas: list[Rota]
    metodo: str
    status: str = ""
    otima: bool = False
    limitante_inferior: float | None = None
    tempo_s: float = 0.0
    nao_atendidos: list[int] = field(default_factory=list)
    iteracoes: int = 0

    @property
    def custo(self) -> float:
        return sum(r.custo for r in self.rotas)

    @property
    def distancia(self) -> float:
        return sum(r.distancia for r in self.rotas)

    @property
    def gap(self) -> float | None:
        """Distância percentual entre a solução e o limitante inferior."""
        if self.limitante_inferior is None or self.custo <= 0:
            return None
        return max(0.0, 100.0 * (self.custo - self.limitante_inferior) / self.custo)


def avaliar_rota(p: Problema, tipo: int, seq: list[int], detalhar: bool = False
                 ) -> Rota | float | None:
    """Custo da rota, ou None se ela for inviável.

    Viável quer dizer: a carga cabe no veículo, cada cliente começa a ser
    atendido dentro da sua janela (chegar cedo é permitido — o veículo espera),
    e o veículo termina dentro da jornada. Com `detalhar=True` devolve a `Rota`
    completa com os horários; sem, só o custo, que é o que a busca local precisa.
    """
    if not seq:
        return Rota(tipo, [], 0.0) if detalhar else 0.0

    t = p.tipos[tipo]
    carga = 0
    for c in seq:
        carga += p.dem[c]
    if carga > t.capacidade:
        return None

    D, T = p.D, p.T
    janelas = p.usar_janelas
    posicao = t.deposito
    relogio = t.jornada_ini
    distancia = 0.0
    viagem = 0.0
    chegadas, inicios = ([], []) if detalhar else (None, None)

    for c in seq:
        trecho = T[posicao][c]
        relogio += trecho
        viagem += trecho
        distancia += D[posicao][c]
        comeco = relogio
        if janelas:
            if comeco < p.jini[c]:
                comeco = p.jini[c]                   # chegou cedo: espera
            if comeco > p.jfim[c] + FOLGA:
                return None
        if detalhar:
            chegadas.append(relogio)
            inicios.append(comeco)
        relogio = comeco + p.srv[c]
        posicao = c

    if p.retornar:
        distancia += D[posicao][t.deposito]
        viagem += T[posicao][t.deposito]
        relogio += T[posicao][t.deposito]

    if relogio > t.jornada_fim + FOLGA:
        return None

    if p.objetivo == "distancia":
        custo = distancia
    else:
        custo = t.custo_fixo + t.custo_km * distancia + t.custo_hora * viagem / 60.0
    if not detalhar:
        return custo
    return Rota(tipo=tipo, clientes=list(seq), custo=custo, distancia=distancia,
                tempo_viagem=viagem, carga=carga, chegada=chegadas, inicio=inicios,
                saida_deposito=t.jornada_ini, fim=relogio)


def melhor_solucao(solucoes: dict[str, Solucao]) -> str | None:
    """Nome da melhor solução: a que deixa menos clientes sem atendimento; entre
    essas, a de menor valor; no empate, a que tem prova de otimalidade."""
    validas = {k: s for k, s in solucoes.items() if s.rotas}
    if not validas:
        return None
    return min(validas, key=lambda k: (len(validas[k].nao_atendidos),
                                       round(validas[k].custo, 4), not validas[k].otima))


def verificar(p: Problema, sol: Solucao) -> list[str]:
    """Lista de problemas da solução (vazia = viável e completa)."""
    erros = []
    vistos: list[int] = []
    uso = [0] * len(p.tipos)
    for r in sol.rotas:
        uso[r.tipo] += 1
        vistos.extend(r.clientes)
        if avaliar_rota(p, r.tipo, r.clientes) is None:
            erros.append(f"rota do tipo {p.tipos[r.tipo].nome} inviável: {r.clientes}")
    faltando = set(p.clientes) - set(vistos) - set(sol.nao_atendidos)
    if faltando:
        erros.append(f"clientes não atendidos: {sorted(faltando)}")
    if len(vistos) != len(set(vistos)):
        erros.append("cliente visitado mais de uma vez")
    for k, t in enumerate(p.tipos):
        if uso[k] > t.quantidade:
            erros.append(f"{uso[k]} veículos do tipo {t.nome}, mas só há {t.quantidade}")
    return erros
