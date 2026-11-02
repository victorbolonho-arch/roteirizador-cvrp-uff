"""
O problema de roteirização e a avaliação de uma rota.

Tudo o que o solver exato e a heurística precisam saber sobre a instância fica
em `Problema`, e a única regra de custo e viabilidade fica em `avaliar_rota`.
As duas abordagens são conferidas por essa mesma função — se o VRPSolverEasy e
o GRASP calculassem custo cada um do seu jeito, a comparação entre eles não
valeria nada.

Unidades: distância em km, tempo em minutos contados a partir da meia-noite
(08:00 = 480), dinheiro em reais.
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

    custo = t.custo_fixo + t.custo_km * distancia + t.custo_hora * viagem / 60.0
    if not detalhar:
        return custo
    return Rota(tipo=tipo, clientes=list(seq), custo=custo, distancia=distancia,
                tempo_viagem=viagem, carga=carga, chegada=chegadas, inicio=inicios,
                saida_deposito=t.jornada_ini, fim=relogio)


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
