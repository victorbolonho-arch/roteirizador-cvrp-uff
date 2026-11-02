"""
Solução exata com o VRPSolverEasy (Inria / UFF).

O VRPSolverEasy é um branch-cut-and-price (Pessoa, Sadykov, Uchoa e
Vanderbeck, 2020): resolve o problema até a otimalidade, e quando o tempo
acaba devolve a melhor solução encontrada junto com um limitante inferior —
o que permite dizer o quão longe do ótimo ela pode estar, no máximo.

Três cuidados descobertos ao testar a biblioteca:

  1. Toda janela de tempo precisa ser informada. O padrão `tw_end=0` não quer
     dizer "sem janela", quer dizer "janela [0, 0]" — e o problema vira
     inviável. Sem janelas, usa-se um horizonte longo.
  2. A rota aberta nativa (`end_point_id=-1`) não funcionou na versão 0.1.3.
     "Não voltar ao depósito" é modelado com arcos de volta de custo zero, que
     é a formulação padrão.
  3. A versão 0.1.4 do PyPI veio sem a biblioteca compilada; é preciso a 0.1.3.

O solver roda em um processo separado. A biblioteca nativa não pode ser
interrompida e, se falhar, derrubaria junto o servidor do aplicativo.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from modelo import HORIZONTE, Problema, Rota, Solucao, avaliar_rota

STATUS = {
    0: "ótima (provada pelo solver)",
    1: "melhor solução encontrada no tempo limite",
    2: "a solução de partida já era a ótima",
    3: "o solver não melhorou a solução de partida no tempo limite",
    4: "solução viável encontrada",
    -1: "interrompido por erro",
    -2: "inviável com a frota disponível",
}


def _montar(p: Problema, tempo_limite: float, limitante: float | None,
            frota_livre: bool = False):
    import VRPSolverEasy as vrpse

    m = vrpse.Model()

    for d in p.depositos:
        m.add_depot(id=d, name=f"dep{d}", tw_begin=0.0, tw_end=HORIZONTE)

    for c in p.clientes:
        ini, fim = (p.jini[c], p.jfim[c]) if p.usar_janelas else (0.0, HORIZONTE)
        m.add_customer(id=c, name=f"cli{c}", demand=p.dem[c],
                       service_time=p.srv[c], tw_begin=ini, tw_end=fim)

    deposito = set(p.depositos)
    pontos = p.depositos + p.clientes
    for i in pontos:
        for j in pontos:
            if i == j or (i in deposito and j in deposito):
                continue
            if not p.retornar and j in deposito:
                distancia, tempo = 0.0, 0.0      # volta "de graça" = rota aberta
            else:
                distancia, tempo = p.D[i][j], p.T[i][j]
            m.add_link(start_point_id=i, end_point_id=j, is_directed=True,
                       distance=distancia, time=tempo)

    for k, t in enumerate(p.tipos):
        m.add_vehicle_type(
            id=k + 1, name=t.nome,
            start_point_id=t.deposito, end_point_id=t.deposito,
            capacity=t.capacidade,
            max_number=len(p.clientes) if frota_livre else t.quantidade,
            fixed_cost=t.custo_fixo, var_cost_dist=t.custo_km,
            var_cost_time=t.custo_hora / 60.0,
            tw_begin=t.jornada_ini, tw_end=t.jornada_fim,
        )

    # A heurística interna do BaPCod acha soluções viáveis mais cedo; nos
    # testes com dois depósitos cortou o tempo até o ótimo de 16 s para 10 s.
    parametros = dict(time_limit=float(tempo_limite), solver_name="CLP",
                      heuristic_used=True,
                      time_limit_heuristic=max(1.0, tempo_limite / 3))
    if limitante is not None:
        # Um pouco acima, para que, se a solução de partida já for a ótima, o
        # solver ainda a devolva (com o limitante exato ele responderia só
        # "não existe melhor", sem rota nenhuma).
        parametros["upper_bound"] = limitante * (1 + 1e-6) + 0.01
    m.set_parameters(**parametros)
    return m


def _resolver_aqui(p: Problema, tempo_limite: float, limitante: float | None,
                   frota_livre: bool = False) -> dict:
    """Roda o solver neste processo e devolve um dicionário simples."""
    inicio = time.perf_counter()
    m = _montar(p, tempo_limite, limitante, frota_livre)
    m.solve()
    resultado = {"status": int(m.status), "mensagem": str(m.message),
                 "rotas": [], "valor": None, "limitante_inferior": None,
                 "tempo": time.perf_counter() - inicio}
    try:
        resultado["limitante_inferior"] = float(m.statistics.best_lb)
    except Exception:
        pass
    if m.solution.is_defined():
        resultado["valor"] = float(m.solution.value)
        clientes = set(p.clientes)
        for r in m.solution.routes:
            resultado["rotas"].append(
                (int(r.vehicle_type_id) - 1, [int(x) for x in r.point_ids if x in clientes]))
    return resultado


def _rodar_isolado(p: Problema, tempo_limite: float, limitante: float | None,
                   frota_livre: bool) -> dict | str:
    """Executa o solver num processo-filho. Devolve o resultado ou uma mensagem de erro."""
    with tempfile.TemporaryDirectory() as pasta:
        entrada = Path(pasta) / "entrada.pkl"
        saida = Path(pasta) / "saida.pkl"
        log = Path(pasta) / "solver.log"
        entrada.write_bytes(pickle.dumps((p, tempo_limite, limitante, frota_livre)))
        # O BaPCod nem sempre respeita o próprio limite de tempo (num teste,
        # uma execução de 30 s levou 27 minutos). Por isso o processo-filho é
        # vigiado de fora e, passado o prazo, morto junto com seus filhos. A
        # saída dele vai para um arquivo, que serve para diagnosticar falhas.
        with open(log, "w", encoding="utf-8", errors="replace") as arquivo_log:
            processo = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(entrada), str(saida)],
                stdout=arquivo_log, stderr=subprocess.STDOUT,
                cwd=pasta,       # o BaPCod grava arquivos de depuração no diretório atual
            )
            try:
                processo.wait(timeout=tempo_limite + 60)
            except subprocess.TimeoutExpired:
                _encerrar(processo)
                return "o solver passou do tempo e foi interrompido"
        if not saida.exists():
            linhas = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return f"falha no solver: {linhas[-1] if linhas else 'sem mensagem'}"
        return pickle.loads(saida.read_bytes())


def _encerrar(processo: subprocess.Popen) -> None:
    """Mata o processo do solver e qualquer filho que ele tenha criado."""
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(processo.pid)],
                       capture_output=True)
    else:
        processo.kill()
    try:
        processo.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _uso_da_frota(p: Problema, rotas: list[tuple[int, list[int]]]) -> bool:
    """As rotas respeitam a quantidade de veículos de cada tipo?"""
    uso = [0] * len(p.tipos)
    for tipo, _ in rotas:
        uso[tipo] += 1
    return all(uso[k] <= t.quantidade for k, t in enumerate(p.tipos))


def resolver_exato(p: Problema, tempo_limite: float = 30.0,
                   partida: Solucao | None = None) -> Solucao:
    """Resolve com o VRPSolverEasy, opcionalmente partindo de uma solução heurística.

    Com `partida`, o custo dela entra como limitante superior: o solver descarta
    de cara tudo que for pior. Se concluir que não existe nada melhor, isso
    PROVA que a heurística achou o ótimo.

    Quando há vários depósitos, o limite de quantidade de veículos deixa o
    branch-cut-and-price muito mais lento (nos testes, de 10 s até o ótimo
    para nenhuma solução em 30 s). Por isso o problema é resolvido em duas
    fases:

      1. sem o limite de quantidade (uma RELAXAÇÃO, mais fácil). Se o ótimo
         dela já respeita a frota real, ele é o ótimo do problema real — não
         existe solução viável melhor que o ótimo de um problema mais livre;
      2. caso contrário, o valor da fase 1 vale como limitante inferior, e o
         solver roda de novo com a frota real no tempo que sobrou.
    """
    limitante = partida.custo if partida and partida.rotas and not partida.nao_atendidos else None
    inicio = time.perf_counter()
    limitante_relaxacao = None
    restante = tempo_limite

    frota_pode_apertar = len(p.depositos) > 1 and         any(t.quantidade < len(p.clientes) for t in p.tipos)

    if frota_pode_apertar:
        r = _rodar_isolado(p, tempo_limite / 2, limitante, frota_livre=True)
        if isinstance(r, dict):
            if r["status"] == 0 and r["rotas"] and _uso_da_frota(p, r["rotas"]):
                return _montar_solucao(p, r, partida, time.perf_counter() - inicio,
                                       nota="a relaxação sem limite de frota já respeitava a frota real")
            if r["status"] == 2 and partida is not None:
                # Nada melhor que a heurística nem com frota ilimitada: ela é ótima.
                return _montar_solucao(p, r, partida, time.perf_counter() - inicio)
            if r["status"] == 0 and r["valor"] is not None:
                limitante_relaxacao = r["valor"]
            elif r["limitante_inferior"] is not None:
                limitante_relaxacao = r["limitante_inferior"]
        restante = max(5.0, tempo_limite - (time.perf_counter() - inicio))

    r = _rodar_isolado(p, restante, limitante, frota_livre=False)
    if isinstance(r, str):
        return Solucao(partida.rotas if partida else [], "VRPSolverEasy", status=r,
                       tempo_s=time.perf_counter() - inicio,
                       limitante_inferior=limitante_relaxacao)
    if limitante_relaxacao is not None:
        r["limitante_inferior"] = max(r["limitante_inferior"] or 0.0, limitante_relaxacao)
    return _montar_solucao(p, r, partida, time.perf_counter() - inicio)


def _montar_solucao(p: Problema, r: dict, partida: Solucao | None, tempo: float,
                    nota: str = "") -> Solucao:
    """Converte o resultado bruto do solver numa `Solucao` conferida."""
    codigo = r["status"]
    status = STATUS.get(codigo, r["mensagem"])
    sol = Solucao([], "VRPSolverEasy", status=status + (f" — {nota}" if nota else ""),
                  tempo_s=tempo, limitante_inferior=r["limitante_inferior"])

    if r["rotas"]:
        sol.rotas = [avaliar_rota(p, tipo, seq, detalhar=True) for tipo, seq in r["rotas"]]
        sol.otima = codigo == 0
    elif codigo in (2, 3) and partida is not None:
        # O solver não devolveu rotas porque nenhuma é melhor que a de partida.
        sol.rotas = partida.rotas
        sol.otima = codigo == 2
        if codigo == 2:
            sol.metodo = "VRPSolverEasy (confirmou o GRASP)"

    if any(rota is None for rota in sol.rotas):
        # Só aconteceria se o solver e a avaliação discordassem sobre uma rota.
        sol.rotas = [rota for rota in sol.rotas if rota is not None]
        sol.status += " — atenção: uma rota do solver não passou na conferência"
    if sol.otima and sol.rotas:
        sol.limitante_inferior = sol.custo
    return sol


if __name__ == "__main__":
    # Modo processo-filho: lê o problema, resolve, grava o resultado.
    origem, destino = Path(sys.argv[1]), Path(sys.argv[2])
    problema, limite, ub, livre = pickle.loads(origem.read_bytes())
    destino.write_bytes(pickle.dumps(_resolver_aqui(problema, limite, ub, livre)))
