"""
Saídas do plano de rotas: planilha Excel e links do Google Maps.

A planilha é o que a operação usa no dia: uma aba por visão (resumo, ordem de
visita de cada veículo, clientes). O link do Google Maps abre a rota no celular
do motorista com a navegação passo a passo.
"""

from __future__ import annotations

import io
from urllib.parse import quote

import pandas as pd

from modelo import Problema, Solucao

# O Google Maps aceita no máximo ~10 pontos num link de rota; rotas maiores
# são divididas em trechos que continuam um do outro.
PONTOS_POR_LINK = 10


def _hora(minutos: float) -> str:
    minutos = int(round(minutos))
    return f"{minutos // 60 % 24:02d}:{minutos % 60:02d}"


def links_google_maps(p: Problema, sol: Solucao, coords) -> list[list[str]]:
    """Para cada rota, a lista de links (um por trecho de até 10 pontos)."""
    resultado = []
    for rota in sol.rotas:
        deposito = p.tipos[rota.tipo].deposito
        pontos = [deposito] + rota.clientes + ([deposito] if p.retornar else [])
        links = []
        inicio = 0
        while inicio < len(pontos) - 1:
            trecho = pontos[inicio:inicio + PONTOS_POR_LINK]
            caminho = "/".join(quote(f"{coords[i][0]:.6f},{coords[i][1]:.6f}") for i in trecho)
            links.append(f"https://www.google.com/maps/dir/{caminho}")
            inicio += PONTOS_POR_LINK - 1
        resultado.append(links)
    return resultado


def tabela_paradas(p: Problema, sol: Solucao, enderecos: list[str]) -> pd.DataFrame:
    """Uma linha por parada, na ordem em que cada veículo faz as visitas."""
    linhas = []
    for k, rota in enumerate(sol.rotas):
        tipo = p.tipos[rota.tipo]
        deposito = tipo.deposito
        linhas.append({"Rota": k + 1, "Veículo": tipo.nome, "Parada": 0,
                       "Endereço": f"(saída) {enderecos[deposito]}", "Demanda": 0,
                       "Carga no veículo": rota.carga, "Chegada": "",
                       "Início do atendimento": _hora(rota.saida_deposito),
                       "Espera (min)": 0.0, "Km acumulado": 0.0})
        carga = rota.carga
        anterior = deposito
        km = 0.0
        for ordem, c in enumerate(rota.clientes):
            km += p.D[anterior][c]
            carga -= p.dem[c]
            linhas.append({
                "Rota": k + 1, "Veículo": tipo.nome, "Parada": ordem + 1,
                "Endereço": enderecos[c], "Demanda": p.dem[c],
                "Carga no veículo": carga,
                "Chegada": _hora(rota.chegada[ordem]),
                "Início do atendimento": _hora(rota.inicio[ordem]),
                "Espera (min)": round(rota.inicio[ordem] - rota.chegada[ordem], 1),
                "Km acumulado": round(km, 2),
            })
            anterior = c
        if p.retornar:
            km += p.D[anterior][deposito]
            linhas.append({"Rota": k + 1, "Veículo": tipo.nome,
                           "Parada": len(rota.clientes) + 1,
                           "Endereço": f"(retorno) {enderecos[deposito]}", "Demanda": 0,
                           "Carga no veículo": 0, "Chegada": _hora(rota.fim),
                           "Início do atendimento": "", "Espera (min)": 0.0,
                           "Km acumulado": round(km, 2)})
    tabela = pd.DataFrame(linhas)
    if not p.usar_janelas and not tabela.empty:
        # Sem janelas o veículo nunca espera: o atendimento começa na chegada.
        tabela["Chegada"] = tabela["Chegada"].where(tabela["Chegada"] != "",
                                                    tabela["Início do atendimento"])
        tabela = tabela.drop(columns=["Início do atendimento", "Espera (min)"]) \
                       .rename(columns={"Chegada": "Horário"})
    return tabela


def tabela_rotas(p: Problema, sol: Solucao) -> pd.DataFrame:
    """Uma linha por veículo usado."""
    return pd.DataFrame([{
        "Rota": k + 1,
        "Veículo": p.tipos[r.tipo].nome,
        "Depósito": p.nomes[p.tipos[r.tipo].deposito],
        "Paradas": len(r.clientes),
        "Carga": r.carga,
        "Capacidade": p.tipos[r.tipo].capacidade,
        "Ocupação (%)": round(100.0 * r.carga / p.tipos[r.tipo].capacidade, 1),
        "Distância (km)": round(r.distancia, 2),
        "Dirigindo (min)": round(r.tempo_viagem, 1),
        "Saída": _hora(r.saida_deposito),
        "Término": _hora(r.fim),
        "Custo (R$)": round(r.custo, 2),
    } for k, r in enumerate(sol.rotas)])


def planilha(p: Problema, sol: Solucao, enderecos: list[str], coords,
             comparacao: pd.DataFrame | None = None) -> bytes:
    """Excel com resumo, rotas, ordem de visita e clientes."""
    resumo = pd.DataFrame({
        "Indicador": ["Método", "Situação", "Custo total (R$)", "Distância total (km)",
                      "Veículos usados", "Clientes atendidos", "Clientes não atendidos",
                      "Limitante inferior (R$)", "Gap para o ótimo (%)",
                      "Tempo de cálculo (s)"],
        "Valor": [sol.metodo, sol.status, round(sol.custo, 2), round(sol.distancia, 2),
                  len(sol.rotas), sum(len(r.clientes) for r in sol.rotas),
                  len(sol.nao_atendidos),
                  "" if sol.limitante_inferior is None else round(sol.limitante_inferior, 2),
                  "" if sol.gap is None else round(sol.gap, 2), round(sol.tempo_s, 2)],
    })

    rotas = tabela_rotas(p, sol)
    rotas["Google Maps"] = [" ".join(l) for l in links_google_maps(p, sol, coords)]

    clientes = pd.DataFrame([{
        "Ponto": enderecos[i], "Tipo": "depósito" if i in p.depositos else "cliente",
        "Demanda": p.dem[i], "Atendimento (min)": p.srv[i],
        "Latitude": coords[i][0], "Longitude": coords[i][1],
    } for i in p.depositos + p.clientes])

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        abas = {"Resumo": resumo, "Rotas": rotas,
                "Ordem de visita": tabela_paradas(p, sol, enderecos), "Pontos": clientes}
        if comparacao is not None:
            abas["Comparação"] = comparacao
        for nome, df in abas.items():
            df.to_excel(escritor, sheet_name=nome, index=False)
            aba = escritor.sheets[nome]
            for coluna in aba.columns:
                largura = max(len(str(c.value or "")) for c in coluna[:200])
                aba.column_dimensions[coluna[0].column_letter].width = min(max(largura + 2, 10), 60)
            aba.freeze_panes = "A2"
    return buffer.getvalue()
