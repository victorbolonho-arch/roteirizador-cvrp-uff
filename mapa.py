"""
Desenho das rotas no mapa (Folium / Leaflet, sobre o OpenStreetMap).

Cada rota segue o traçado real das ruas devolvido pelo OSRM, numa cor própria e
numa camada própria — dá para ligar e desligar veículos no controle de camadas.
As paradas levam o número da ordem de visita, que é o que o motorista precisa.
"""

from __future__ import annotations

import folium
from folium.plugins import Fullscreen

import geo
from modelo import Problema, Solucao

# Mapas de fundo sem chave de API. (O "Positron" da CARTO, muito usado em
# exemplos do Folium, passou a exigir chave e aparece com marca d'água.)
FUNDO_CINZA = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
               "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")
CREDITO_CINZA = "Esri, HERE, Garmin, © OpenStreetMap contributors"


def _fundos(mapa: folium.Map) -> None:
    folium.TileLayer("openstreetmap", name="OpenStreetMap").add_to(mapa)
    folium.TileLayer(FUNDO_CINZA, attr=CREDITO_CINZA, name="Cinza (destaca as rotas)",
                     show=False, max_zoom=16).add_to(mapa)


# Paleta categórica em ordem fixa (validada para daltonismo nos trabalhos 3 e 4).
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#4a3aa7", "#e34948"]


def cor_da_rota(indice: int) -> str:
    return CORES[indice % len(CORES)]


def _hora(minutos: float) -> str:
    minutos = int(round(minutos))
    return f"{minutos // 60 % 24:02d}:{minutos % 60:02d}"


def _marcador_numerado(numero: int, cor: str) -> folium.DivIcon:
    return folium.DivIcon(
        icon_size=(26, 26), icon_anchor=(13, 13),
        html=(f'<div style="background:{cor};color:#fff;border:2px solid #fff;'
              f'border-radius:50%;width:26px;height:26px;line-height:22px;'
              f'text-align:center;font:600 12px system-ui,sans-serif;'
              f'box-shadow:0 1px 4px rgba(0,0,0,.45)">{numero}</div>'),
    )


def tracados(p: Problema, sol: Solucao, coords: list[tuple[float, float]]
             ) -> list[list[tuple[float, float]]]:
    """Traçado pelas ruas de cada rota (ou linha reta, se o OSRM falhar)."""
    resultado = []
    for rota in sol.rotas:
        deposito = p.tipos[rota.tipo].deposito
        pontos = [deposito] + rota.clientes + ([deposito] if p.retornar else [])
        sequencia = [coords[i] for i in pontos]
        resultado.append(geo.tracado_osrm(sequencia) or sequencia)
    return resultado


def desenhar(p: Problema, sol: Solucao, coords: list[tuple[float, float]],
             enderecos: list[str], linhas: list[list[tuple[float, float]]] | None = None
             ) -> folium.Map:
    """Monta o mapa com depósitos, paradas numeradas e o traçado de cada rota."""
    mapa = folium.Map(location=coords[p.depositos[0]], zoom_start=13, tiles=None,
                      control_scale=True)
    _fundos(mapa)
    Fullscreen(position="topleft").add_to(mapa)

    if linhas is None:
        linhas = [[coords[p.tipos[r.tipo].deposito]] + [coords[c] for c in r.clientes]
                  for r in sol.rotas]

    for k, (rota, linha) in enumerate(zip(sol.rotas, linhas)):
        cor = cor_da_rota(k)
        tipo = p.tipos[rota.tipo]
        camada = folium.FeatureGroup(
            name=f"Rota {k + 1} — {tipo.nome} ({rota.distancia:.1f} km)", show=True)

        # Contorno branco por baixo: separa rotas que dividem a mesma rua.
        folium.PolyLine(linha, color="#ffffff", weight=7, opacity=0.9).add_to(camada)
        folium.PolyLine(linha, color=cor, weight=4, opacity=0.95,
                        tooltip=f"Rota {k + 1} — {tipo.nome}").add_to(camada)

        for ordem, cliente in enumerate(rota.clientes):
            janela = ""
            if p.usar_janelas:
                janela = (f"<br>Janela: {_hora(p.jini[cliente])}–{_hora(p.jfim[cliente])}")
            espera = rota.inicio[ordem] - rota.chegada[ordem]
            popup = (f"<b>Rota {k + 1}, parada {ordem + 1}</b><br>{enderecos[cliente]}"
                     f"<br>Demanda: {p.dem[cliente]}"
                     f"<br>Chegada: {_hora(rota.chegada[ordem])}"
                     + (f" (espera {espera:.0f} min)" if espera > 0.5 else "")
                     + janela)
            folium.Marker(coords[cliente], icon=_marcador_numerado(ordem + 1, cor),
                          popup=folium.Popup(popup, max_width=280),
                          tooltip=f"{ordem + 1}. {enderecos[cliente]}").add_to(camada)
        camada.add_to(mapa)

    for d in p.depositos:
        folium.Marker(coords[d], tooltip=f"Depósito: {enderecos[d]}",
                      popup=folium.Popup(f"<b>Depósito</b><br>{enderecos[d]}", max_width=260),
                      icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(mapa)

    for c in sol.nao_atendidos:
        folium.Marker(coords[c], tooltip=f"NÃO ATENDIDO: {enderecos[c]}",
                      icon=folium.Icon(color="gray", icon="exclamation", prefix="fa")).add_to(mapa)

    todos = [coords[i] for i in p.depositos + p.clientes]
    mapa.fit_bounds([[min(a for a, _ in todos), min(b for _, b in todos)],
                     [max(a for a, _ in todos), max(b for _, b in todos)]], padding=(30, 30))
    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa


def mapa_de_conferencia(pontos: list[tuple[str, float, float, bool]]) -> folium.Map:
    """Mapa da etapa 1: só os pontos geocodificados, para o usuário conferir."""
    validos = [(n, la, lo, d) for n, la, lo, d in pontos if la is not None]
    centro = (validos[0][1], validos[0][2]) if validos else (-22.9, -43.1)
    mapa = folium.Map(location=centro, zoom_start=13, tiles="openstreetmap")
    for k, (nome, lat, lon, deposito) in enumerate(validos):
        if deposito:
            icone = folium.Icon(color="black", icon="home", prefix="fa")
        else:
            icone = _marcador_numerado(k + 1, "#2a78d6")
        folium.Marker((lat, lon), tooltip=nome, icon=icone).add_to(mapa)
    if len(validos) > 1:
        mapa.fit_bounds([[min(v[1] for v in validos), min(v[2] for v in validos)],
                         [max(v[1] for v in validos), max(v[2] for v in validos)]],
                        padding=(30, 30))
    return mapa
