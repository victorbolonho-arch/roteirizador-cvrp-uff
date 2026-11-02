"""
Geolocalização dos endereços e matriz de distâncias.

Três serviços abertos, sem chave de API:

    Nominatim (OpenStreetMap)  endereço -> coordenadas
    Photon (Komoot)            reserva, se o Nominatim não achar o endereço
    OSRM                       distância e tempo PELAS RUAS entre todos os pares,
                               e o traçado de cada rota para desenhar no mapa

Se o OSRM estiver fora do ar, a matriz cai para a distância em linha reta
multiplicada por um fator de tortuosidade — e o app avisa que está usando essa
aproximação, porque ela erra bastante em cidades com morros e baías, como
Niterói.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

# O Nominatim exige um User-Agent que identifique a aplicação.
USER_AGENT = "roteirizador-cvrp-uff/1.0 (trabalho academico, Topicos Especiais UFF)"
URL_NOMINATIM = "https://nominatim.openstreetmap.org/search"
URL_PHOTON = "https://photon.komoot.io/api/"
URL_OSRM = "https://router.project-osrm.org"

# Política de uso do Nominatim: no máximo uma requisição por segundo.
INTERVALO_NOMINATIM = 1.05

# Aproximação usada quando o OSRM não responde.
FATOR_TORTUOSIDADE = 1.40     # rua ~40% mais longa que a reta (medido em Niterói: 1,56)
VELOCIDADE_MEDIA_KMH = 28.0   # trânsito urbano

ARQUIVO_CACHE = Path(__file__).parent / "dados" / "cache_geocodificacao.json"

_ultima_consulta = 0.0


@dataclass
class Local:
    """Resultado da geocodificação de um endereço."""
    endereco: str               # como o usuário digitou
    lat: float | None
    lon: float | None
    encontrado: str = ""        # endereço que o serviço devolveu
    fonte: str = ""             # nominatim, photon, cache ou manual

    @property
    def ok(self) -> bool:
        return self.lat is not None and self.lon is not None


# ---------------------------------------------------------------------------
# Cache em disco
# ---------------------------------------------------------------------------
#
# Geocodificar é lento (1 s por endereço) e o resultado não muda. O cache
# também faz o exemplo embutido abrir instantaneamente e funcionar mesmo sem
# internet.

def _ler_cache() -> dict:
    try:
        return json.loads(ARQUIVO_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _gravar_cache(cache: dict) -> None:
    try:
        ARQUIVO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ARQUIVO_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    except OSError:
        pass        # disco somente leitura (ex.: algumas hospedagens): segue sem cache


def _chave(endereco: str) -> str:
    return " ".join(endereco.lower().split())


# ---------------------------------------------------------------------------
# Geocodificação
# ---------------------------------------------------------------------------

def _menciona_cidade(endereco: str, cidade: str) -> bool:
    """O endereço já cita a cidade padrão? (comparação sem acento e caixa)."""
    import unicodedata

    def normalizar(texto: str) -> str:
        sem_acento = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in sem_acento if not unicodedata.combining(c)).lower()

    nome_cidade = normalizar(cidade.split(",")[0].strip())
    return bool(nome_cidade) and nome_cidade in normalizar(endereco)


def _consultar_nominatim(consulta: str) -> tuple[float, float, str] | None:
    global _ultima_consulta
    espera = INTERVALO_NOMINATIM - (time.monotonic() - _ultima_consulta)
    if espera > 0:
        time.sleep(espera)
    try:
        resposta = requests.get(
            URL_NOMINATIM,
            params={"q": consulta, "format": "jsonv2", "limit": 1, "countrycodes": "br"},
            headers={"User-Agent": USER_AGENT, "Accept-Language": "pt-BR"},
            timeout=15,
        )
        _ultima_consulta = time.monotonic()
        resposta.raise_for_status()
        dados = resposta.json()
    except (requests.RequestException, ValueError):
        _ultima_consulta = time.monotonic()
        return None
    if not dados:
        return None
    return float(dados[0]["lat"]), float(dados[0]["lon"]), dados[0].get("display_name", "")


def _consultar_photon(consulta: str) -> tuple[float, float, str] | None:
    try:
        resposta = requests.get(
            URL_PHOTON, params={"q": consulta, "limit": 1},
            headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        resposta.raise_for_status()
        dados = resposta.json().get("features", [])
    except (requests.RequestException, ValueError):
        return None
    if not dados:
        return None
    lon, lat = dados[0]["geometry"]["coordinates"]
    p = dados[0].get("properties", {})
    partes = [p.get("name"), p.get("street"), p.get("housenumber"), p.get("district"),
              p.get("city"), p.get("state")]
    return float(lat), float(lon), ", ".join(str(x) for x in partes if x)


def geocodificar(endereco: str, cidade_padrao: str = "") -> Local:
    """Converte um endereço em coordenadas, tentando do mais ao menos específico.

    Se o endereço não cita a cidade padrão, ela é acrescentada na primeira
    tentativa: "Rua da Conceição 100" sozinho existe em dezenas de cidades, e
    o serviço escolheria qualquer uma.
    """
    endereco = endereco.strip()
    if not endereco:
        return Local(endereco, None, None)

    cache = _ler_cache()
    chave = _chave(f"{endereco} | {cidade_padrao}")
    if chave in cache:
        c = cache[chave]
        return Local(endereco, c["lat"], c["lon"], c.get("encontrado", ""), "cache")

    tentativas = []
    if cidade_padrao and not _menciona_cidade(endereco, cidade_padrao):
        tentativas.append(f"{endereco}, {cidade_padrao}")
    tentativas.append(endereco)

    for consulta in tentativas:
        for fonte, consultar in (("nominatim", _consultar_nominatim),
                                 ("photon", _consultar_photon)):
            achado = consultar(consulta)
            if achado:
                lat, lon, texto = achado
                cache[chave] = {"lat": lat, "lon": lon, "encontrado": texto}
                _gravar_cache(cache)
                return Local(endereco, lat, lon, texto, fonte)

    return Local(endereco, None, None, "não encontrado", "")


# ---------------------------------------------------------------------------
# Matriz de distâncias e tempos
# ---------------------------------------------------------------------------

def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distância em linha reta sobre a superfície da Terra, em km."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def matriz_linha_reta(coords: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Aproximação sem internet: reta x tortuosidade, a velocidade média fixa."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = haversine_km(coords[i], coords[j]) * FATOR_TORTUOSIDADE
    tempo = dist / VELOCIDADE_MEDIA_KMH * 60.0
    return dist, tempo


def _texto_coords(coords) -> str:
    return ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)


def matriz_osrm(coords: list[tuple[float, float]], bloco: int = 50
                ) -> tuple[np.ndarray, np.ndarray]:
    """Distância (km) e tempo (min) pelas ruas entre todos os pares de pontos.

    A matriz é ASSIMÉTRICA: ir de A a B pode ser mais longo que voltar, por
    causa das ruas de mão única. O servidor público limita o tamanho de cada
    consulta, por isso a matriz é montada em blocos de `bloco` x `bloco`.
    """
    n = len(coords)
    dist = np.zeros((n, n))
    tempo = np.zeros((n, n))

    for ini_o in range(0, n, bloco):
        origens = list(range(ini_o, min(ini_o + bloco, n)))
        for ini_d in range(0, n, bloco):
            destinos = list(range(ini_d, min(ini_d + bloco, n)))
            pontos = sorted(set(origens) | set(destinos))
            posicao = {p: k for k, p in enumerate(pontos)}

            url = (f"{URL_OSRM}/table/v1/driving/{_texto_coords([coords[p] for p in pontos])}")
            resposta = requests.get(url, params={
                "annotations": "distance,duration",
                "sources": ";".join(str(posicao[p]) for p in origens),
                "destinations": ";".join(str(posicao[p]) for p in destinos),
            }, headers={"User-Agent": USER_AGENT}, timeout=60)
            resposta.raise_for_status()
            dados = resposta.json()
            if dados.get("code") != "Ok":
                raise RuntimeError(f"OSRM respondeu {dados.get('code')}: {dados.get('message')}")

            for a, i in enumerate(origens):
                for b, j in enumerate(destinos):
                    d = dados["distances"][a][b]
                    t = dados["durations"][a][b]
                    if d is None or t is None:
                        raise RuntimeError("OSRM não encontrou caminho entre dois pontos")
                    dist[i, j] = d / 1000.0
                    tempo[i, j] = t / 60.0

    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(tempo, 0.0)
    return dist, tempo


def calcular_matriz(coords: list[tuple[float, float]], usar_ruas: bool = True
                    ) -> tuple[np.ndarray, np.ndarray, str]:
    """Matriz pelas ruas se possível; senão, a aproximação. Devolve também a fonte."""
    if usar_ruas:
        try:
            dist, tempo = matriz_osrm(coords)
            return dist, tempo, "ruas (OSRM)"
        except (requests.RequestException, RuntimeError, ValueError, KeyError):
            pass
    dist, tempo = matriz_linha_reta(coords)
    return dist, tempo, f"linha reta × {FATOR_TORTUOSIDADE}"


# ---------------------------------------------------------------------------
# Traçado das rotas
# ---------------------------------------------------------------------------

def tracado_osrm(coords: list[tuple[float, float]], bloco: int = 90
                 ) -> list[tuple[float, float]] | None:
    """Sequência de (lat, lon) seguindo as ruas, passando pelos pontos em ordem.

    Devolve None se o serviço falhar — o mapa então liga os pontos em linha reta.
    """
    if len(coords) < 2:
        return list(coords)
    caminho: list[tuple[float, float]] = []
    inicio = 0
    while inicio < len(coords) - 1:
        trecho = coords[inicio:inicio + bloco]
        try:
            resposta = requests.get(
                f"{URL_OSRM}/route/v1/driving/{_texto_coords(trecho)}",
                params={"overview": "full", "geometries": "geojson"},
                headers={"User-Agent": USER_AGENT}, timeout=60,
            )
            resposta.raise_for_status()
            dados = resposta.json()
            if dados.get("code") != "Ok":
                return None
            linha = [(lat, lon) for lon, lat in dados["routes"][0]["geometry"]["coordinates"]]
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None
        caminho.extend(linha if not caminho else linha[1:])
        inicio += bloco - 1          # o último ponto de um trecho abre o próximo
    return caminho
