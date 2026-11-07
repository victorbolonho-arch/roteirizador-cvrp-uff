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

def _normalizar(texto: str) -> str:
    """Minúsculas e sem acento, para comparar nomes."""
    import unicodedata
    sem_acento = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in sem_acento if not unicodedata.combining(c)).lower()


def _menciona_cidade(texto: str, cidade: str) -> bool:
    """O texto cita a cidade? (comparação sem acento e caixa)."""
    nome_cidade = _normalizar(cidade.split(",")[0].strip())
    return bool(nome_cidade) and nome_cidade in _normalizar(texto)


# Tipos de logradouro e palavras de ligação: não ajudam a identificar a rua.
_GENERICAS = {"rua", "r", "avenida", "av", "ave", "estrada", "estr", "est", "travessa", "tv",
              "trav", "praca", "pca", "pc", "alameda", "al", "largo", "lgo", "rodovia", "rod",
              "beco", "ladeira", "ld", "via", "vila", "de", "da", "do", "das", "dos", "e", "n", "no"}
_TIPOS_LOGRADOURO = _GENERICAS - {"de", "da", "do", "das", "dos", "e", "n", "no"}
# Abreviações comuns em endereços brasileiros (o OpenStreetMap usa o nome por extenso).
_ABREVIACOES = {"dr": "doutor", "prof": "professor", "profa": "professora", "pres": "presidente",
                "gal": "general", "gen": "general", "cel": "coronel", "mal": "marechal",
                "alm": "almirante", "sen": "senador", "dep": "deputado", "eng": "engenheiro",
                "visc": "visconde", "cons": "conselheiro", "des": "desembargador",
                "sta": "santa", "sto": "santo", "sao": "sao", "nsa": "nossa", "ns": "nossa"}


def _palavras(texto: str) -> list[str]:
    import re
    return re.findall(r"[a-z]+", _normalizar(texto))


def _palavras_da_rua(endereco: str) -> list[str]:
    """Palavras que identificam a rua: o 1º trecho do endereço, sem tipo nem número."""
    rua = endereco.split(",")[0]
    return [_ABREVIACOES.get(p, p) for p in _palavras(rua) if p not in _GENERICAS]


def _rua_confere(endereco: str, candidato: dict) -> bool:
    """O lugar achado é mesmo a rua digitada?

    Os serviços devolvem o "mais parecido" mesmo quando a rua não existe na
    cidade pedida: "Rua Pio Correa 80, Niterói" voltava como Rua Vilar Correa.
    Por isso exige-se que as palavras do nome da rua apareçam no nome achado
    (as abreviações valem pelo começo: "Prof." confere com "Professor").
    """
    procuradas = _palavras_da_rua(endereco)
    if not procuradas:
        return True
    tem_numero = any(ch.isdigit() for ch in endereco.split(",")[0])
    # Com número, é uma rua: confere-se o nome da rua. Sem número pode ser um
    # lugar ("Shopping Plaza"), e vale também o nome do lugar.
    alvo = candidato["rua"] if tem_numero else f"{candidato['rua']} {candidato['nome']}"
    achadas = set(_palavras(alvo))
    bate = sum(any(a.startswith(p) or p.startswith(a) for a in achadas if len(a) > 1)
               for p in procuradas)
    return bate >= max(1, math.ceil(0.75 * len(procuradas)))


def _consultar_nominatim(consulta: str) -> list[dict]:
    global _ultima_consulta
    espera = INTERVALO_NOMINATIM - (time.monotonic() - _ultima_consulta)
    if espera > 0:
        time.sleep(espera)
    try:
        resposta = requests.get(
            URL_NOMINATIM,
            params={"q": consulta, "format": "jsonv2", "limit": 5, "countrycodes": "br",
                    "addressdetails": 1},
            headers={"User-Agent": USER_AGENT, "Accept-Language": "pt-BR"},
            timeout=15,
        )
        _ultima_consulta = time.monotonic()
        resposta.raise_for_status()
        dados = resposta.json()
    except (requests.RequestException, ValueError):
        _ultima_consulta = time.monotonic()
        return []
    candidatos = []
    for d in dados:
        a = d.get("address", {})
        candidatos.append({
            "lat": float(d["lat"]), "lon": float(d["lon"]),
            "texto": d.get("display_name", ""), "nome": d.get("name") or "",
            "rua": a.get("road") or a.get("pedestrian") or a.get("square") or "",
            "cidade": a.get("city") or a.get("town") or a.get("municipality") or a.get("village") or "",
            "uf": a.get("ISO3166-2-lvl4", "")[-2:],
        })
    return candidatos


def _consultar_photon(consulta: str) -> list[dict]:
    try:
        resposta = requests.get(
            URL_PHOTON, params={"q": consulta, "limit": 5},
            headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        resposta.raise_for_status()
        dados = resposta.json().get("features", [])
    except (requests.RequestException, ValueError):
        return []
    candidatos = []
    for d in dados:
        if (d.get("properties", {}).get("countrycode") or "BR").upper() != "BR":
            continue
        lon, lat = d["geometry"]["coordinates"]
        p = d.get("properties", {})
        partes = [p.get("name"), p.get("street"), p.get("housenumber"), p.get("district"),
                  p.get("city"), p.get("state")]
        eh_rua = p.get("osm_key") == "highway"
        candidatos.append({
            "lat": float(lat), "lon": float(lon),
            "texto": ", ".join(str(x) for x in partes if x), "nome": p.get("name") or "",
            "rua": p.get("street") or (p.get("name") if eh_rua else "") or "",
            "cidade": p.get("city") or "", "uf": "",
        })
    return candidatos


def geocodificar(endereco: str, cidade_padrao: str = "") -> Local:
    """Converte um endereço em coordenadas.

    A cidade padrão desempata endereços que não dizem a cidade ("Rua da
    Conceição 100" existe em dezenas de cidades), mas não pode mandar no
    resultado: o endereço pode ser de outra cidade. A ordem das tentativas é:

      1. o endereço como foi digitado, aceito só se a cidade achada estiver
         escrita nele ("Av. Rio Branco 156, Centro, Rio de Janeiro");
      2. o endereço + a cidade padrão;
      3. o endereço como foi digitado, em qualquer cidade (de preferência no
         estado da cidade padrão);
      4. se faltou o tipo de logradouro ("São Clemente 244"), tudo de novo com
         "Rua" na frente.

    Em todas, o nome da rua achada tem que conferir com o digitado.
    """
    endereco = endereco.strip()
    if not endereco:
        return Local(endereco, None, None)

    cache = _ler_cache()
    chave = _chave(f"{endereco} | {cidade_padrao}")
    if chave in cache:
        c = cache[chave]
        return Local(endereco, c["lat"], c["lon"], c.get("encontrado", ""), "cache")

    partes = [p.strip() for p in cidade_padrao.split(",")]
    uf_padrao = partes[-1].upper() if partes and len(partes[-1]) == 2 else ""
    respostas: dict[tuple[str, str], list[dict]] = {}   # cada consulta vai ao serviço uma vez só

    def procurar(base: str, consulta: str, aceitar) -> Local | None:
        """Consulta os serviços e devolve o primeiro lugar que confere com `base`."""
        for fonte, consultar in (("nominatim", _consultar_nominatim),
                                 ("photon", _consultar_photon)):
            if (fonte, consulta) not in respostas:
                respostas[fonte, consulta] = consultar(consulta)
            candidatos = sorted(respostas[fonte, consulta],
                                key=lambda c: bool(uf_padrao) and c["uf"] not in ("", uf_padrao))
            for c in candidatos:
                if _rua_confere(base, c) and aceitar(c):
                    cache[chave] = {"lat": c["lat"], "lon": c["lon"], "encontrado": c["texto"]}
                    _gravar_cache(cache)
                    return Local(endereco, c["lat"], c["lon"], c["texto"], fonte)
        return None

    def cidade_escrita_ou_padrao(base: str):
        depois_da_rua = ",".join(base.split(",")[1:])
        return lambda c: bool(c["cidade"]) and (_menciona_cidade(depois_da_rua, c["cidade"])
                                                or _menciona_cidade(c["cidade"], cidade_padrao))

    variantes = [endereco]
    primeira = _palavras(endereco.split(",")[0])
    if primeira and primeira[0] not in _TIPOS_LOGRADOURO:
        variantes.append(f"Rua {endereco}")

    for base in variantes:
        achado = None
        if cidade_padrao and not _menciona_cidade(base, cidade_padrao):
            achado = (procurar(base, base, cidade_escrita_ou_padrao(base))
                      or procurar(base, f"{base}, {cidade_padrao}", lambda c: True))
        achado = achado or procurar(base, base, lambda c: True)
        if achado:
            return achado

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
