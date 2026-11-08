"""
Roteirizador CVRP — trabalho final de Tópicos Especiais (UFF).
Victor Balieiro e Fabiano Lucas.

    Etapa 1  pontos: endereços -> coordenadas -> matriz de distâncias pelas ruas
    Etapa 2  frota: tipos de veículo, quantidades, custos, depósitos, jornada
    Etapa 3  rotas: GRASP + VRPSolverEasy, mapa, plano de visitas, exportação

Rodar:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import exportar
import geo
import heuristica
import mapa
import solver
from modelo import HORIZONTE, Problema, TipoVeiculo, avaliar_rota, verificar

PASTA = Path(__file__).parent
EXEMPLO = PASTA / "dados" / "exemplo_niteroi.csv"

COLUNAS_PONTOS = ["Endereço", "Demanda", "Depósito", "Atendimento (min)"]
COLUNAS_FROTA = ["Tipo", "Capacidade", "Quantidade", "Custo fixo (R$)", "Custo por km (R$)",
                 "Custo por hora (R$)", "Depósito", "Início da jornada", "Fim da jornada"]

st.set_page_config(page_title="Roteirizador CVRP — UFF", page_icon="🚚", layout="wide")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def minutos(texto) -> float | None:
    """'08:30' -> 510. Vazio -> None. Aceita também '8h30' e '8'."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return None
    texto = str(texto).strip().lower().replace("h", ":")
    if not texto:
        return None
    if ":" not in texto:
        texto += ":00"
    hora, _, minuto = texto.partition(":")
    try:
        return int(hora) * 60 + int(minuto or 0)
    except ValueError:
        return None


def hora(m: float) -> str:
    m = int(round(m))
    return f"{m // 60 % 24:02d}:{m % 60:02d}"


def brl(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tabela_vazia() -> pd.DataFrame:
    return pd.DataFrame([{"Endereço": "", "Demanda": 0, "Depósito": True,
                          "Atendimento (min)": 0}])


def ler_planilha(arquivo) -> pd.DataFrame:
    """Lê CSV/Excel e encaixa as colunas, tolerando nomes aproximados."""
    if arquivo.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(arquivo)
    else:
        bruto = arquivo.getvalue().decode("utf-8-sig", errors="replace")
        separador = ";" if bruto.count(";") > bruto.count(",") else ","
        df = pd.read_csv(io.StringIO(bruto), sep=separador)

    apelidos = {"endereço": "Endereço", "endereco": "Endereço", "address": "Endereço",
                "demanda": "Demanda", "demand": "Demanda", "depósito": "Depósito",
                "deposito": "Depósito", "atendimento": "Atendimento (min)",
                "atendimento (min)": "Atendimento (min)"}
    df = df.rename(columns={c: apelidos.get(str(c).strip().lower(), c) for c in df.columns})
    if "Endereço" not in df.columns:
        raise ValueError("a planilha precisa de uma coluna 'Endereço'")
    for coluna, padrao in (("Demanda", 0), ("Depósito", False), ("Atendimento (min)", 0)):
        if coluna not in df.columns:
            df[coluna] = padrao
    df["Depósito"] = df["Depósito"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "sim", "s", "x", "verdadeiro"])
    return df[COLUNAS_PONTOS]


def frota_padrao(depositos: list[str]) -> pd.DataFrame:
    primeiro = depositos[0] if depositos else ""
    return pd.DataFrame([
        {"Tipo": "Van", "Capacidade": 300, "Quantidade": 3, "Custo fixo (R$)": 80.0,
         "Custo por km (R$)": 1.20, "Custo por hora (R$)": 25.0, "Depósito": primeiro,
         "Início da jornada": "07:00", "Fim da jornada": "19:00"},
        # Cadastrado com quantidade 0: mostra a frota heterogênea sem que ela
        # entre no cálculo padrão. Com tipos de custo fixo diferente o
        # VRPSolverEasy (com o solver livre CLP) raramente prova o ótimo em
        # menos de um minuto; com um tipo só, prova em ~1 s.
        {"Tipo": "Caminhão 3/4", "Capacidade": 600, "Quantidade": 0, "Custo fixo (R$)": 150.0,
         "Custo por km (R$)": 2.00, "Custo por hora (R$)": 35.0, "Depósito": primeiro,
         "Início da jornada": "07:00", "Fim da jornada": "19:00"},
    ])


@st.cache_data(show_spinner=False)
def tracados_em_cache(coords_rotas: tuple) -> list:
    return [geo.tracado_osrm(list(seq)) or list(seq) for seq in coords_rotas]


def estado(chave, padrao):
    if chave not in st.session_state:
        st.session_state[chave] = padrao
    return st.session_state[chave]


estado("etapa", 1)
estado("pontos", tabela_vazia())
estado("versao_tabela", 0)

# Opções da lista de cidades padrão. Dá para digitar outra direto na lista.
SEM_CIDADE = "Nenhuma (os endereços já dizem a cidade)"
CIDADES = ["Niterói, RJ", "Rio de Janeiro, RJ", "São Gonçalo, RJ", "Maricá, RJ",
           "Itaboraí, RJ", "Duque de Caxias, RJ", "Nova Iguaçu, RJ", "Petrópolis, RJ",
           "São Paulo, SP", "Belo Horizonte, MG", "Vitória, ES", "Curitiba, PR",
           "Porto Alegre, RS", "Brasília, DF", "Salvador, BA", "Recife, PE",
           "Fortaleza, CE", SEM_CIDADE]
estado("cidade", CIDADES[0])         # a cidade em uso; a lista só muda ao clicar Aplicar


def ir(etapa: int):
    st.session_state.etapa = etapa


def aplicar_cidade():
    """Passa a usar a cidade escolhida; endereços já buscados são buscados de novo."""
    escolha = (st.session_state.cidade_escolha or "").strip()
    st.session_state.cidade = escolha or SEM_CIDADE
    if "locais" in st.session_state:
        for chave in ("locais", "matriz", "resultado"):
            st.session_state.pop(chave, None)
        st.session_state.regeocodificar = True
        ir(1)
    st.toast(f"Cidade padrão: {st.session_state.cidade}", icon="📍")


def cidade_em_uso() -> str:
    """A cidade que completa os endereços ('' quando nenhuma)."""
    return "" if st.session_state.cidade == SEM_CIDADE else st.session_state.cidade


def carregar_exemplo():
    st.session_state.pontos = pd.read_csv(EXEMPLO)
    st.session_state.versao_tabela += 1
    for chave in ("locais", "matriz", "frota", "resultado"):
        st.session_state.pop(chave, None)
    # Os endereços do exemplo não dizem a cidade: dependem de Niterói como padrão.
    st.session_state.cidade = st.session_state.cidade_escolha = "Niterói, RJ"
    ir(1)


def recomecar():
    for chave in list(st.session_state.keys()):
        del st.session_state[chave]


def alcancavel(etapa: int) -> bool:
    """Só dá para ir a uma etapa cujos dados já existem."""
    return etapa == 1 or (etapa == 2 and "matriz" in st.session_state) or \
        (etapa == 3 and "resultado" in st.session_state)


NOMES_ETAPAS = ["Pontos", "Frota", "Rotas"]


def navegacao(atual: int):
    """Botões no topo da página para voltar às etapas anteriores ou começar de novo."""
    colunas = st.columns([1.2] * (atual - 1) + [1.4, 6 - atual])
    for k in range(1, atual):
        colunas[k - 1].button(f"← {k}. {NOMES_ETAPAS[k - 1]}", on_click=ir, args=(k,),
                              width="stretch", key=f"nav_{atual}_{k}")
    colunas[atual - 1].button("🏠 Novo roteiro", on_click=recomecar, width="stretch",
                              key=f"nav_{atual}_inicio",
                              help="Apaga tudo e volta para a tela inicial.")


def ajuda(texto: str):
    """Botão ⓘ que abre uma explicação."""
    with st.popover("ⓘ", help="O que é isto?"):
        st.markdown(texto)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🚚 Roteirizador CVRP")
    st.caption("Tópicos Especiais · UFF  \nVictor Balieiro e Fabiano Lucas")

    for k, nome in enumerate(NOMES_ETAPAS, start=1):
        atual = k == st.session_state.etapa
        marca = "▶" if atual else ("✅" if alcancavel(k) else "○")
        st.button(f"{marca} {k}. {nome}", on_click=ir, args=(k,), width="stretch",
                  type="primary" if atual else "tertiary", disabled=not alcancavel(k),
                  key=f"lateral_{k}")
    st.caption("Clique numa etapa já feita para voltar a ela.")

    st.divider()
    opcoes = CIDADES if st.session_state.cidade in CIDADES else [st.session_state.cidade] + CIDADES
    escolha = st.selectbox(
        "Cidade padrão dos endereços", opcoes, index=opcoes.index(st.session_state.cidade),
        key="cidade_escolha", accept_new_options=True,
        help="Completa os endereços que não dizem a cidade ('Rua da Conceição 100' existe "
             "em dezenas de cidades). Clique na setinha para ver outras cidades, ou digite "
             "o nome de outra. Se o endereço disser a cidade, vale a do endereço.")
    st.button("✔ Aplicar cidade", on_click=aplicar_cidade, width="stretch",
              disabled=(escolha or SEM_CIDADE) == st.session_state.cidade,
              help="Passa a usar esta cidade. Se os endereços já foram buscados, "
                   "eles são buscados de novo.")
    st.caption(f"Em uso: **{st.session_state.cidade}**")
    fonte_distancia = st.radio(
        "Distâncias", ["Pelas ruas (OSRM)", "Linha reta corrigida"],
        help="Pelas ruas usa o traçado real e as mãos de direção. A linha reta é só "
             "uma aproximação para quando não houver internet.")

    st.button("📍 Carregar exemplo de Niterói", on_click=carregar_exemplo, width="stretch")

    st.button("🏠 Novo roteiro (recomeçar)", on_click=recomecar, width="stretch")

    with st.expander("Como funciona"):
        st.markdown(
            "- **Geocodificação**: Nominatim (OpenStreetMap), com o Photon de reserva.\n"
            "- **Distâncias e tempos**: OSRM, pelas ruas — a matriz é assimétrica "
            "por causa das mãos de direção.\n"
            "- **Roteamento**: primeiro o **GRASP** (trabalho 4) acha uma boa solução "
            "em segundos; ela entra como limitante superior no **VRPSolverEasy**, "
            "que melhora a solução ou **prova que ela é ótima**.")


# ---------------------------------------------------------------------------
# Etapa 1 — Pontos
# ---------------------------------------------------------------------------

def etapa_pontos():
    st.title("Etapa 1 · Pontos de atendimento")
    st.write("Informe o depósito e os clientes pelo endereço. Marque **Depósito** em ao "
             "menos um ponto.")

    aba_digitar, aba_importar = st.tabs(["✏️ Digitar", "📄 Importar planilha"])

    with aba_importar:
        st.caption("CSV ou Excel com a coluna **Endereço** e, opcionalmente, Demanda, "
                   "Depósito, Atendimento (min).")
        arquivo = st.file_uploader("Planilha de pontos", type=["csv", "xlsx", "xls"],
                                   label_visibility="collapsed")
        if arquivo is not None and st.button("Usar esta planilha"):
            try:
                st.session_state.pontos = ler_planilha(arquivo)
                st.session_state.versao_tabela += 1
                st.session_state.pop("locais", None)
                st.rerun()
            except Exception as erro:
                st.error(f"Não consegui ler a planilha: {erro}")
        modelo_csv = pd.read_csv(EXEMPLO).to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar planilha-modelo", modelo_csv, "modelo_pontos.csv", "text/csv")

    with aba_digitar:
        editada = st.data_editor(
            st.session_state.pontos, num_rows="dynamic", width="stretch",
            key=f"editor_pontos_{st.session_state.versao_tabela}",
            column_config={
                "Endereço": st.column_config.TextColumn(width="large", required=True),
                "Demanda": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
                "Depósito": st.column_config.CheckboxColumn(),
                "Atendimento (min)": st.column_config.NumberColumn(min_value=0, step=1,
                                                                   help="Tempo parado no cliente"),
            })

    pontos = editada.copy()
    pontos["Endereço"] = pontos["Endereço"].fillna("").astype(str).str.strip()
    pontos = pontos[pontos["Endereço"] != ""].reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Depósitos", int(pontos["Depósito"].fillna(False).astype(bool).sum()))
    c2.metric("Clientes", int((~pontos["Depósito"].fillna(False).astype(bool)).sum()))
    c3.metric("Demanda total", int(pd.to_numeric(pontos["Demanda"], errors="coerce").fillna(0)
                                   [~pontos["Depósito"].fillna(False).astype(bool)].sum()))

    # "Aplicar cidade" com os endereços já buscados pede uma nova busca.
    refazer = st.session_state.pop("regeocodificar", False) and not pontos.empty
    if st.button("🌎 Geocodificar endereços", type="primary") or refazer:
        if pontos.empty:
            st.error("Informe ao menos um endereço.")
            return
        st.session_state.pontos = pontos
        barra = st.progress(0.0, text="Geocodificando…")
        locais = []
        for k, endereco in enumerate(pontos["Endereço"]):
            barra.progress((k + 1) / len(pontos), text=f"Geocodificando {endereco}")
            locais.append(geo.geocodificar(endereco, cidade_em_uso()))
        barra.empty()
        st.session_state.locais = locais
        st.session_state.pop("matriz", None)
        st.rerun()

    if "locais" in st.session_state and len(st.session_state.locais) == len(pontos):
        conferir_geocodificacao(pontos)


def conferir_geocodificacao(pontos: pd.DataFrame):
    st.subheader("Confira a localização")
    st.caption("Se algum ponto caiu no lugar errado, corrija a latitude e a longitude na "
               "tabela (no Google Maps, clique com o botão direito no local para copiá-las).")

    locais = st.session_state.locais
    revisao = pd.DataFrame({
        "Endereço": [l.endereco for l in locais],
        "Encontrado": [l.encontrado for l in locais],
        "Latitude": [l.lat for l in locais],
        "Longitude": [l.lon for l in locais],
        "Situação": ["✅" if l.ok else "❌ não encontrado" for l in locais],
    })
    revisao = st.data_editor(
        revisao, width="stretch", hide_index=True,
        disabled=["Endereço", "Encontrado", "Situação"],
        column_config={"Latitude": st.column_config.NumberColumn(format="%.6f"),
                       "Longitude": st.column_config.NumberColumn(format="%.6f")},
        key="editor_revisao")

    depositos = pontos["Depósito"].fillna(False).astype(bool).tolist()
    st_folium(mapa.mapa_de_conferencia(
        [(e, la if pd.notna(la) else None, lo if pd.notna(lo) else None, d)
         for e, la, lo, d in zip(revisao["Endereço"], revisao["Latitude"],
                                  revisao["Longitude"], depositos)]),
        height=420, width="stretch", returned_objects=[])

    faltando = revisao[revisao["Latitude"].isna() | revisao["Longitude"].isna()]
    if not faltando.empty:
        st.warning(f"{len(faltando)} endereço(s) sem coordenadas. Corrija o endereço e "
                   "geocodifique de novo, ou informe a latitude e a longitude à mão.")

    if st.button("Confirmar pontos e calcular distâncias →", type="primary",
                 disabled=not faltando.empty):
        if not any(depositos):
            st.error("Marque ao menos um ponto como depósito.")
            return
        if all(depositos):
            st.error("Todos os pontos estão marcados como depósito — não há clientes.")
            return
        coords = list(zip(revisao["Latitude"].astype(float), revisao["Longitude"].astype(float)))
        with st.spinner("Calculando distâncias e tempos entre todos os pares de pontos…"):
            dist, tempo, fonte = geo.calcular_matriz(
                coords, usar_ruas=fonte_distancia.startswith("Pelas ruas"))
        st.session_state.matriz = {"coords": coords, "dist": dist, "tempo": tempo,
                                   "fonte": fonte, "pontos": pontos.copy()}
        st.session_state.pop("resultado", None)
        ir(2)
        st.rerun()


# ---------------------------------------------------------------------------
# Etapa 2 — Frota
# ---------------------------------------------------------------------------

def montar_problema(frota: pd.DataFrame, retornar: bool) -> tuple[Problema, list, list[str]]:
    """Converte as tabelas da interface no `Problema`, com os depósitos na frente."""
    m = st.session_state.matriz
    pontos = m["pontos"]
    eh_dep = pontos["Depósito"].fillna(False).astype(bool).tolist()
    ordem = [i for i, d in enumerate(eh_dep) if d] + [i for i, d in enumerate(eh_dep) if not d]
    n_dep = sum(eh_dep)

    enderecos = [pontos["Endereço"].iloc[i] for i in ordem]
    coords = [m["coords"][i] for i in ordem]
    dist = m["dist"][np.ix_(ordem, ordem)]
    tempo = m["tempo"][np.ix_(ordem, ordem)]

    demanda = pd.to_numeric(pontos["Demanda"], errors="coerce").fillna(0).astype(int).to_numpy()[ordem]
    servico = pd.to_numeric(pontos["Atendimento (min)"], errors="coerce").fillna(0).to_numpy(float)[ordem]
    for novo, antigo in enumerate(ordem):
        if eh_dep[antigo]:
            demanda[novo], servico[novo] = 0, 0.0

    nome_para_indice = {enderecos[k]: k for k in range(n_dep)}
    tipos = []
    for _, linha in frota.iterrows():
        if not str(linha["Tipo"]).strip() or pd.isna(linha["Capacidade"]):
            continue
        tipos.append(TipoVeiculo(
            nome=str(linha["Tipo"]).strip(),
            capacidade=int(linha["Capacidade"]),
            quantidade=int(linha["Quantidade"] or 0),
            custo_fixo=float(linha["Custo fixo (R$)"] or 0),
            custo_km=float(linha["Custo por km (R$)"] or 0),
            custo_hora=float(linha["Custo por hora (R$)"] or 0),
            deposito=nome_para_indice.get(linha["Depósito"], 0),
            jornada_ini=minutos(linha["Início da jornada"]) or 0.0,
            jornada_fim=minutos(linha["Fim da jornada"]) or HORIZONTE,
        ))
    tipos = [t for t in tipos if t.quantidade > 0 and t.capacidade > 0]

    p = Problema(nomes=enderecos, depositos=list(range(n_dep)),
                 clientes=list(range(n_dep, len(ordem))), demanda=demanda, servico=servico,
                 janela_ini=np.zeros(len(ordem)), janela_fim=np.full(len(ordem), HORIZONTE),
                 dist=dist, tempo=tempo, tipos=tipos, retornar=retornar)
    return p, coords, enderecos


def diagnostico(p: Problema) -> list[str]:
    """Problemas que tornariam o roteamento impossível, explicados em português."""
    erros = []
    if not p.tipos:
        return ["Cadastre ao menos um tipo de veículo com capacidade e quantidade."]
    if p.demanda_total > p.capacidade_total:
        erros.append(f"A frota leva {p.capacidade_total} unidades no total, mas a demanda é "
                     f"{p.demanda_total}. Aumente a quantidade ou a capacidade dos veículos.")
    maior = max(t.capacidade for t in p.tipos)
    for c in p.clientes:
        if p.dem[c] > maior:
            erros.append(f"'{p.nomes[c]}' pede {p.dem[c]} unidades, mais que o maior veículo "
                         f"({maior}).")
        elif all(avaliar_rota(p, k, [c]) is None for k in range(len(p.tipos))):
            erros.append(f"Nenhum veículo consegue atender '{p.nomes[c]}' sozinho dentro da "
                         "jornada. Aumente o horário de fim da jornada.")
    return erros


AJUDA_FROTA = """
**Cada linha é um _tipo_ de veículo.** Três vans iguais = uma linha com
Quantidade 3. Dois modelos diferentes = duas linhas.

| Coluna | O que é | Exemplo |
|---|---|---|
| **Tipo** | Um nome para você identificar | Van |
| **Capacidade** | Quanto cabe em **um** veículo, na mesma unidade da *Demanda* da etapa 1 (caixas, kg, volumes…) | 300 |
| **Quantidade** | Quantos veículos desse tipo você tem. **0 = não usar** | 3 |
| **Custo fixo** | Pago uma vez por veículo que sai (diária do motorista, aluguel). Por isso o app evita sair com veículo à toa | R$ 80 |
| **Custo por km** | Combustível, pneu, manutenção | R$ 1,20 |
| **Custo por hora** | Hora do motorista, sobre o tempo dirigindo | R$ 25 |
| **Depósito** | De onde o veículo sai (e para onde volta) | o ponto marcado como depósito |
| **Início / Fim da jornada** | Horário em que o veículo pode rodar (HH:MM) | 07:00 e 19:00 |

**Como o custo de uma rota é calculado**

custo fixo + km rodados × custo por km + horas dirigindo × custo por hora

Ex.: uma van que roda 40 km em 1h30 custa 80 + 40 × 1,20 + 1,5 × 25 = **R$ 165,50**.
O app escolhe as rotas que somam o **menor custo total**.

**Editar a tabela:** clique numa célula para mudar o valor. Para incluir um tipo,
use a linha vazia no fim da tabela. Para apagar, marque a caixinha à esquerda
da linha e clique na lixeira.

O *Caminhão 3/4* vem com Quantidade 0 só de exemplo: mude para 1 para usar
a frota mista (o cálculo exato fica mais demorado).
"""

AJUDA_OPCOES = """
**Veículos retornam ao depósito** — ligado, cada rota termina de volta no
depósito. Desligado, termina no último cliente (ex.: o motorista leva o veículo
para casa) e a volta não é cobrada.

**Método**
- **GRASP + VRPSolverEasy (recomendado)**: o GRASP (heurística do trabalho 4)
  acha uma boa solução em segundos; o VRPSolverEasy (método exato) tenta
  melhorá-la ou **prova que é a melhor possível**.
- **Só GRASP**: mais rápido, mas sem garantia de ser a melhor.
- **Só VRPSolverEasy**: o método exato sem a ajuda da heurística.

**Tempos** — quanto cada método pode rodar, no máximo. No exemplo de Niterói o
ótimo sai em cerca de 1 s. Só vale aumentar se o resultado mostrar um *gap*
(diferença para o ótimo) que você queira fechar.
"""


def etapa_frota():
    st.title("Etapa 2 · Frota")
    navegacao(2)
    st.write("Cadastre os veículos que vão fazer as entregas e escolha as opções do "
             "cálculo. Na dúvida, clique no **ⓘ** ao lado de cada parte.")
    m = st.session_state.matriz
    pontos = m["pontos"]
    eh_dep = pontos["Depósito"].fillna(False).astype(bool)
    nomes_depositos = pontos.loc[eh_dep, "Endereço"].tolist()
    demanda = int(pd.to_numeric(pontos.loc[~eh_dep, "Demanda"], errors="coerce").fillna(0).sum())

    st.caption(f"Matriz de distâncias: **{m['fonte']}** · {len(pontos)} pontos")
    if not m["fonte"].startswith("ruas"):
        st.warning("O serviço de rotas não respondeu; as distâncias são uma aproximação "
                   "em linha reta. Volte à etapa 1 e confirme de novo para tentar outra vez.")

    titulo, botao = st.columns([12, 1], vertical_alignment="bottom")
    titulo.subheader("Veículos disponíveis")
    with botao:
        ajuda(AJUDA_FROTA)
    st.caption("Passe o mouse sobre o nome de uma coluna para ver o que ela significa.")

    frota = estado("frota", frota_padrao(nomes_depositos))
    frota.loc[~frota["Depósito"].isin(nomes_depositos), "Depósito"] = nomes_depositos[0]
    frota = st.data_editor(
        frota, num_rows="dynamic", width="stretch", key="editor_frota",
        column_config={
            "Tipo": st.column_config.TextColumn(
                help="Nome do tipo de veículo, para você identificar (ex.: Van)."),
            "Capacidade": st.column_config.NumberColumn(min_value=1, step=1, format="%d",
                help="Quanto cabe em UM veículo, na mesma unidade da Demanda da etapa 1."),
            "Quantidade": st.column_config.NumberColumn(min_value=0, step=1, format="%d",
                help="Quantos veículos deste tipo existem. 0 = não usar este tipo."),
            "Custo fixo (R$)": st.column_config.NumberColumn(min_value=0.0, format="%.2f",
                help="Pago uma vez por veículo que sai: diária do motorista, aluguel…"),
            "Custo por km (R$)": st.column_config.NumberColumn(min_value=0.0, format="%.2f",
                help="Por km rodado: combustível, pneus, manutenção."),
            "Custo por hora (R$)": st.column_config.NumberColumn(min_value=0.0, format="%.2f",
                help="Por hora dirigindo: salário do motorista."),
            "Depósito": st.column_config.SelectboxColumn(options=nomes_depositos, required=True,
                help="De onde o veículo sai e para onde volta."),
            "Início da jornada": st.column_config.TextColumn(
                help="A partir de que horas o veículo pode sair (HH:MM, ex.: 07:00)."),
            "Fim da jornada": st.column_config.TextColumn(
                help="Até que horas o veículo precisa terminar a rota (HH:MM, ex.: 19:00)."),
        })
    st.session_state.frota = frota

    titulo, botao = st.columns([12, 1], vertical_alignment="bottom")
    titulo.subheader("Opções do cálculo")
    with botao:
        ajuda(AJUDA_OPCOES)

    col1, col2 = st.columns(2)
    with col1:
        retornar = st.toggle("Veículos retornam ao depósito", value=True,
                             help="Desligado: a rota termina no último cliente e a volta "
                                  "não é cobrada.")
    with col2:
        metodo = st.radio("Método", [
            "GRASP + VRPSolverEasy (recomendado)", "Só VRPSolverEasy", "Só GRASP"],
            help="No recomendado, o GRASP acha uma solução em segundos e o VRPSolverEasy "
                 "tenta melhorá-la ou provar que ela é ótima.")
        tempo_exato = st.slider("Tempo máximo do VRPSolverEasy (s)", 10, 300, 60, 10,
                                help="Se acabar antes de provar o ótimo, o app mostra a "
                                     "melhor solução achada e o gap.")
        tempo_grasp = st.slider("Tempo do GRASP (s)", 1, 30, 5,
                                help="Quanto mais tempo, mais tentativas a heurística faz.")

    p, coords, enderecos = montar_problema(frota, retornar)
    capacidade = p.capacidade_total
    cor = "green" if capacidade >= demanda else "red"
    st.markdown(f"Capacidade da frota: **:{cor}[{capacidade}]** · Demanda: **{demanda}** · "
                f"Veículos disponíveis: **{sum(t.quantidade for t in p.tipos)}**")

    erros = diagnostico(p)
    for erro in erros:
        st.error(erro)

    voltar, calcular, _ = st.columns([1, 2, 5])
    voltar.button("← Voltar", on_click=ir, args=(1,))
    if calcular.button("🧮 Calcular rotas", type="primary", disabled=bool(erros)):
        calcular_rotas(p, coords, enderecos, metodo, tempo_exato, tempo_grasp)


def calcular_rotas(p, coords, enderecos, metodo, tempo_exato, tempo_grasp):
    solucoes = {}
    with st.status("Calculando rotas…", expanded=True) as caixa:
        if metodo != "Só VRPSolverEasy":
            st.write(f"GRASP: construção aleatorizada + busca local por {tempo_grasp} s…")
            solucoes["GRASP"] = heuristica.grasp(p, tempo_limite=tempo_grasp)
            st.write(f"→ {brl(solucoes['GRASP'].custo)} em {solucoes['GRASP'].iteracoes} iterações")
        if metodo != "Só GRASP":
            st.write(f"VRPSolverEasy: branch-cut-and-price por até {tempo_exato} s…")
            partida = solucoes.get("GRASP")
            solucoes["VRPSolverEasy"] = solver.resolver_exato(p, tempo_exato, partida=partida)
            st.write(f"→ {solucoes['VRPSolverEasy'].status}")
        caixa.update(label="Rotas calculadas", state="complete", expanded=False)

    validas = {k: s for k, s in solucoes.items() if s.rotas}
    if not validas:
        st.error("Nenhum método encontrou solução. " +
                 " / ".join(f"{k}: {s.status}" for k, s in solucoes.items()))
        return
    # Fica a de menos clientes sem atendimento; entre essas, a mais barata; e,
    # no empate de custo, a que vem com prova de otimalidade.
    melhor = min(validas, key=lambda k: (len(validas[k].nao_atendidos),
                                         round(validas[k].custo, 4), not validas[k].otima))
    st.session_state.resultado = {"problema": p, "coords": coords, "enderecos": enderecos,
                                  "solucoes": solucoes, "escolhida": melhor}
    ir(3)
    st.rerun()


# ---------------------------------------------------------------------------
# Etapa 3 — Rotas
# ---------------------------------------------------------------------------

def etapa_rotas():
    r = st.session_state.resultado
    p, coords, enderecos = r["problema"], r["coords"], r["enderecos"]
    solucoes = r["solucoes"]
    sol = solucoes[r["escolhida"]]

    st.title("Etapa 3 · Rotas")
    navegacao(3)

    erros = verificar(p, sol)
    if sol.otima:
        st.success("**Solução ótima** — o VRPSolverEasy provou que não existe plano mais "
                   f"barato com esta frota ({sol.tempo_s:.1f} s).", icon="🏆")
    elif sol.gap is not None:
        st.info(f"Solução viável, no máximo **{sol.gap:.2f}%** acima do ótimo "
                f"(limitante inferior {brl(sol.limitante_inferior)}). Aumente o tempo do "
                f"VRPSolverEasy para tentar fechar essa diferença.", icon="📐")
    else:
        st.info(f"Solução do {sol.metodo}: {sol.status}", icon="🧭")
    if sol.nao_atendidos:
        st.warning("Sem veículo que atenda: " + "; ".join(enderecos[c] for c in sol.nao_atendidos))
    if erros:
        st.error("Conferência da solução falhou: " + "; ".join(erros))

    usados = len(sol.rotas)
    disponiveis = sum(t.quantidade for t in p.tipos)
    ocupacao = 100.0 * sum(x.carga for x in sol.rotas) / max(1, sum(p.tipos[x.tipo].capacidade for x in sol.rotas))
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Custo total", brl(sol.custo))
    k2.metric("Distância", f"{sol.distancia:.1f} km")
    k3.metric("Veículos", f"{usados} de {disponiveis}")
    k4.metric("Dirigindo", f"{sum(x.tempo_viagem for x in sol.rotas) / 60:.1f} h")
    k5.metric("Ocupação média", f"{ocupacao:.0f}%")

    sequencias = []
    for x in sol.rotas:
        dep = p.tipos[x.tipo].deposito
        pts = [dep] + x.clientes + ([dep] if p.retornar else [])
        sequencias.append(tuple(coords[i] for i in pts))
    with st.spinner("Traçando as rotas pelas ruas…"):
        linhas = tracados_em_cache(tuple(sequencias))

    coluna_mapa, coluna_lista = st.columns([3, 1.25])
    with coluna_mapa:
        st_folium(mapa.desenhar(p, sol, coords, enderecos, linhas), height=620,
                  width="stretch", returned_objects=[])
    with coluna_lista:
        links = exportar.links_google_maps(p, sol, coords)
        for k, x in enumerate(sol.rotas):
            t = p.tipos[x.tipo]
            cor = mapa.cor_da_rota(k)
            st.markdown(
                f"<div style='border-left:6px solid {cor};padding:4px 10px;margin-bottom:4px'>"
                f"<b>Rota {k + 1}</b> — {t.nome}<br>"
                f"<span style='font-size:.88em'>{len(x.clientes)} paradas · {x.carga}/{t.capacidade} "
                f"un · {x.distancia:.1f} km<br>{hora(x.saida_deposito)} → {hora(x.fim)} · "
                f"{brl(x.custo)}</span></div>", unsafe_allow_html=True)
            for parte, link in enumerate(links[k], start=1):
                rotulo = "Abrir no Google Maps" if len(links[k]) == 1 else f"Google Maps — trecho {parte}"
                st.link_button(rotulo, link, width="stretch")

    aba_paradas, aba_rotas, aba_metodos, aba_matriz = st.tabs(
        ["Ordem de visita", "Resumo por veículo", "Comparação dos métodos", "Matriz de distâncias"])

    with aba_paradas:
        st.dataframe(exportar.tabela_paradas(p, sol, enderecos), width="stretch",
                     hide_index=True)
    with aba_rotas:
        st.dataframe(exportar.tabela_rotas(p, sol), width="stretch", hide_index=True)
    with aba_metodos:
        comparacao = tabela_comparacao(solucoes)
        st.dataframe(comparacao, width="stretch", hide_index=True)
        if "GRASP" in solucoes and "VRPSolverEasy" in solucoes and solucoes["VRPSolverEasy"].rotas:
            g, e = solucoes["GRASP"], solucoes["VRPSolverEasy"]
            dif = 100.0 * (g.custo - e.custo) / e.custo if e.custo else 0.0
            if e.otima and abs(dif) < 1e-6:
                st.success("O GRASP já tinha encontrado o ótimo — o VRPSolverEasy só provou isso.")
            elif e.custo < g.custo:
                st.info(f"O VRPSolverEasy melhorou o GRASP em {dif:.2f}%.")
        escolha = st.radio("Mostrar no mapa", list(solucoes), horizontal=True,
                           index=list(solucoes).index(r["escolhida"]))
        if escolha != r["escolhida"] and solucoes[escolha].rotas:
            r["escolhida"] = escolha
            st.rerun()
    with aba_matriz:
        rotulos = [f"{k}. {e[:28]}" for k, e in enumerate(enderecos)]
        st.caption("Distância em km pelas ruas, da linha para a coluna. Ida e volta podem "
                   "diferir por causa das mãos de direção.")
        st.dataframe(pd.DataFrame(p.dist.round(2), index=rotulos, columns=rotulos),
                     width="stretch")

    planilha = exportar.planilha(p, sol, enderecos, coords, tabela_comparacao(solucoes))
    a, b, c, _ = st.columns([1.3, 1.3, 1.3, 2.7])
    a.download_button("⬇️ Baixar plano (Excel)", planilha, "plano_de_rotas.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                      type="primary", width="stretch")
    b.button("← Ajustar frota", on_click=ir, args=(2,), width="stretch")
    c.button("🏠 Novo roteiro", on_click=recomecar, width="stretch", key="fim_inicio")


def tabela_comparacao(solucoes) -> pd.DataFrame:
    linhas = []
    for nome, s in solucoes.items():
        linhas.append({
            "Método": s.metodo,
            "Custo (R$)": round(s.custo, 2) if s.rotas else None,
            "Distância (km)": round(s.distancia, 2) if s.rotas else None,
            "Veículos": len(s.rotas),
            "Limitante inferior (R$)": None if s.limitante_inferior is None else round(s.limitante_inferior, 2),
            "Gap (%)": None if s.gap is None else round(s.gap, 2),
            "Ótima provada": "sim" if s.otima else "não",
            "Tempo (s)": round(s.tempo_s, 2),
            "Situação": s.status,
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------

if st.session_state.etapa == 3 and "resultado" in st.session_state:
    etapa_rotas()
elif st.session_state.etapa >= 2 and "matriz" in st.session_state:
    etapa_frota()
else:
    etapa_pontos()
