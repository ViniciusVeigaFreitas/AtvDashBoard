from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats


st.set_page_config(page_title="Dashboard — Atividade 07",
                   page_icon="📊", layout="wide")

PALETA = {
    "primaria":   "#1B4F72",   
    "secundaria": "#E67E22",  
    "destaque":   "#27AE60",   
    "alerta":     "#C0392B",   
    "neutra":     "#7F8C8D",   
}
PALETA_SEQ = ["#1B4F72", "#2874A6", "#5DADE2", "#85C1E9", "#AED6F1"]
PALETA_CAT = ["#1B4F72", "#E67E22", "#27AE60", "#C0392B", "#8E44AD"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.edgecolor": "#2C3E50",
    "font.size": 10,
})

DATA_DIR = Path(__file__).parent

ALUNO = {
    "nome": "Vinicius Veiga Freitas",
    "numero_chamada": "40",
    "turma": "3B2",
}


@st.cache_data(show_spinner=False)
def carregar(nome_arquivo: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / nome_arquivo, **kwargs)

def figura(tamanho=(8, 5)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            fig, ax = plt.subplots(figsize=tamanho)
            try:
                fn(fig, ax, *args, **kwargs)
            finally:
                plt.tight_layout()
            return fig
        return wrapper
    return decorator

class Tipo(Enum):
    DISPERSAO = "1. Dispersão — Altura × Peso"
    DISTRIBUICAO = "2. Distribuição — Idades dos Clientes"
    FREQUENCIA = "3. Frequência — Tipos de Atendimento"
    HEATMAP = "4. Heatmap — Movimento por Dia e Horário"
    LINHAS = "5. Linhas — Evolução das Vendas"
    PIZZA = "6. Pizza — Faturamento por Canal"
    REGRESSAO = "7. Regressão — Gasto × Renda"
    BARRAS = "8. Barras Agrupadas — Vendas por Categoria e Região"
    BOXPLOT = "9. Boxplot — Salários por Departamento"

class Galeria:

    # ---------- 1 ----------
    @staticmethod
    @figura()
    def dispersao(fig, ax):
        df = carregar("dados_dispersao.csv")
        for sexo, cor, marcador in (
            ("Feminino",  PALETA["secundaria"], "o"),
            ("Masculino", PALETA["primaria"],   "s"),
        ):
            sub = df[df["sexo"] == sexo]
            ax.scatter(sub["altura_cm"], sub["peso_kg"],
                       color=cor, alpha=0.8, label=sexo,
                       edgecolor="#2C3E50", s=65, marker=marcador)
        ax.set(title="Dispersão: Altura × Peso por Sexo",
               xlabel="Altura (cm)", ylabel="Peso (kg)")
        ax.legend(frameon=True, facecolor="white")

    # ---------- 2 ----------
    @staticmethod
    @figura()
    def distribuicao(fig, ax):
        df = carregar("dados_distribuicao.csv")
        sns.histplot(data=df, x="idade", bins=12, kde=True,
                     color=PALETA["primaria"], edgecolor="#2C3E50",
                     alpha=0.75, ax=ax)
        for valor, cor, estilo, rotulo in (
            (df["idade"].mean(),   PALETA["alerta"],    "--", "Média"),
            (df["idade"].median(), PALETA["destaque"],  ":",  "Mediana"),
        ):
            ax.axvline(valor, color=cor, linestyle=estilo, linewidth=2,
                       label=f"{rotulo} = {valor:.1f}")
        ax.set(title="Distribuição das Idades dos Clientes",
               xlabel="Idade (anos)", ylabel="Frequência")
        ax.legend()

    # ---------- 3 ----------
    @staticmethod
    @figura()
    def frequencia(fig, ax):
        df = carregar("dados_frequencia.csv")
        freq = df["tipo_atendimento"].value_counts()
        pct = freq.div(freq.sum()).mul(100).round(1)
        sns.barplot(x=freq.values, y=freq.index, hue=freq.index,
                    palette=PALETA_CAT[:len(freq)],
                    legend=False, edgecolor="#2C3E50", ax=ax)
        for i, (v, p) in enumerate(zip(freq.values, pct.values)):
            ax.text(v + 1, i, f"{v} ({p}%)", va="center", fontsize=10)
        ax.set(title="Frequência dos Tipos de Atendimento",
               xlabel="Número de chamados", ylabel="")
        ax.set_xlim(0, freq.max() * 1.18)

    # ---------- 4 ----------
    @staticmethod
    @figura()
    def heatmap(fig, ax):
        df = carregar("dados_heatmap.csv")
        pivot = df.pivot(index="dia_semana", columns="horario",
                         values="movimento_clientes")
        ordem = ["Segunda", "Terça", "Quarta", "Quinta",
                 "Sexta", "Sábado", "Domingo"]
        pivot = pivot.reindex(ordem)
        sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Nº de clientes"}, ax=ax)
        ax.set(title="Movimento por Dia e Horário",
               xlabel="Horário", ylabel="")

    # ---------- 5 ----------
    @staticmethod
    @figura()
    def linhas(fig, ax):
        df = carregar("dados_linhas.csv",
                      parse_dates=["data"]).sort_values("data")
        sns.lineplot(data=df, x="data", y="vendas", hue="loja",
                     palette={"Loja A": PALETA["primaria"],
                              "Loja B": PALETA["secundaria"]},
                     marker="o", markersize=4, linewidth=1.5, ax=ax)
        df["mm7"] = df.groupby("loja")["vendas"].transform(
            lambda s: s.rolling(7).mean())
        for loja, grupo in df.groupby("loja"):
            ax.plot(grupo["data"], grupo["mm7"], linestyle="--",
                    linewidth=2, alpha=0.9,
                    color=PALETA["destaque"] if loja == "Loja A"
                    else PALETA["alerta"],
                    label=f"{loja} (MM7)")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.tick_params(axis="x", rotation=45)
        ax.set(title="Evolução das Vendas Diárias",
               xlabel="Data", ylabel="Vendas (R$)")
        ax.legend(fontsize=8)

    # ---------- 6 ----------
    @staticmethod
    @figura()
    def pizza(fig, ax):
        df = carregar("dados_pizza.csv").sort_values(
            "faturamento", ascending=False).reset_index(drop=True)
        cores = PALETA_CAT[:len(df)]
        explode = [0.06] + [0.0] * (len(df) - 1)
        wedges, _, autotexts = ax.pie(
            df["faturamento"], labels=df["canal_venda"], colors=cores,
            autopct=lambda p: f"{p:.1f}%", startangle=90,
            counterclock=False, explode=explode,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 10})
        for t in autotexts:
            t.set_color("white")
            t.set_fontweight("bold")
        total = df["faturamento"].sum()
        ax.set_title(f"Faturamento por Canal\nTotal: R$ {total:,.0f}",
                     fontweight="bold")
        legendas = [f"{c}: R$ {v:,.0f}" for c, v in
                    zip(df["canal_venda"], df["faturamento"])]
        ax.legend(wedges, legendas, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), fontsize=8)

    # ---------- 7 ----------
    @staticmethod
    @figura()
    def regressao(fig, ax):
        df = carregar("dados_regressao.csv")
        res = stats.linregress(df["renda_mensal"], df["gasto_mensal"])
        a, b, r2 = res.intercept, res.slope, res.rvalue ** 2
        sns.scatterplot(data=df, x="renda_mensal", y="gasto_mensal",
                        color=PALETA["primaria"], alpha=0.75, s=55,
                        edgecolor="#2C3E50", ax=ax, label="Clientes")
        x = np.linspace(df["renda_mensal"].min(),
                        df["renda_mensal"].max(), 100)
        ax.plot(x, a + b * x, color=PALETA["alerta"], linewidth=2.5,
                label=f"y = {a:.0f} + {b:.3f}·x")
        ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor=PALETA["neutra"], alpha=0.9))
        ax.set(title="Regressão: Gasto × Renda",
               xlabel="Renda mensal (R$)", ylabel="Gasto mensal (R$)")
        ax.legend(loc="lower right", fontsize=8)

    # ---------- 8 ----------
    @staticmethod
    @figura()
    def barras(fig, ax):
        df = carregar("dados_barras.csv")
        ordem = (df.groupby("categoria")["vendas_medias"].mean()
                   .sort_values(ascending=False).index)
        sns.barplot(data=df, x="categoria", y="vendas_medias",
                    hue="regiao", order=ordem,
                    palette={"Norte": PALETA["primaria"],
                             "Sul":   PALETA["secundaria"],
                             "Leste": PALETA["destaque"],
                             "Oeste": PALETA["alerta"]},
                    edgecolor="#2C3E50", linewidth=0.8, ax=ax)
        ax.set(title="Vendas Médias por Categoria e Região",
               xlabel="Categoria", ylabel="Vendas médias (R$)")
        ax.legend(title="Região", loc="upper left",
                  bbox_to_anchor=(1.0, 1.0), fontsize=8)
        ax.tick_params(axis="x", rotation=15)

    # ---------- 9 ----------
    @staticmethod
    @figura()
    def boxplot(fig, ax):
        df = carregar("dados_boxplot.csv")
        ordem = (df.groupby("departamento")["salario"].median()
                   .sort_values(ascending=False).index)
        sns.boxplot(data=df, x="departamento", y="salario", order=ordem,
                    hue="departamento", legend=False,
                    palette=PALETA_CAT[:len(ordem)],
                    width=0.6, linewidth=1.5,
                    flierprops=dict(marker="D",
                                    markerfacecolor=PALETA["alerta"],
                                    markeredgecolor="#2C3E50",
                                    markersize=6, alpha=0.85), ax=ax)
        sns.stripplot(data=df, x="departamento", y="salario", order=ordem,
                      color="#2C3E50", alpha=0.35, size=3,
                      jitter=True, ax=ax)
        ax.set(title="Distribuição Salarial por Departamento",
               xlabel="Departamento", ylabel="Salário mensal (R$)")

ANALISES: dict[Tipo, str] = {
    Tipo.DISPERSAO: (
        "A nuvem de pontos mostra correlação positiva moderada entre altura e "
        "peso, mas com dispersão considerável — altura explica só parte da "
        "variação. Não há separação nítida por sexo: os dois grupos se "
        "misturam na faixa de 160–180 cm. Os pontos isolados (extremos de "
        "altura) puxam a reta imaginária e merecem checagem de medição."
    ),
    Tipo.DISTRIBUICAO: (
        "A distribuição é assimétrica à direita: a maioria dos clientes está "
        "entre 20 e 30 anos, com cauda longa até ~55. Média e mediana ficam "
        "próximas, mas a mediana representa melhor o cliente típico. O formato "
        "sugere público jovem — útil para campanhas e definição de canais."
    ),
    Tipo.FREQUENCIA: (
        "Dúvidas e reclamações dominam a base de chamados, indicando que a "
        "jornada do cliente ainda gera atrito. Cancelamentos, embora menos "
        "numerosos, têm peso financeiro desproporcional. Elogios são minoria — "
        "bom sinal de que há espaço para melhorar a experiência antes de "
        "investir em marketing."
    ),
    Tipo.HEATMAP: (
        "Os picos se concentram no fim de semana, especialmente sábado à noite "
        "e domingo ao meio-dia. Dias úteis têm movimento mais uniforme, com "
        "quedas no fim da tarde. Isso sugere escalar equipe e estoque para "
        "sábado/domingo e usar a semana para tarefas internas."
    ),
    Tipo.LINHAS: (
        "As duas lojas crescem ao longo do período, com a Loja A ligeiramente "
        "à frente. A média móvel de 7 dias suaviza oscilações e confirma "
        "tendência de alta. Quedas pontuais isoladas merecem investigação "
        "(feriados, ruptura de estoque, falha de sistema)."
    ),
    Tipo.PIZZA: (
        "O canal online (Site + Aplicativo) responde pela maior parte do "
        "faturamento, confirmando a migração digital. Loja Física ainda tem "
        "peso relevante, mas telefone e parceiros são cauda. Estratégia "
        "coerente: reforçar o digital sem abandonar a loja como ponto de "
        "experiência."
    ),
    Tipo.REGRESSAO: (
        "Relação linear forte entre renda e gasto: o R² alto indica que a "
        "renda explica a maior parte da variação do gasto. A inclinação "
        "mostra quanto do aumento de renda se converte em consumo. Lembrete: "
        "correlação não é causalidade — variáveis omitidas (educação, "
        "localização) podem influenciar ambos."
    ),
    Tipo.BARRAS: (
        "Há grande disparidade entre regiões dentro de cada categoria. "
        "Algumas combinações categoria × região vendem 3–4× mais que outras, "
        "o que indica oportunidade de redistribuir esforço comercial e "
        "investigar por que certas regiões ficam abaixo do potencial."
    ),
    Tipo.BOXPLOT: (
        "Os departamentos se dividem em dois blocos salariais claros, com "
        "medianas bem distintas. A dispersão interna também varia: alguns "
        "departamentos têm caudas longas (salários altos pontuais). Os "
        "outliers destacados no topo pedem verificação — podem ser bônus, "
        "cargos de liderança ou erros de lançamento."
    ),
}

def render_sidebar() -> None:
    box = st.sidebar.container()
    box.title("Informações do Aluno")
    box.divider()
    for rotulo, chave in (("Nome", "nome"),
                          ("Número da chamada", "numero_chamada"),
                          ("Turma", "turma")):
        box.markdown(f"**{rotulo}:** {ALUNO[chave]}")
    box.divider()
    box.caption("Atividade 07 — Visualização de Dados com Python.")

def chunked(seq, tamanho):
    it = iter(seq)
    while (bloco := list(itertools.islice(it, tamanho))):
        yield bloco

def render() -> None:
    render_sidebar()
    st.title("Dashboard — Atividade 07")
    st.markdown("Visualização dos 9 gráficos com análises resumidas.")
    st.divider()

    for bloco in chunked(list(Tipo), 3):
        colunas = st.columns(3)
        for coluna, tipo in zip(colunas, bloco):
            with coluna:
                st.subheader(tipo.value)
                st.pyplot(getattr(Galeria, tipo.name.lower())())
        for tipo in bloco:
            with st.expander(f"Análise — {tipo.value}", expanded=True):
                st.write(ANALISES[tipo])
        st.divider()

    st.caption("Dashboard gerado com Streamlit")


if __name__ == "__main__":
    render()