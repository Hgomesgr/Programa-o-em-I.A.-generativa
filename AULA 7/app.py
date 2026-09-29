import re
from collections import Counter

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Analisador de Feedbacks",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# DOWNLOAD DOS RECURSOS NLTK
# ============================================================

@st.cache_resource
def carregar_nltk():
    recursos = [
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ]

    for caminho, recurso in recursos:
        try:
            nltk.data.find(caminho)
        except LookupError:
            nltk.download(recurso, quiet=True)

    return True


carregar_nltk()


# ============================================================
# CONFIGURAÇÕES DE ANÁLISE
# ============================================================

PORTUGUESE_STOPWORDS = set(stopwords.words("portuguese"))

# Algumas palavras importantes para análise de reclamações.
# Elas não serão removidas como stopwords.
PALAVRAS_IMPORTANTES = {
    "não",
    "nunca",
    "sem",
    "problema",
    "problemas",
    "ruim",
    "péssimo",
    "péssima",
    "horrível",
    "demora",
    "demorado",
    "demorada",
    "erro",
    "erros",
    "falha",
    "falhas",
    "atraso",
    "atrasado",
    "atrasada",
}

PORTUGUESE_STOPWORDS -= PALAVRAS_IMPORTANTES


# ============================================================
# DICIONÁRIOS DE SENTIMENTO
# ============================================================

PALAVRAS_POSITIVAS = {
    "bom",
    "boa",
    "ótimo",
    "ótima",
    "excelente",
    "perfeito",
    "perfeita",
    "satisfeito",
    "satisfeita",
    "satisfação",
    "gostei",
    "gosto",
    "adoro",
    "adorei",
    "rápido",
    "rápida",
    "eficiente",
    "eficiência",
    "resolvido",
    "resolvida",
    "recomendo",
    "recomendar",
    "parabéns",
    "qualidade",
    "facilidade",
    "fácil",
    "melhor",
    "melhorou",
    "sucesso",
    "excelência",
}

PALAVRAS_NEGATIVAS = {
    "ruim",
    "péssimo",
    "péssima",
    "horrível",
    "terrível",
    "problema",
    "problemas",
    "erro",
    "erros",
    "falha",
    "falhas",
    "atraso",
    "atrasado",
    "atrasada",
    "demora",
    "demorado",
    "demorada",
    "lento",
    "lenta",
    "lentidão",
    "insatisfeito",
    "insatisfeita",
    "reclamação",
    "reclamações",
    "defeito",
    "defeitos",
    "cancelamento",
    "cancelar",
    "cobrança",
    "cobrado",
    "cobrada",
    "caro",
    "cara",
    "difícil",
    "dificuldade",
    "frustração",
    "frustrado",
    "frustrada",
    "decepcionado",
    "decepcionada",
    "péssimo",
    "péssima",
    "não",
}


# ============================================================
# FUNÇÕES
# ============================================================

def normalizar_texto(texto):
    """
    Remove acentos, pontuação e transforma o texto em minúsculas.
    """
    texto = texto.lower()

    # Remove caracteres especiais, mantendo letras e espaços.
    texto = re.sub(r"[^a-záàâãéêíóôõúç\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def tokenizar(texto):
    """
    Transforma o texto em palavras.
    """
    texto_normalizado = normalizar_texto(texto)

    palavras = texto_normalizado.split()

    palavras_filtradas = [
        palavra
        for palavra in palavras
        if palavra not in PORTUGUESE_STOPWORDS
        and len(palavra) > 2
    ]

    return palavras_filtradas


def analisar_palavras(textos):
    """
    Analisa as palavras mais frequentes de um conjunto de textos.
    """
    todas_palavras = []

    for texto in textos:
        todas_palavras.extend(tokenizar(texto))

    contador = Counter(todas_palavras)

    return contador


def analisar_sentimento(texto):
    """
    Classificação simples de sentimento.

    O VADER do NLTK foi desenvolvido principalmente para inglês.
    Por isso, combinamos sua pontuação com um dicionário
    específico para português.
    """

    palavras = tokenizar(texto)

    positivas = [
        palavra for palavra in palavras
        if palavra in PALAVRAS_POSITIVAS
    ]

    negativas = [
        palavra for palavra in palavras
        if palavra in PALAVRAS_NEGATIVAS
    ]

    pontos_positivos = len(positivas)
    pontos_negativos = len(negativas)

    # Regras simples para português.
    if pontos_negativos > pontos_positivos:
        sentimento = "Negativo"

    elif pontos_positivos > pontos_negativos:
        sentimento = "Positivo"

    else:
        sentimento = "Neutro"

    return {
        "sentimento": sentimento,
        "positivas": positivas,
        "negativas": negativas,
        "pontos_positivos": pontos_positivos,
        "pontos_negativos": pontos_negativos,
    }


def gerar_feedback(resultado):
    """
    Gera um feedback empresarial para o usuário.
    """

    sentimento = resultado["sentimento"]

    if sentimento == "Negativo":
        return (
            "A mensagem apresenta indicadores de insatisfação. "
            "Recomenda-se avaliar os pontos mencionados pelo cliente "
            "para identificar oportunidades de melhoria no produto ou serviço."
        )

    if sentimento == "Positivo":
        return (
            "A mensagem apresenta indicadores de satisfação. "
            "Os aspectos positivos identificados podem ser utilizados "
            "como referência para manutenção da qualidade do produto ou serviço."
        )

    return (
        "A mensagem não apresenta indicadores suficientes para uma "
        "classificação claramente positiva ou negativa. "
        "Recomenda-se analisar o contexto da mensagem."
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .titulo {
        font-size: 32px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitulo {
        font-size: 16px;
        color: #667085;
        margin-bottom: 30px;
    }

    .card {
        background-color: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #e4e7ec;
        margin-bottom: 20px;
    }

    .positivo {
        background-color: #ecfdf3;
        border-left: 5px solid #12b76a;
        padding: 18px;
        border-radius: 8px;
        color: #027a48;
    }

    .negativo {
        background-color: #fef3f2;
        border-left: 5px solid #f04438;
        padding: 18px;
        border-radius: 8px;
        color: #b42318;
    }

    .neutro {
        background-color: #f2f4f7;
        border-left: 5px solid #667085;
        padding: 18px;
        border-radius: 8px;
        color: #344054;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CABEÇALHO
# ============================================================

st.markdown(
    '<div class="titulo">📊 Analisador de Feedbacks</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitulo">
        Análise automatizada de mensagens de clientes,
        identificação de sentimento e palavras recorrentes.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MENU
# ============================================================

aba_individual, aba_lote = st.tabs(
    [
        "💬 Análise de mensagem",
        "📈 Análise de reclamações",
    ]
)


# ============================================================
# ABA 1 - ANÁLISE INDIVIDUAL
# ============================================================

with aba_individual:

    st.markdown("### Analisar uma mensagem")

    st.write(
        "Digite abaixo uma mensagem recebida de um cliente "
        "para identificar o sentimento e os principais termos."
    )

    mensagem = st.text_area(
        "Mensagem do cliente",
        placeholder=(
            "Exemplo: Estou muito insatisfeito com o produto. "
            "A entrega demorou e o atendimento não resolveu meu problema."
        ),
        height=180,
    )

    analisar = st.button(
        "🔎 Analisar mensagem",
        type="primary",
        use_container_width=True,
    )

    if analisar:

        if not mensagem.strip():
            st.warning("Digite uma mensagem para realizar a análise.")

        else:

            resultado = analisar_sentimento(mensagem)

            sentimento = resultado["sentimento"]

            st.divider()

            st.markdown("### Resultado da análise")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Classificação",
                    sentimento,
                )

            with col2:
                st.metric(
                    "Indicadores positivos",
                    resultado["pontos_positivos"],
                )

            with col3:
                st.metric(
                    "Indicadores negativos",
                    resultado["pontos_negativos"],
                )

            st.markdown("### Feedback")

            if sentimento == "Positivo":
                classe = "positivo"
                emoji = "🟢"

            elif sentimento == "Negativo":
                classe = "negativo"
                emoji = "🔴"

            else:
                classe = "neutro"
                emoji = "⚪"

            st.markdown(
                f"""
                <div class="{classe}">
                    <strong>{emoji} Sentimento: {sentimento}</strong>
                    <br><br>
                    {gerar_feedback(resultado)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### Termos positivos")

                if resultado["positivas"]:
                    for palavra in set(resultado["positivas"]):
                        st.success(palavra)
                else:
                    st.info("Nenhum termo positivo identificado.")

            with col2:

                st.markdown("### Termos negativos")

                if resultado["negativas"]:
                    for palavra in set(resultado["negativas"]):
                        st.error(palavra)
                else:
                    st.info("Nenhum termo negativo identificado.")


# ============================================================
# ABA 2 - ANÁLISE EM LOTE
# ============================================================

with aba_lote:

    st.markdown("### Analisar múltiplas reclamações")

    st.write(
        "Insira uma reclamação por linha para identificar "
        "os termos mais recorrentes e o sentimento das mensagens."
    )

    mensagens_lote = st.text_area(
        "Reclamações",
        placeholder="""O produto apresentou problema.
A entrega demorou muito.
Gostei muito do atendimento.
O aplicativo está lento.
O produto chegou em perfeito estado.""",
        height=250,
    )

    analisar_lote = st.button(
        "📊 Analisar reclamações",
        type="primary",
        use_container_width=True,
    )

    if analisar_lote:

        if not mensagens_lote.strip():

            st.warning(
                "Digite pelo menos uma mensagem para realizar a análise."
            )

        else:

            lista_mensagens = [
                mensagem.strip()
                for mensagem in mensagens_lote.split("\n")
                if mensagem.strip()
            ]

            resultados = [
                analisar_sentimento(mensagem)
                for mensagem in lista_mensagens
            ]

            total_positivas = sum(
                1
                for resultado in resultados
                if resultado["sentimento"] == "Positivo"
            )

            total_negativas = sum(
                1
                for resultado in resultados
                if resultado["sentimento"] == "Negativo"
            )

            total_neutras = sum(
                1
                for resultado in resultados
                if resultado["sentimento"] == "Neutro"
            )

            st.divider()

            # ==================================================
            # MÉTRICAS
            # ==================================================

            st.markdown("### Visão geral")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total de mensagens",
                    len(lista_mensagens),
                )

            with col2:
                st.metric(
                    "Positivas",
                    total_positivas,
                )

            with col3:
                st.metric(
                    "Negativas",
                    total_negativas,
                )

            with col4:
                st.metric(
                    "Neutras",
                    total_neutras,
                )

            # ==================================================
            # PALAVRAS MAIS FREQUENTES
            # ==================================================

            st.markdown("### Palavras mais recorrentes")

            contador = analisar_palavras(lista_mensagens)

            palavras_frequentes = contador.most_common(20)

            if palavras_frequentes:

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("#### Ranking de palavras")

                    for posicao, (palavra, quantidade) in enumerate(
                        palavras_frequentes,
                        start=1,
                    ):

                        st.write(
                            f"**{posicao}. {palavra}** — "
                            f"{quantidade} ocorrência(s)"
                        )

                with col2:

                    st.markdown("#### Distribuição")

                    dados_grafico = {
                        palavra: quantidade
                        for palavra, quantidade in palavras_frequentes
                    }

                    st.bar_chart(dados_grafico)

            else:

                st.info(
                    "Não foram encontradas palavras suficientes para análise."
                )

            # ==================================================
            # RESULTADOS INDIVIDUAIS
            # ==================================================

            st.markdown("### Resultado por mensagem")

            for indice, (mensagem, resultado) in enumerate(
                zip(lista_mensagens, resultados),
                start=1,
            ):

                sentimento = resultado["sentimento"]

                if sentimento == "Positivo":
                    emoji = "🟢"
                elif sentimento == "Negativo":
                    emoji = "🔴"
                else:
                    emoji = "⚪"

                with st.expander(
                    f"{emoji} Mensagem {indice} — {sentimento}"
                ):

                    st.write(mensagem)

                    st.write(
                        f"**Feedback:** {gerar_feedback(resultado)}"
                    )

                    if resultado["positivas"]:
                        st.write(
                            "**Termos positivos:** "
                            + ", ".join(
                                sorted(
                                    set(resultado["positivas"])
                                )
                            )
                        )

                    if resultado["negativas"]:
                        st.write(
                            "**Termos negativos:** "
                            + ", ".join(
                                sorted(
                                    set(resultado["negativas"])
                                )
                            )
                        )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Sistema de análise de feedbacks | "
    "Python + Streamlit + NLTK"
)
