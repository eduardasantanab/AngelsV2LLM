import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from bert_score import score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

nlp = spacy.load("pt_core_news_sm")

st.header("Angels V2 LLM")

if "historico" not in st.session_state:
    st.session_state.historico = []

for autor, mensagem in st.session_state.historico:

    if autor == "Usuário":
        st.chat_message("user").write(mensagem)

    else:
        st.chat_message("assistant").write(mensagem)



llm = OllamaLLM(model="llama3", temperature=0)

# Recebe input do usuário

user_input = st.chat_input("Faça sua pergunta")

# Pré-processamento para a base e para o input do usuário

def preprocessar_texto(texto):

    # lowercase
    texto = texto.lower()

    # remove quebras
    texto = texto.replace("\n", " ")

    # remove espaços duplicados
    texto = re.sub(r"\s+", " ", texto)

    # normalização unicode
    texto = unicodedata.normalize("NFKD", texto)

    # spaCy
    doc = nlp(texto)

    tokens_processados = []

    for token in doc:

        # remove stopwords e pontuação
        if token.is_stop or token.is_punct:
            continue

        # mantém apenas classes importantes
        if token.pos_ not in [
            "NOUN",
            "PROPN",
            "VERB",
            "ADJ"
        ]:
            continue

        # lematização
        lemma = token.lemma_.strip()

        if len(lemma) > 2:
            tokens_processados.append(lemma)

    return " ".join(tokens_processados)


def detectar_intencao(pergunta):

    pergunta = pergunta.lower()

    if "o que é" in pergunta:
        return "Definição"

    elif "como" in pergunta:
        return "Procedimento"

    elif "quando" in pergunta:
        return "Tempo"

    elif "por que" in pergunta:
        return "Justificativa"

    elif "qual" in pergunta:
        return "Informação"

    elif "sintoma" in pergunta:
        return "Sintomas"

    return "Geral"


# Pré-processamento para a base e para o input do usuário

@st.cache_resource
def carregar_base():

    # Carrega PDFs

    loader = PyPDFLoader(
        "Dataset/APS_Pre_Natal.pdf"
    )

    docs = loader.load()

    # Divide em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs_divididos = splitter.split_documents(docs)


    # Corpus textual
    corpus_original = [
        doc.page_content
        for doc in docs_divididos
    ]

    corpus_processado = [

        preprocessar_texto(doc.page_content)

        for doc in docs_divididos

    ]

    # TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    matriz_tfidf = tfidf.fit_transform(corpus_processado)

    # Sentence transformer
    modelo_bert = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    embeddings_chunks = modelo_bert.encode(corpus_processado)

    embeddings_np = np.array(
        embeddings_chunks,
        dtype=np.float32
    )

    faiss.normalize_L2(embeddings_np)

    # FAISS
    dimensao = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dimensao)
    index.add(embeddings_np)

    # Retorna tudo
    return (
        corpus_original,
        corpus_processado,
        tfidf,
        matriz_tfidf,
        modelo_bert,
        embeddings_np,
        index,
        docs_divididos
    )
(
    rcorpus_original,
    rcorpus_processado,
    rtfidf,
    rmatriz_tfidf,
    rmodelo_bert,
    rembeddings_np,
    rindex,
    rdocs_divididos
) = carregar_base()

mostrar_chunks = st.checkbox("Mostrar todos os chunks (Debug)")

if mostrar_chunks:

    for i, doc in enumerate(rdocs_divididos):

        st.markdown(f"### Chunk {i}")

        st.write(doc.page_content)

        st.divider()

def recuperar_chunks(pergunta):

    vetor_pergunta = rtfidf.transform([pergunta])

    similaridades = cosine_similarity(
        vetor_pergunta,
        rmatriz_tfidf
    ).flatten()

    # Top 50 lexical
    top_indices = similaridades.argsort()[-50:][::-1]

    subcorpus_original = [
        rcorpus_original[int(i)]
        for i in top_indices
    ]

    embeddings_subcorpus = rembeddings_np[top_indices]

    faiss.normalize_L2(embeddings_subcorpus)

    embedding_pergunta = rmodelo_bert.encode(
        [pergunta]
    )

    embedding_pergunta = np.array(
        embedding_pergunta,
        dtype=np.float32
    )

    faiss.normalize_L2(embedding_pergunta)

    similaridade_semantica = np.dot(
        embeddings_subcorpus,
        embedding_pergunta.T
    ).flatten()

    indices_finais = similaridade_semantica.argsort()[-10:][::-1]

    chunks_recuperados = [
        int(top_indices[i])
        for i in indices_finais
    ]

    print("Chunks recuperados:", chunks_recuperados)

    for indice, score in zip(chunks_recuperados,
                             similaridade_semantica[indices_finais]):
        print(indice, score)

    chunks_relevantes = [
        subcorpus_original[int(i)]
        for i in indices_finais
    ]

    return chunks_recuperados, chunks_relevantes


def montar_contexto(chunks_relevantes):

    return "\n".join(chunks_relevantes)

def gerar_prompt(
        pergunta,
        contexto,
        historico,
        intencao
):

    return f"""
Você é um assistente virtual especializado em protocolos clínicos gestacionais.

Sua tarefa é responder EXATAMENTE ao que foi perguntado.

Regras:

- Responda apenas com base no contexto fornecido.
- Não invente informações.
- Identifique a intenção da pergunta antes de responder.
- Se a pergunta pedir definição, explique o que é.
- Se pedir procedimento, explique como é feito.
- Se pedir importância, explique a finalidade.
- Não responda "sim" ou "não" sem explicação.
- Responda de forma objetiva, clara e coerente com a pergunta.

Histórico:

{historico}

Contexto:

{contexto}

Tipo da pergunta:

{intencao}

Pergunta:

{pergunta}
"""


# Injeção no prompt

def responder(pergunta, salvar_historico=True):

    pergunta = preprocessar_texto(pergunta)

    intencao = detectar_intencao(pergunta)

    chunks, chunks_relevantes = recuperar_chunks(
        pergunta
    )

    contexto = montar_contexto(
        chunks_relevantes
    )

    historico = ""

    for autor, mensagem in st.session_state.historico[-6:]:

        historico += f"{autor}: {mensagem}\n"

    prompt = gerar_prompt(
        pergunta,
        contexto,
        historico,
        intencao
    )

    resposta = llm.invoke(prompt)

    if salvar_historico:

        st.session_state.historico.append(
            ("Usuário", pergunta)
        )

        st.session_state.historico.append(
            ("Assistente", resposta)
        )

    return resposta, chunks


# Conversa normal do usuário
if user_input:

    resposta, chunks = responder(user_input)

    st.chat_message("assistant").write(resposta)

    with st.expander("Chunks recuperados"):
        for indice in chunks:
            st.write(
                "PDF:",
                rdocs_divididos[indice].metadata["source"]
            )

            st.write(
                rdocs_divididos[indice].page_content
            )

            st.divider()

    with st.expander("Texto dos chunks"):

        for i in chunks:
            st.markdown(f"### Chunk {i}")

            st.write(rdocs_divididos[i].page_content)

            st.divider()

# Entrega final: Métricas de avaliação de desempenho.

dataset_teste = [
    {
        "id": "Q01",
        "pergunta": "Em quais consultas deve ser realizada a estratificação do risco gestacional?",
        "intencao": "Procedimento",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Estratificação de Risco",
        "observacao": "Avalia se o modelo recupera corretamente o protocolo de estratificação de risco durante o acompanhamento pré-natal."
    },

    {
        "id": "Q02",
        "pergunta": "Quais testes rápidos devem ser realizados durante o pré-natal?",
        "intencao": "Informação",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Exames Laboratoriais",
        "observacao": "Resposta relacionada aos testes rápidos recomendados no protocolo assistencial."
    },

    {
        "id": "Q03",
        "pergunta": "Quando deve ser realizado o Teste Oral de Tolerância à Glicose (TOTG)?",
        "intencao": "Tempo",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Diabetes Gestacional",
        "observacao": "Resposta referente ao período gestacional indicado para realização do TOTG."
    },

    {
        "id": "Q04",
        "pergunta": "Quais são os critérios diagnósticos para Diabetes Mellitus Gestacional?",
        "intencao": "Informação",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Diabetes Gestacional",
        "observacao": "Resposta baseada na tabela com os valores diagnósticos de glicemia."
    },

    {
        "id": "Q05",
        "pergunta": "Qual deve ser a conduta quando o Coombs indireto for positivo?",
        "intencao": "Procedimento",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Exames Laboratoriais",
        "observacao": "Avalia a recuperação do protocolo de encaminhamento para pré-natal de alto risco."
    },

    {
        "id": "Q06",
        "pergunta": "Quais cuidados devem ser realizados na primeira semana após o parto?",
        "intencao": "Procedimento",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Puerpério",
        "observacao": "Resposta relacionada às orientações do puerpério imediato e da visita pós-parto."
    },

    {
        "id": "Q07",
        "pergunta": "Qual é a finalidade da estratificação do risco gestacional?",
        "intencao": "Justificativa",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Estratificação de Risco",
        "observacao": "Verifica se o modelo explica corretamente o objetivo da estratificação de risco durante o pré-natal."
    },

    {
        "id": "Q08",
        "pergunta": "Como devo planejar uma gestação?",
        "intencao": "Procedimento",
        "resposta_esperada": """
        ABORDAGEM PRÉ-CONCEPCIONAL Orientação nutricional visando a adequação, em tempo oportuno...
        """,
        "chunks_esperados": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Planejamento Reprodutivo",
        "observacao": "Avalia recuperação de informações sobre planejamento pré-concepcional."
    },

    {
        "id": "Q09",
        "pergunta": "Quais vacinas são recomendadas durante o pré-natal?",
        "intencao": "Informação",
        "resposta_esperada": """
        Calendário Nacional de Vacinação da Pessoa Adulta VACINA ESQUEMA BÁSICO REFORÇO IDADE RECOMENDADA...
        """,
        "chunks_esperados": [65, 66, 67, 68, 69, 385],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Vacinação",
        "observacao": "Avalia recuperação da tabela e recomendações de vacinação."
    },

    {
        "id": "Q10",
        "pergunta": "O que é a consulta pré-natal?",
        "intencao": "Definição",
        "resposta_esperada": """
        O exame pré-natal é um procedimento médico fundamental para avaliar a saúde da gestante e do feto durante a gestação....
        """,
        "chunks_esperados": [29, 30, 32, 33, 91],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Consultas Pré-Natal",
        "observacao": "Avalia a recuperação da definição e dos objetivos da consulta pré-natal."
    },

    {
        "id": "Q11",
        "pergunta": "Quando o pré-natal deve ser iniciado?",
        "intencao": "Tempo",
        "resposta_esperada": """
        O pré-natal deve ser iniciado preferencialmente até a 12ª semana...
        """,
        "chunks_esperados": [70, 71, 72, 73, 74, 75, 76, 77, 85],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Início do Pré-Natal",
        "observacao": "Avalia a recuperação da recomendação de início precoce do pré-natal."
    },

    {
        "id": "Q12",
        "pergunta": "Qual a frequência de consultas pré-natal durante a gestação?",
        "intencao": "Informação",
        "resposta_esperada": """
        O Ministério da Saúde recomenda um número mínimo de seis consultas de pré-natal....
        """,
        "chunks_esperados": [85, 86, 87],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Consultas Pré-Natal",
        "observacao": "Avalia recuperação da periodicidade mínima das consultas."
    },

    {
        "id": "Q13",
        "pergunta": "O que é planejamento reprodutivo?",
        "intencao": "Definição",
        "resposta_esperada": """
        Direitos sexuais e reprodutivos são direitos humanos, que devem ser assegurados sem distinção de situação social, raça, cor, etnia, nacionalidade, cultura, religião, gênero, orientação sexual ou outro....
        """,
        "chunks_esperados": [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 399],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Planejamento Reprodutivo",
        "observacao": "Avalia recuperação dos conceitos e direitos relacionados ao planejamento reprodutivo."
    },

    {
        "id": "Q14",
        "pergunta": "É preciso fazer pré-natal no puerpério?",
        "intencao": "Justificativa",
        "resposta_esperada": """
        Consulta de puerpério: O período de realização da 1ª consulta após o parto ...
        """,
        "chunks_esperados": [94, 95, 96],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Puerpério",
        "observacao": "Avalia recuperação das recomendações referentes ao acompanhamento puerperal."
    },

    {
        "id": "Q15",
        "pergunta": "Quais exames laboratoriais devem ser solicitados na primeira consulta do pré-natal?",
        "intencao": "Informação",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Exames Laboratoriais",
        "observacao": "Avalia a recuperação dos exames laboratoriais previstos para a primeira consulta de pré-natal."
    },

    {
        "id": "Q16",
        "pergunta": "Em quais situações a gestante deve ser encaminhada para o pré-natal de alto risco?",
        "intencao": "Procedimento",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Estratificação de Risco",
        "observacao": "Avalia a recuperação dos critérios de encaminhamento para o pré-natal de alto risco."
    },

    {
        "id": "Q17",
        "pergunta": "Quais orientações devem ser fornecidas durante o planejamento pré-concepcional?",
        "intencao": "Procedimento",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Planejamento Reprodutivo",
        "observacao": "Avalia a recuperação das orientações recomendadas durante o planejamento pré-concepcional."
    },

    {
        "id": "Q18",
        "pergunta": "Quais exames de imagem podem ser solicitados durante o pré-natal?",
        "intencao": "Informação",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Exames de Imagem",
        "observacao": "Avalia a recuperação das recomendações relacionadas aos exames de imagem durante a gestação."
    },

    {
        "id": "Q19",
        "pergunta": "Quais sinais e sintomas indicam necessidade de encaminhamento para atendimento especializado durante a gestação?",
        "intencao": "Informação",
        "resposta_esperada": "",
        "chunks_esperados": [],
        "fonte": "APS_Pre_Natal.pdf",
        "categoria": "Sinais de Alerta",
        "observacao": "Avalia se o sistema recupera corretamente os sinais de alerta que exigem encaminhamento imediato."
    }
]

def calcular_precision(recuperados, esperados):

    recuperados = set(recuperados)
    esperados = set(esperados)

    verdadeiros_positivos = len(
        recuperados.intersection(esperados)
    )

    if len(recuperados) == 0:
        return 0

    return verdadeiros_positivos / len(recuperados)


def calcular_recall(recuperados, esperados):

    recuperados = set(recuperados)
    esperados = set(esperados)

    verdadeiros_positivos = len(
        recuperados.intersection(esperados)
    )

    if len(esperados) == 0:
        return 0

    return verdadeiros_positivos / len(esperados)


def jaccard(resposta_modelo,
             resposta_esperada):

    modelo = set(
        resposta_modelo.lower().split()
    )

    esperado = set(
        resposta_esperada.lower().split()
    )

    intersecao = modelo.intersection(
        esperado
    )

    uniao = modelo.union(
        esperado
    )

    return len(intersecao) / len(uniao)

def calcular_bertscore(
        resposta_modelo,
        resposta_esperada
):

    P, R, F1 = score(
        [resposta_modelo],
        [resposta_esperada],
        lang="pt"
    )

    return float(F1.mean())

if st.button("Avaliar Modelo"):

    with st.spinner("Executando avaliação..."):

        lista_precision = []
        lista_recall = []
        lista_jaccard = []
        lista_bertscore = []
        lista_f1 = []

        for exemplo in dataset_teste:

            resposta, chunks = responder(
                exemplo["pergunta"],
                salvar_historico=False
            )

            precision = calcular_precision(
                chunks,
                exemplo["chunks_esperados"]
            )

            recall = calcular_recall(
                chunks,
                exemplo["chunks_esperados"]
            )

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            lista_f1.append(f1)

            f1_medio = np.mean(lista_f1)

            st.metric("F1-Score", f"{f1_medio:.3f}")

            jac = jaccard(
                resposta,
                exemplo["resposta_esperada"]
            )

            bert = calcular_bertscore(
                resposta,
                exemplo["resposta_esperada"]
            )

            st.write("Pergunta:", exemplo["pergunta"])

            st.write("Chunks recuperados:", chunks)

            st.write("Chunks esperados:", exemplo["chunks_esperados"])

            st.write("Precision:", precision)

            st.write("Recall:", recall)

            st.write("Jaccard:", jac)

            st.write("BERTScore:", bert)

            st.divider()

            lista_precision.append(precision)
            lista_recall.append(recall)
            lista_jaccard.append(jac)
            lista_bertscore.append(bert)


    precision_media = np.mean(lista_precision)
    recall_medio = np.mean(lista_recall)
    jaccard_medio = np.mean(lista_jaccard)
    bertscore_medio = np.mean(lista_bertscore)
    f1_medio = np.mean(lista_f1)

    st.subheader("Resultados da Avaliação")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Precision", f"{precision_media:.3f}")
        st.metric("Recall", f"{recall_medio:.3f}")
        st.metric("F1-Score", f"{f1_medio:.3f}")

    with col2:
        st.metric("Jaccard", f"{jaccard_medio:.3f}")
        st.metric("BERTScore", f"{bertscore_medio:.3f}")

    # Gráfico Precision x Recall

    fig, ax = plt.subplots()

    ax.plot(lista_precision, marker="o", label="Precision")
    ax.plot(lista_recall, marker="o", label="Recall")

    ax.legend()

    st.pyplot(fig)


    fig_precision, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        ["Precision", "Recall"],
        [precision_media, recall_medio]
    )

    ax.set_ylim(0,1)
    ax.set_title("Eficiência do Sistema")

    st.pyplot(fig_precision)


    def calcular_f1(precision, recall):

        if precision + recall == 0:
            return 0

        return (
                2 * precision * recall
        ) / (precision + recall)






    # Gráfico Jaccard

    fig_jaccard, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        ["Jaccard"],
        [jaccard_medio]
    )

    ax.set_ylim(0,1)
    ax.set_title("Overlap entre Respostas")

    st.pyplot(fig_jaccard)

    # Gráfico BERTScore

    fig_bert, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        ["BERTScore"],
        [bertscore_medio]
    )

    ax.set_ylim(0,1)
    ax.set_title("Qualidade Semântica")

    st.pyplot(fig_bert)

    st.success("Avaliação concluída com sucesso.")

