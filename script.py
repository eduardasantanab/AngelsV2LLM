import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

nlp = spacy.load("pt_core_news_sm")

st.header("Angels V2 LLM")

if "historico" not in st.session_state:
    st.session_state.historico = []

llm = OllamaLLM(model="llama3", temperature=0)

# Recebe input do usuário

user_input = st.text_input("Faça a sua pergunta:")

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

    # Pré-processa chunks
    for doc in docs_divididos:
        doc.page_content = preprocessar_texto(
            doc.page_content
        )

    # Corpus textual
    corpus = [
        doc.page_content
        for doc in docs_divididos
    ]

    # TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    matriz_tfidf = tfidf.fit_transform(corpus)

    # Sentence transformer
    modelo_bert = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    embeddings_chunks = modelo_bert.encode(corpus)

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
        corpus,
        tfidf,
        matriz_tfidf,
        dimensao,
        modelo_bert,
        embeddings_np,
        index,
        docs_divididos
    )

rcorpus, rtfidf, rmatriz_tfidf, rdimensao, rmodelo_bert, rembeddings_np, rindex, rdocs_divididos = carregar_base()


# Injeção no prompt

if user_input:
    user_input = preprocessar_texto(user_input)

    vetor_pergunta = rtfidf.transform([user_input])

    similaridades = cosine_similarity(
        vetor_pergunta,
        rmatriz_tfidf
    ).flatten()

    # Busca similaridade (top-20 chunks)
    top_indices = similaridades.argsort()[-20:][::-1]

    subcorpus = [
        rcorpus[int(i)]
        for i in top_indices
    ]

    embeddings_subcorpus = rembeddings_np[top_indices]

    faiss.normalize_L2(embeddings_subcorpus)

    rdimensao = embeddings_subcorpus.shape[1]

    embedding_pergunta = rmodelo_bert.encode(
        [user_input]
    )

    embedding_pergunta = np.array(
        embedding_pergunta,
        dtype=np.float32
    )

    similaridade_semantica = np.dot(
        embeddings_subcorpus,
        embedding_pergunta.T
    ).flatten()

    indices_finais = similaridade_semantica.argsort()[-5:][::-1]

    faiss.normalize_L2(embedding_pergunta)

    chunks_relevantes = [
        subcorpus[int(i)]
        for i in indices_finais
    ]

    # Monta contexto (com base nos chunks encontrados) - Generation
    contexto = "\n".join(chunks_relevantes)

    historico_texto = ""

    for autor, mensagem in st.session_state.historico[-6:]:
        historico_texto += f"{autor}: {mensagem}\n"

    # Prompt
    prompt = f"""
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

    Histórico da conversa:
    {historico_texto}

    Contexto:
    {contexto}

    Pergunta atual:
    {user_input}
    """

    # resposta
    resposta = llm.invoke(prompt)

    st.session_state.historico.append(
        ("Usuário", user_input)
    )

    st.session_state.historico.append(
        ("Assistente", resposta)
    )

    st.write(resposta)


# Entrega final: Utilizar métricas de avaliação de desempenho.