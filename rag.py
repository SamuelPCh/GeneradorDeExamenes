from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Directorio donde se guardará el vector store
CHROMA_DIR = "./chroma_db"

# Modelo de embeddings local
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def crear_vector_store(texto, tema):
    """Divide el texto en chunks y los guarda en Chroma."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents([texto])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=tema.replace(" ", "_").encode("ascii", "ignore").decode("ascii")
    )
    return vector_store

def buscar_contexto(tema, query, k=3):
    """Busca los chunks más relevantes para el tema."""
    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=tema.replace(" ", "_")
        )
        resultados = vector_store.similarity_search(query, k=k)
        contexto = "\n\n".join([r.page_content for r in resultados])
        return contexto
    except Exception:
        return ""