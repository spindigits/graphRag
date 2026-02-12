import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from functools import partial

# Répertoire de stockage du knowledge graph
WORKING_DIR = "./storage"
os.makedirs(WORKING_DIR, exist_ok=True)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:14b",       # ton modèle 32K+
        llm_model_max_async=2,               # parallélisme (adapte selon RAM)
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768}    # CRITIQUE : forcer le context 32K
        },
        max_total_tokens=32768,              # contexte total pour le RAG
        chunk_token_size=1200,               # taille des chunks
        embedding_func=EmbeddingFunc(
            embedding_dim=768,               # nomic-embed-text = 768
            max_token_size=8192,
            func=partial(
                ollama_embed,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            )
        ),
        vector_db_storage_cls_kwargs={
            "embedding_dim": 768             # CRITIQUE : forcer la dimension pour le vector store
        }
    )

    # CRITIQUE : Initialiser les storages avant utilisation
    await rag.initialize_storages()

    return rag

if __name__ == "__main__":
    rag = asyncio.run(initialize_rag())
    print("GraphRAG initialisé avec succès")