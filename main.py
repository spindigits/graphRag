"""
Configuration LightRAG + Ollama
Initialisation async correcte pour Streamlit
"""

import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from functools import partial

WORKING_DIR = "./storage"
os.makedirs(WORKING_DIR, exist_ok=True)


async def initialize_rag() -> LightRAG:
    """
    Crée et initialise une instance LightRAG avec Ollama.
    IMPORTANT : appeler une seule fois et mettre en cache dans session_state.
    """
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:14b",
        llm_model_max_async=2,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768}   # CRITIQUE : 32K minimum pour LightRAG
        },
        max_total_tokens=32768,
        chunk_token_size=1200,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,              # nomic-embed-text = 768 dims
            max_token_size=8192,
            func=partial(
                ollama_embed,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            )
        ),
        vector_db_storage_cls_kwargs={
            "embedding_dim": 768
        }
    )

    # CRITIQUE : initialiser les storages (requis depuis LightRAG v1.x)
    await rag.initialize_storages()

    # Initialiser le pipeline status (requis dans les versions récentes)
    try:
        from lightrag.operate import initialize_pipeline_status
        await initialize_pipeline_status(rag)
    except (ImportError, AttributeError):
        pass  # Versions antérieures ne nécessitent pas cette étape

    return rag


if __name__ == "__main__":
    rag = asyncio.run(initialize_rag())
    print("✅ GraphRAG initialisé avec succès")
