"""
CaféIA - GraphRAG avec Ollama
Interface Streamlit — version corrigée
Fixes : event loop Streamlit, cache RAG instance, gestion async propre
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime

# ─── FIX CRITIQUE ─────────────────────────────────────────────────────────────
# nest_asyncio DOIT être appliqué avant tout import Streamlit/asyncio
# Streamlit tourne dans son propre event loop ; nest_asyncio permet d'imbriquer
import nest_asyncio
nest_asyncio.apply()
import asyncio
# ──────────────────────────────────────────────────────────────────────────────

from main import initialize_rag
from lightrag import QueryParam
from document_processor import DocumentProcessor

# ─── Configuration page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CaféIA - GraphRAG",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    color: #1f77b4 !important;
    margin-bottom: 0.2rem !important;
}
.sub-header {
    font-size: 1rem;
    text-align: center;
    color: #666;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers async ────────────────────────────────────────────────────────────

def run_async(coro):
    """
    Exécute une coroutine depuis du code synchrone Streamlit.
    Utilise l'event loop existant (rendu possible par nest_asyncio).
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ─── Initialisation RAG (singleton dans session_state) ───────────────────────

@st.cache_resource(show_spinner="⏳ Initialisation du knowledge graph...")
def get_rag_instance():
    """
    Initialise LightRAG UNE SEULE FOIS pour toute la session Streamlit.
    st.cache_resource persiste l'objet entre les reruns.
    """
    return run_async(initialize_rag())


# ─── Fonctions RAG ────────────────────────────────────────────────────────────

def insert_document(text: str, filename: str):
    """Insère un document dans le RAG (synchrone, utilise l'event loop)."""
    rag = get_rag_instance()
    run_async(rag.ainsert(text))


def query_rag(question: str, mode: str) -> str:
    """Interroge le RAG et retourne la réponse (str)."""
    rag = get_rag_instance()
    result = run_async(
        rag.aquery(question, param=QueryParam(mode=mode))
    )
    # Sécurité : aquery peut retourner None si le graph est vide
    if result is None:
        return "⚠️ Aucune réponse générée. Vérifiez que des documents sont indexés (storage/ non vide) et que le modèle Ollama répond."
    return result


# ─── Session state ────────────────────────────────────────────────────────────

if 'uploaded_files_count' not in st.session_state:
    st.session_state.uploaded_files_count = 0
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


# ─── Interface ────────────────────────────────────────────────────────────────

def main():

    # Header avec logos
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if os.path.exists("IMG/upvd_logo.png"):
            st.image("IMG/upvd_logo.png", width=150)
    with col2:
        st.markdown('<p class="main-header">☕ CaféIA - GraphRAG</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Interface de gestion documentaire et interrogation LLM avec Ollama</p>', unsafe_allow_html=True)
    with col3:
        if os.path.exists("IMG/mensaflow_logo.jpg"):
            st.image("IMG/mensaflow_logo.jpg", width=150)

    # ── Vérification Ollama ──────────────────────────────────────────────────
    if not st.session_state.get('ollama_ok', False):
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                st.session_state.ollama_ok = True
            else:
                st.error("Ollama ne répond pas — lancez `ollama serve`")
                st.stop()
        except Exception:
            st.error("❌ Impossible de joindre Ollama sur localhost:11434")
            st.info("Lancez : `ollama serve` dans un terminal séparé")
            st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Informations")

        # Vérifier si le storage existe déjà (documents déjà indexés)
        storage_exists = os.path.exists("./storage") and any(
            os.path.isfile(os.path.join("./storage", f))
            for f in os.listdir("./storage")
        ) if os.path.exists("./storage") else False

        st.info(f"""
        **Modèle LLM :** qwen2.5:14b
        **Embedding :** nomic-embed-text
        **Documents session :** {st.session_state.uploaded_files_count}
        **Storage persistant :** {'✅ Oui' if storage_exists else '❌ Vide'}
        """)

        if not storage_exists:
            st.warning("⚠️ Aucun document indexé. Commencez par l'onglet Upload.")

        st.header("⚙️ Configuration")
        st.caption("Paramètres dans `main.py` — context 32K, embedding 768 dims")

        with st.expander("📖 Modes de recherche"):
            st.markdown("""
            | Mode | Cas d'usage |
            |------|-------------|
            | **naive** | RAG classique, 1 document |
            | **local** | Entités proches (1 hop) |
            | **global** | Patterns transversaux |
            | **hybrid** | Multi-hop ✅ recommandé |
            """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "💬 Interroger le RAG", "📜 Historique"])

    # ── TAB 1 : Upload ───────────────────────────────────────────────────────
    with tab1:
        st.header("📤 Importer des documents")

        uploaded_files = st.file_uploader(
            "Glissez-déposez vos documents ou cliquez pour parcourir",
            type=['pdf', 'docx', 'xlsx', 'txt'],
            accept_multiple_files=True,
            help="Formats : PDF, DOCX, XLSX, TXT — Limite 200MB/fichier"
        )

        if uploaded_files:
            st.markdown(f"📁 **{len(uploaded_files)} fichier(s) sélectionné(s)**")
            for f in uploaded_files:
                st.caption(f"  • {f.name} ({f.size / 1024:.1f} KB)")

            if st.button("🚀 Indexer les documents", type="primary", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                errors = []

                for idx, uploaded_file in enumerate(uploaded_files):
                    status.text(f"⏳ Traitement : {uploaded_file.name} ...")
                    try:
                        ext = Path(uploaded_file.name).suffix
                        text = DocumentProcessor.process_uploaded_file(uploaded_file, ext)

                        if text and text.strip():
                            insert_document(text, uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name} indexé")
                            st.session_state.uploaded_files_count += 1
                            if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files_list]:
                                st.session_state.uploaded_files_list.append({
                                    'name': uploaded_file.name,
                                    'size': uploaded_file.size,
                                    'type': ext,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                        else:
                            st.warning(f"⚠️ {uploaded_file.name} — texte vide ou illisible")

                    except Exception as e:
                        err_msg = str(e)
                        errors.append((uploaded_file.name, err_msg))
                        st.error(f"❌ {uploaded_file.name} : {err_msg}")

                    progress.progress((idx + 1) / len(uploaded_files))

                status.text("✨ Indexation terminée !")
                if not errors:
                    st.balloons()
                else:
                    st.warning(f"{len(errors)} fichier(s) en erreur — voir détails ci-dessus")

        with st.expander("ℹ️ Formats supportés"):
            st.markdown("""
            - **PDF** — texte extractible (pas d'OCR sur images scannées)
            - **DOCX** — Word, paragraphes + tableaux
            - **XLSX** — Excel, toutes feuilles
            - **TXT** — UTF-8 / Latin-1
            """)

    # ── TAB 2 : Query ────────────────────────────────────────────────────────
    with tab2:
        st.header("💬 Poser une question au RAG")

        question = st.text_area(
            "Votre question :",
            height=100,
            placeholder="Ex: Quel technicien certifié est disponible en Occitanie ?",
        )

        col_mode, col_btn = st.columns([3, 1])
        with col_mode:
            query_mode = st.selectbox(
                "Mode de recherche",
                options=['hybrid', 'naive', 'local', 'global'],
                index=0,
            )
        with col_btn:
            st.write("")
            st.write("")
            search_btn = st.button("🔍 Rechercher", type="primary", use_container_width=True)

        if search_btn:
            if not question.strip():
                st.warning("⚠️ Saisissez une question.")
            else:
                with st.spinner(f"Recherche en mode **{query_mode}**..."):
                    try:
                        result = query_rag(question, query_mode)

                        st.subheader("📝 Réponse")
                        st.markdown(result)

                        # Sources
                        with st.expander("📚 Détail du retrieval"):
                            mode_info = {
                                "naive":  "🔍 **Naive** — similarité vectorielle sur chunks",
                                "local":  "🔗 **Local** — entités proches dans le graph (1 hop)",
                                "global": "🌐 **Global** — patterns transversaux du graph",
                                "hybrid": "🎯 **Hybrid** — local + global, optimal pour multi-hop",
                            }
                            st.info(mode_info.get(query_mode, query_mode))

                            if st.session_state.uploaded_files_list:
                                st.markdown("**Documents indexés cette session :**")
                                for fi in st.session_state.uploaded_files_list:
                                    st.caption(f"• {fi['name']} ({fi['size']/1024:.1f} KB) — {fi['timestamp']}")
                            else:
                                st.caption("Documents indexés depuis une session précédente (storage/ persistant)")

                        # Historique
                        st.session_state.query_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'question': question,
                            'mode': query_mode,
                            'answer': result
                        })

                    except Exception as e:
                        st.error(f"❌ Erreur : {str(e)}")
                        st.info("Vérifiez que Ollama tourne et que les modèles sont disponibles : `ollama list`")

    # ── TAB 3 : Historique ───────────────────────────────────────────────────
    with tab3:
        st.header("📜 Historique des requêtes")

        if st.session_state.query_history:
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.query_history = []
                st.rerun()

            for query in reversed(st.session_state.query_history):
                with st.expander(f"🕐 {query['timestamp']} — [{query['mode']}] {query['question'][:60]}..."):
                    st.markdown(f"**Question :** {query['question']}")
                    st.markdown("---")
                    st.markdown(query['answer'])
        else:
            st.info("Aucune requête pour l'instant.")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#888;">☕ CaféIA — Powered by LightRAG & Ollama</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
