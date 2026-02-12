"""
Application Streamlit pour GraphRAG avec Ollama
Interface pour uploader des documents et interroger le RAG
"""

import streamlit as st
import asyncio
import os
from pathlib import Path
from datetime import datetime

from main import initialize_rag
from lightrag import QueryParam
from document_processor import DocumentProcessor

# Configuration de la page
st.set_page_config(
    page_title="CaféIA - graphRAG avec Ollama",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 5rem !important;
        font-weight: bold !important;
        text-align: center !important;
        color: #1f77b4 !important;
        margin-bottom: 0.5rem !important;
        margin-top: 1rem !important;
        line-height: 1.2 !important;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .query-section {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stAlert {
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = None
if 'uploaded_files_count' not in st.session_state:
    st.session_state.uploaded_files_count = 0
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


async def init_rag():
    """Initialise l'instance RAG de manière asynchrone."""
    # Ne pas mettre en cache pour éviter les problèmes d'event loop
    rag = await initialize_rag()
    return rag


async def insert_document_to_rag(text: str, filename: str):
    """Insère un document dans le RAG."""
    rag = await initialize_rag()
    await rag.ainsert(text)


async def query_rag(question: str, mode: str):
    """Interroge le RAG avec la question et le mode spécifiés."""
    rag = await initialize_rag()
    result = await rag.aquery(
        question,
        param=QueryParam(mode=mode)
    )
    return result


def main():
    # En-tête avec logos
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.image("IMG/upvd_logo.png", width=150)

    with col2:
        st.markdown('<p class="main-header">☕ CaféIA - GraphRAG</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Interface de gestion documentaire et interrogation LLM avec Ollama</p>', unsafe_allow_html=True)

    with col3:
        st.image("IMG/mensaflow_logo.jpg", width=150)

    # Vérifier qu'Ollama est disponible (test au démarrage)
    if not st.session_state.get('ollama_checked', False):
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                st.session_state.ollama_checked = True
            else:
                st.error("Ollama ne répond pas correctement")
                st.info("Assurez-vous qu'Ollama est lancé avec: ollama serve")
                st.stop()
        except Exception as e:
            st.error("Impossible de se connecter à Ollama")
            st.info("Assurez-vous qu'Ollama est lancé et que les modèles sont disponibles (qwen2.5:14b et nomic-embed-text)")
            st.stop()

    # Barre latérale - Informations et configuration
    with st.sidebar:
        st.header("📊 Informations")
        st.info(f"""
        **Modèle LLM:** qwen2.5:14b
        **Modèle Embedding:** nomic-embed-text
        **Documents indexés:** {st.session_state.uploaded_files_count}
        """)

        st.header("⚙️ Configuration")
        st.caption("Les paramètres sont définis dans main.py")

        st.header("📚 Guide d'utilisation")
        with st.expander("Comment utiliser cette application ?"):
            st.markdown("""
            1. **Uploader des documents** dans la section appropriée
            2. Les documents seront automatiquement indexés dans le RAG
            3. **Poser vos questions** dans la section de requête
            4. Choisir le **mode de recherche** adapté à votre besoin
            """)

        with st.expander("Modes de recherche"):
            st.markdown("""
            - **Naive:** RAG classique, recherche simple
            - **Local:** Entités et relations proches (1 hop)
            - **Global:** Patterns globaux du knowledge graph
            - **Hybrid:** Combinaison local + global (recommandé)
            """)

    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "💬 Interroger le RAG", "📜 Historique"])

    # --- TAB 1: Upload de documents ---
    with tab1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📤 Importer des documents")

        uploaded_files = st.file_uploader(
            "Glissez-déposez vos documents ou cliquez pour parcourir",
            type=['pdf', 'docx', 'xlsx', 'txt'],
            accept_multiple_files=True,
            help="Formats supportés : PDF, DOCX, XLSX, TXT"
        )

        if uploaded_files:
            st.subheader(f"📁 {len(uploaded_files)} fichier(s) sélectionné(s)")

            if st.button("🚀 Indexer les documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Traitement de {uploaded_file.name}...")

                        # Extraire le texte selon le format
                        file_extension = Path(uploaded_file.name).suffix
                        text = DocumentProcessor.process_uploaded_file(uploaded_file, file_extension)

                        if text and text.strip():
                            # Insérer dans le RAG
                            asyncio.run(insert_document_to_rag(text, uploaded_file.name))
                            st.success(f"✅ {uploaded_file.name} indexé avec succès!")
                            st.session_state.uploaded_files_count += 1
                            # Ajouter à la liste des fichiers uploadés
                            if uploaded_file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append({
                                    'name': uploaded_file.name,
                                    'size': uploaded_file.size,
                                    'type': file_extension,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                        else:
                            st.warning(f"⚠️ {uploaded_file.name} ne contient pas de texte exploitable")

                    except Exception as e:
                        st.error(f"❌ Erreur avec {uploaded_file.name}: {str(e)}")

                    # Mise à jour de la barre de progression
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.text("✨ Indexation terminée!")
                st.balloons()

        st.markdown('</div>', unsafe_allow_html=True)

        # Afficher les formats supportés
        with st.expander("ℹ️ Formats supportés"):
            st.markdown("""
            - **PDF** (.pdf) : Documents Adobe PDF
            - **Word** (.docx) : Documents Microsoft Word
            - **Excel** (.xlsx) : Feuilles de calcul Excel
            - **Texte** (.txt) : Fichiers texte brut
            """)

    # --- TAB 2: Interrogation du RAG ---
    with tab2:
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        st.header("💬 Poser une question au RAG")

        # Zone de saisie de la question
        question = st.text_area(
            "Votre question :",
            height=100,
            placeholder="Ex: Quel technicien certifié est disponible en Occitanie ?",
            help="Posez une question sur les documents que vous avez indexés"
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            query_mode = st.selectbox(
                "Mode de recherche :",
                options=['hybrid', 'naive', 'local', 'global'],
                index=0,
                help="Hybrid est recommandé pour des questions complexes"
            )

        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            query_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)

        if query_button and question.strip():
            try:
                with st.spinner(f"Recherche en mode {query_mode}..."):
                    result = asyncio.run(query_rag(question, query_mode))

                    # Afficher le résultat
                    st.subheader("📝 Réponse :")
                    st.markdown(result)

                    # Section Sources du Retrieval
                    with st.expander("📚 Sources utilisées pour cette réponse", expanded=False):
                        st.markdown(f"""
                        **Mode de recherche :** `{query_mode}`

                        **Type de retrieval :**
                        """)

                        if query_mode == "naive":
                            st.info("🔍 **RAG classique** - Recherche par similarité vectorielle dans les chunks de documents")
                        elif query_mode == "local":
                            st.info("🔗 **Graph local** - Entités et relations proches (1 hop) dans le knowledge graph")
                        elif query_mode == "global":
                            st.info("🌐 **Graph global** - Patterns et structures globales du knowledge graph")
                        elif query_mode == "hybrid":
                            st.info("🎯 **Hybrid (Recommandé)** - Combinaison de recherche locale ET globale pour des réponses multi-hop complexes")

                        st.markdown("---")
                        st.markdown("### 📄 Documents indexés dans le knowledge graph")

                        if st.session_state.uploaded_files_list:
                            for idx, file_info in enumerate(st.session_state.uploaded_files_list, 1):
                                st.markdown(f"""
                                **{idx}. {file_info['name']}**
                                - 📁 Type: `{file_info['type']}`
                                - 📊 Taille: {file_info['size'] / 1024:.2f} KB
                                - 🕐 Indexé le: {file_info['timestamp']}
                                """)
                        else:
                            st.warning("Aucun document indexé pour le moment")

                        st.markdown("---")
                        st.markdown(f"""
                        **Total de documents :** {st.session_state.uploaded_files_count}

                        **Répertoire de stockage :** `./storage/`

                        💡 *Les sources proviennent du knowledge graph construit à partir de ces documents.*
                        """)

                        # Lien vers le dossier storage pour inspection manuelle
                        if os.path.exists("./storage"):
                            st.caption("Les fichiers du knowledge graph sont disponibles dans le dossier `storage/`")

                    # Ajouter à l'historique
                    st.session_state.query_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'question': question,
                        'mode': query_mode,
                        'answer': result
                    })

                    st.success("✅ Réponse générée avec succès!")

            except Exception as e:
                st.error(f"❌ Erreur lors de la requête : {str(e)}")

        elif query_button and not question.strip():
            st.warning("⚠️ Veuillez saisir une question avant de lancer la recherche.")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: Historique ---
    with tab3:
        st.header("📜 Historique des requêtes")

        if st.session_state.query_history:
            # Bouton pour effacer l'historique
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.query_history = []
                st.rerun()

            # Afficher l'historique en ordre inverse (plus récent en premier)
            for idx, query in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"🕐 {query['timestamp']} - Mode: {query['mode']}"):
                    st.markdown(f"**Question:** {query['question']}")
                    st.markdown("---")
                    st.markdown(f"**Réponse:**")
                    st.markdown(query['answer'])
        else:
            st.info("Aucune requête dans l'historique. Commencez par poser une question dans l'onglet 'Interroger le RAG'.")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888;">☕ CaféIA - Powered by GraphRAG & Ollama</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
