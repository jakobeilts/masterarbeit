# ============================================================================
# main.py – StudIT RAG‑Chat   (Hybrid Vector + BM25)
#   • Fix: richtige Dateipfade & Fallback, damit kein TypeError entsteht
# ============================================================================

import json, uuid, datetime, pathlib, pickle
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.retrievers import (
    ParentDocumentRetriever,
    BM25Retriever,
    EnsembleRetriever,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.storage import InMemoryStore

# ----------------------------------------------------------------------------
# 1) System‑ & Prompt‑Vorlagen
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Du bist der hilfreiche StudIT‑Assistent der Universität Göttingen.\n"
    "• Antworte bevorzugt auf Deutsch.\n"
    "• Antworte so kurz wie möglich, aber so ausführlich wie nötig.\n"
    "• Gib immer konkrete Antworten. Keine allgemeinen Tipps.\n"
    "• Falls du die Antwort nicht sicher weißt, schlage konkrete Anlaufstellen (Link oder E‑Mail) vor oder schlag vor über den Supportanfrage stellen Button den Support direkt aus dem Chat anzufragen.\n"
    "  Gib niemals eine Antwort von der du nicht 100% sicher bist, dass sie stimmt."
)

condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Formuliere aus der Folgefrage unter Berücksichtigung des bisherigen "
     "Gesprächs eine eigenständige, präzise Frage. Gib **nur** diese Frage zurück."),
    ("system", "Bisheriger Chat:\n{chat_history}"),
    ("human", "Folgefrage: {question}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_PROMPT
        + "\nNutze ausschließlich die folgenden Kontext‑Ausschnitte, wenn du die Frage beantwortest:\n{context}\n",
    ),
    ("user", "{question}"),
])

# ----------------------------------------------------------------------------
# 2) Embeddings & Stores laden
# ----------------------------------------------------------------------------

embedder = OpenAIEmbeddings(
    api_key=st.secrets["OPENAI_API_KEY"],
    model="text-embedding-3-large",
)

# --- Child‑VectorStore (FAISS) ----------------------------------------------
child_index = FAISS.load_local(
    "faiss_children",
    embedder,
    allow_dangerous_deserialization=True,
)

# --- Parent‑DocStore ---------------------------------------------------------
PARENT_STORE_PATH = pathlib.Path("parent_docstore.pkl")
if PARENT_STORE_PATH.exists():
    docstore: InMemoryStore = pickle.loads(PARENT_STORE_PATH.read_bytes())
else:
    st.warning("⚠️  parent_docstore.pkl nicht gefunden – leerer InMemoryStore verwendet.")
    docstore = InMemoryStore()

# --- Dokumentliste (für BM25) ------------------------------------------------
DOC_LIST_PATH = pathlib.Path("all_docs.pkl")  # in get_data.ipynb erzeugt
if DOC_LIST_PATH.exists():
    docs = pickle.loads(DOC_LIST_PATH.read_bytes())  # => List[Document]
else:
    # Fallback: versuche, alle Docs aus dem DocStore zu ziehen
    try:
        docs = list(docstore.values())
    except Exception:
        docs = []
    st.warning("⚠️  all_docs.pkl nicht gefunden – BM25 nutzt DocStore‑Fallback mit "
               f"{len(docs)} Dokumenten.")

# ----------------------------------------------------------------------------
# 3) Hybrid‑Retriever (Vector + BM25)
# ----------------------------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

parent_retriever = ParentDocumentRetriever(
    vectorstore=child_index,
    docstore=docstore,
    child_splitter=splitter,
    parent_splitter=None,
    search_kwargs={"k": 15},
)

# --- BM25 --------------------------------------------------------------------
if docs:
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 15
    hybrid_retriever = EnsembleRetriever(
        retrievers=[parent_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
else:
    hybrid_retriever = parent_retriever

# --- Kompression / Rerank ----------------------------------------------------
compressor = RankLLMRerank(
    top_n=4,
    model="gpt",
    gpt_model="gpt-4o-mini",
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever,
)

# ----------------------------------------------------------------------------
# 4) LLM, Memory & Chain
# ----------------------------------------------------------------------------

llm = ChatOpenAI(
    base_url=st.secrets["BASE_URL"],
    api_key=st.secrets["GWDG_API_KEY"],
    model_name="meta-llama-3.1-8b-instruct",
    temperature=0,
)

# 1) Memory nur einmal pro Browser-Session erzeugen
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

# 2) QA-Chain einmalig mit diesem Memory bauen
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs=dict(prompt=answer_prompt),
        memory=st.session_state.memory,
        return_source_documents=True,
        verbose = True,
    )

# ----------------------------------------------------------------------------
# 5) Logging
# ----------------------------------------------------------------------------

LOG_PATH = pathlib.Path("logs"); LOG_PATH.mkdir(exist_ok=True)

def log_conversation(user_msg: str, assistant_msg: str):
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    entry = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "user": user_msg,
        "assistant": assistant_msg,
    }
    with open(LOG_PATH / "conversations.jsonl", "a", encoding="utf‑8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------------
# 6) Streamlit UI (wie gehabt)
# ----------------------------------------------------------------------------

st.set_page_config(page_title="StudIT Chatbot", page_icon="💬", layout="centered")
st.title("StudIT Chatbot")
st.logo("helper/images/uni_logo.png")

if "messages" not in st.session_state:
    st.session_state.messages, st.session_state.show_welcome = [], True

if st.session_state.show_welcome:
    st.chat_message("assistant").write(
        "Willkommen zum Chatbot vom StudIT. Frag mich gerne zu Themen wie Eduroam, "
        "dem Drucken auf dem Campus oder deinem Studienausweis. Falls ich dir nicht ausreichend weiterhelfen kann, "
        "kannst du dich auch direkt über den Button an die StudIT wenden."
    )
    st.session_state.show_welcome = False

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

user_prompt = st.chat_input("Stell eine Frage…")

if user_prompt:
    st.chat_message("user").write(user_prompt)
    with st.spinner("Denke nach …"):
        result = st.session_state.qa_chain({"question": user_prompt})
    answer, sources = result["answer"], result.get("source_documents", [])

    st.chat_message("assistant").write(answer)
    if sources:
        with st.expander("🔗 Quellen anzeigen"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**{i}. {doc.metadata.get('url', '(keine URL)')}**")

    st.session_state.messages += [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": answer}]
    log_conversation(user_prompt, answer)
    print(user_prompt)

# ----------------------------------------------------------------------------
# 7) Support‑Formular (wie gehabt)  – gekürzt, da unverändert
# ----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Support-Anfrage-Funktionalität
# -----------------------------------------------------------------------------

st.divider()
if st.button("Supportanfrage stellen"):
    if st.session_state.messages:
        with st.spinner("Fasse den Chat zusammen …"):
            # Erzeuge kurze Zusammenfassung (1–2 Sätze)
            chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            summary_prompt = (
                "Fasse dieses Gespräch in 1-2 Sätzen grob zusammen und adressiere dabei das StudIT-Team. "
                "Der vorliegende Chat ist von einem FAQ Chatbot und die Zusammenfassung soll dem Support Team helfen:\n\n" + chat_text
            )
            short_summary = llm.invoke(summary_prompt).content
            full_chat = chat_text
    else:
        short_summary = ""
        full_chat = ""

    st.session_state.short_summary = short_summary
    st.session_state.full_chat = full_chat
    st.session_state.user_addition = ""
    st.session_state.support_summary = ""
    st.session_state.show_support_form = True


if st.session_state.get("show_support_form", False):
    st.subheader("Support-Anfrage stellen")

    st.session_state.support_name = st.text_input("Name", value=st.session_state.get("support_name", ""))
    st.session_state.support_email = st.text_input("E-Mail-Adresse", value=st.session_state.get("support_email", ""))

    user_addition_required = not bool(st.session_state.messages)
    user_addition_label = (
        "Bitte beschreibe dein Anliegen" if user_addition_required else "Optional: Ergänze dein Anliegen manuell"
    )

    st.session_state.user_addition = st.text_area(
        user_addition_label,
        value=st.session_state.get("user_addition", ""),
        placeholder="Was möchtest du dem Support-Team mitteilen?",
        height=100
    )

    if st.button("✅ Supportanfrage absenden"):
        errors = []

        # Pflichtfelder prüfen
        if not st.session_state.support_name.strip():
            errors.append("Name darf nicht leer sein.")
        if not st.session_state.support_email.strip():
            errors.append("E-Mail-Adresse darf nicht leer sein.")
        if user_addition_required and not st.session_state.user_addition.strip():
            errors.append("Bitte gib dein Anliegen ein, bevor du die Anfrage absendest.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            # Anfrage zusammensetzen
            composed_summary = (
                "Hallo StudIT-Team. Der Nutzer/die Nutzerin unseres Chatbot hat folgendes Anliegen:\n\n"
                f"{st.session_state.short_summary.strip()}\n\n"
                "Der Nutzer/Die Nutzerin hat außerdem die folgende Nachricht hinzugefügt:\n\n"
                f"{st.session_state.user_addition.strip()}\n\n"
                "Hier ist der gesamte Chatverlauf:\n"
                f"{st.session_state.full_chat.strip()}"
            )
            st.session_state.support_summary = composed_summary

            st.success("Die Supportanfrage wurde vorbereitet. (E-Mail-Versand folgt später.)")
            st.markdown("### Vorschau der Anfrage")
            st.markdown(f"**Name:** {st.session_state.support_name}")
            st.markdown(f"**E-Mail:** {st.session_state.support_email}")
            st.markdown("**Nachricht an das Support-Team:**")
            st.markdown(st.session_state.support_summary)
            st.session_state.show_support_form = False
