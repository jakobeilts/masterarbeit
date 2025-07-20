import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from helper.academicCloudEmbeddings import AcademicCloudEmbeddings
import json, uuid, datetime, pathlib

system_prompt = """
Du bist der hilfreiche StudIT‑Assistent der Universität Göttingen.
• Antworte bevorzugt auf Deutsch.
• Antworte so kurz wie möglich, aber so ausführlich wie nötig.
• Falls du die Antwort nicht sicher weißt, schlage konkrete Anlaufstellen (Link oder E‑Mail) vor. Gib niemals eine Antwort von der du nicht 100% sicher bist, dass sie stimmt.
"""

# Vorlage, mit der die Nutzerfrage in einen suchbaren Satz umformuliert wird
condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}")
])

# Vorlage, mit der die eigentliche Antwort formuliert wird
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt +
     "\nNutze ausschließlich die folgenden Kontext‑Ausschnitte, "
     "wenn du die Frage beantwortest:\n{context}\n"),
    ("user", "{question}")
])

# Lege einen Ordner für Logs an (falls nicht vorhanden)
LOG_PATH = pathlib.Path("logs")
LOG_PATH.mkdir(exist_ok=True)

def log_conversation(user_msg: str, assistant_msg: str) -> None:
    """Schreibt eine Chat-Zeile als JSONL.

    Eine Sitzung bekommt beim ersten Aufruf eine UUID,
    danach hängt jede Zeile dieselbe session_id an.
    """
    # Session‑ID einmalig erzeugen & merken
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    entry = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "user": user_msg,
        "assistant": assistant_msg,
    }

    # Append im JSON‑Lines‑Format
    with open(LOG_PATH / "conversations.jsonl", "a", encoding="utf‑8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

embedder = AcademicCloudEmbeddings(
    api_key=st.secrets["GWDG_API_KEY"],
    url=st.secrets["BASE_URL_EMBEDDINGS"],
)

vector_store = FAISS.load_local(
    "faiss_wiki_index",
    embedder,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOpenAI(
    base_url=st.secrets["BASE_URL"],
    api_key=st.secrets["GWDG_API_KEY"],
    model_name="meta-llama-3.1-8b-instruct",
    temperature=0,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    condense_question_prompt=condense_question_prompt,
    combine_docs_chain_kwargs=dict(prompt=answer_prompt),
    memory=memory,
    return_source_documents=True,
)

# -----------------------------------------------------------------------------
# Hilfsfunktion zur Zusammenfassung
# -----------------------------------------------------------------------------
def summarize_chat(chat_history):
    chat_as_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    summary_prompt = f"Fasse diesen Chatverlauf kompakt und verständlich zusammen:\n\n{chat_as_text}"
    response = llm.invoke(summary_prompt)
    return response.content

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="StudIT RAG-Chat", page_icon="💬", layout="centered")
st.title("💬 StudIT RAG-Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_welcome = True  # Nur beim ersten Laden

# Begrüßungsnachricht anzeigen, aber nicht speichern
if st.session_state.get("show_welcome", False):
    st.chat_message("assistant").write(
        "Willkommen zum Chatbot vom StudIT. Frag mich gerne zu Themen wie Eduroam, "
        "dem Drucken auf dem Campus oder deinem Studienausweis. Falls ich dir nicht ausreichend weiterhelfen kann, "
        "kannst du dich auch direkt über den Button an die StudIT wenden."
    )
    st.session_state.show_welcome = False
if "show_support_form" not in st.session_state:
    st.session_state.show_support_form = False
if "support_summary" not in st.session_state:
    st.session_state.support_summary = ""
if "support_name" not in st.session_state:
    st.session_state.support_name = ""
if "support_email" not in st.session_state:
    st.session_state.support_email = ""

# Chatverlauf anzeigen
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Nutzereingabe
user_prompt = st.chat_input("Stell eine Frage...")

if user_prompt:
    st.chat_message("user").write(user_prompt)

    with st.spinner("Denke nach …"):
        result = qa_chain({"question": user_prompt})
        answer: str = result["answer"]
        sources = result.get("source_documents", [])
        # ── im Terminal ausgeben ───────────────────────────────────────────
        for i, doc in enumerate(sources, 1):
            print(f"\n=== Chunk {i} ===")
            print(doc.page_content)  # eigentlicher Text
            print("Metadaten:", doc.metadata)  # z. B. URL, Seiten‑/Chunk‑ID

    st.chat_message("assistant").write(answer)

    if sources:
        with st.expander("🔗 Quellen anzeigen"):
            for i, doc in enumerate(sources, 1):
                source_label = doc.metadata.get("url", "(keine URL)")
                st.markdown(f"**{i}. {source_label}**")

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Hier wird geloggt
    log_conversation(user_prompt, answer)

# -----------------------------------------------------------------------------
# Support-Anfrage-Funktionalität
# -----------------------------------------------------------------------------

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
