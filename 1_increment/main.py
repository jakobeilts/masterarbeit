import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from helper.academicCloudEmbeddings import AcademicCloudEmbeddings

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
    temperature=0, )

# WICHTIG: return_source_documents=True, damit wir die verwendeten Dokumente zurückbekommen
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="📚 Vector‑Aware Chatbot", page_icon="💬", layout="centered")
st.title("💬 Dokument‑Awarer Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": str, "content": str}

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_prompt := st.chat_input("Stell eine Frage zu deinen Dokumenten …"):
    st.chat_message("user").write(user_prompt)

    with st.spinner("Denke nach …"):
        result = qa_chain({"question": user_prompt})
        answer: str = result["answer"]
        sources = result.get("source_documents", [])

    st.chat_message("assistant").write(answer)

    # Quellen in einem Expander anzeigen, damit sie nicht stören
    if sources:
        with st.expander("🔗 Quellen anzeigen"):
            for i, doc in enumerate(sources, 1):
                # Versuche, eine hilfreiche Quelle anzuzeigen – Pfad, URL o.Ä.
                source_label = doc.metadata.get("url", "(keine URL)")
                st.markdown(f"**{i}. {source_label}**")
                # Zeige einen kurzen Auszug aus dem Dokument
                # snippet = doc.page_content.strip()
                # st.markdown(snippet[:400] + (" …" if len(snippet) > 400 else ""))
                # st.markdown("---")

    # Verlauf speichern
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
