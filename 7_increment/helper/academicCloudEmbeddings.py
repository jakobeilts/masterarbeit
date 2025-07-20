import requests, numpy as np, streamlit as st
from langchain_core.embeddings import Embeddings

# ------------------------------------------------------------------
#  Task-prompt library – grow or swap these as you wish
# ------------------------------------------------------------------
PROMPTS = {
    "web_search_query":
        "Instruct: Given a web search query, retrieve relevant passages that answer the query",
    "sts_query":
        "Instruct: Retrieve semantically similar text.",
    "summarization_query":
        "Instruct: Given a news summary, retrieve other semantically similar summaries",
    # add your own e.g.:
    "refute_claim":
        "Instruct: Given a claim, find documents that refute the claim",
}

def build_query(text: str, prompt_name: str) -> str:
    """Turn raw query text into the model-expected format."""
    instruction = PROMPTS[prompt_name]
    return f"{instruction}\nQuery: {text}"


class AcademicCloudEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper for the AcademicCloud /
    OpenAI-compatible endpoint that *automatically* adds the
    instruction prefix on the query side only.
    """
    def __init__(
            self,
            api_key: str,
            url:   str,
            model: str = "e5-mistral-7b-instruct",
            prompt_name: str = "web_search_query",          # e5-mistral-7b-instruct takes instructions before embeddings. Website says it is neccessary and will perform worse without...
    ):
        self.api_key      = api_key
        self.model        = model
        self.url          = url
        self.prompt_name  = prompt_name                # remember which prompt to use

    # ------------- public API required by LangChain ----------------
    def embed_documents(self, texts):
        # NO instruction on the document side
        return self._embed(texts)

    def embed_query(self, text):
        # prepend instruction + "Query:" line
        query_with_instruction = build_query(text, self.prompt_name)
        return self._embed([query_with_instruction])[0]
    # ---------------------------------------------------------------

    # shared helper
    def _embed(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload  = {
            "input":           texts,
            "model":           self.model,
            "encoding_format": "float",
        }
        resp = requests.post(self.url, headers=headers, json=payload)
        resp.raise_for_status()
        return [np.array(d["embedding"]) for d in resp.json()["data"]]