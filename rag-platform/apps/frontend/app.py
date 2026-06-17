import streamlit as st
import httpx
import os

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="ThreatIntel RAG", layout="centered")
st.title("ThreatIntel RAG")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("API Key", type="password", help="x-api-key header")
    search_type = st.selectbox("Search Type", ["hybrid", "vector"], index=0)
    enable_reranking = st.checkbox("Enable Reranking", value=True)

question = st.text_input("Question", placeholder="e.g., What is CVE-2024-1234?")
top_k = st.slider("Top K", 1, 10, 5)

if st.button("Ask", type="primary", disabled=not question):
    headers = {"x-api-key": api_key} if api_key else {}
    payload = {
        "question": question,
        "top_k": top_k,
        "search_type": search_type,
        "enable_reranking": enable_reranking,
    }
    try:
        with st.spinner("Querying..."):
            resp = httpx.post(API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        
        st.markdown(data["answer"])
        
        sources = data.get("sources", [])
        scores = data.get("source_scores", [])
        if sources:
            with st.expander(f"Sources ({data['context_used']} chunks)"):
                for i, (src, score) in enumerate(zip(sources, scores)):
                    st.write(f"**{i+1}. {src}** — score: `{score:.4f}`")
        
        st.caption(f"Context: {data['context_used']} | Type: {data['search_type']} | Reranked: {data['reranked']}")
    
    except httpx.ConnectError:
        st.error("Cannot connect to API. Is server running on http://localhost:8000?")
    except httpx.TimeoutException:
        st.error("Request timed out (>60s). LLM is slow on free tier.")
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Error: {e}")