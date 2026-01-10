import os
import json
from datetime import datetime, timedelta, timezone
import streamlit as st
from langchain.schema import Document
from scrapingbee import ScrapingBeeClient
from reddit_rag import get_reddit_rag

# Get RAG instance
rag = get_reddit_rag()

# --- Clients ---


# Get ScrapingBee API key from environment variables or use the default
SCRAPINGBEE_API_KEY = os.environ.get(
    "SCRAPINGBEE_API_KEY", 
    "U3URPLPZWZ3QHVGEEP5HTXJ95873G9L58RJ3EHS4WSYTXOZAIE71L278CF589042BBMKNXZTRY23VYPF"
)


def scrape_bloomberg():
    client = ScrapingBeeClient(api_key=SCRAPINGBEE_API_KEY)
    url = "https://finance.yahoo.com/topic/latest-news/"

    response = client.get(
        url,
        params={
            "render_js": True,
            "ai_query": (
                "Extract all article headlines with summaries and absolute URLs. "
                "Return one article per line."
            )
        }
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    documents = []

    for line in response.text.split("\n"):
        if line.strip():
            documents.append(
                Document(
                    page_content=line.strip(),
                    metadata={
                        "source": "yahoo_finance",
                        "type": "news"
                    }
                )
            )

    return documents

    
# Add scraped documents to vector store
try:
    documents = scrape_bloomberg()
    if documents:
        st.info(f"Adding {len(documents)} documents to vector store")
        rag.vector_store.add_documents(documents)
    else:
        st.warning("No documents retrieved from Yahoo Finance")
except Exception as e:
    st.error(f"Error scraping Yahoo Finance: {str(e)}")

    
def market_copilot(query):
    return reddit_news_chatbot(query)

def reddit_news_chatbot(query, lookback_hours=24):
    # Query RAG system
    try:
        result = rag.query(query, lookback_hours=lookback_hours)
        
        # Debug information
        st.sidebar.write("Debug Info:")
        st.sidebar.write(f"Answer length: {len(result.get('answer', ''))}")
        st.sidebar.write(f"Number of sources: {len(result.get('sources', []))}")
        
        return result
    except Exception as e:
        st.error(f"Error querying RAG system: {str(e)}")
        return {"answer": f"An error occurred: {str(e)}", "sources": []}


# Streamlit UI
st.title("Market News Chatbot")
st.write("Ask questions about the latest market news")

# Input for question
question = st.chat_input("Type your question and press Enter...")
st.write("Questions or feedback? Email hello@stockdoc.biz.")

if question:
    st.write(f"**Your question:** {question}")
    with st.spinner("Searching for information..."):
        response = market_copilot(question)
    
    # Display answer
    st.markdown("### Answer:")
    st.write(response.get("answer", "No answer found"))
    
    # Display sources
    if response.get("sources"):
        st.markdown("### Sources:")
        for i, source in enumerate(response["sources"]):
            st.markdown(f"**{i+1}. {source.get('title', 'Unknown title')}**")
            if "url" in source and source["url"]:
                st.markdown(f"[Link]({source['url']})")
    else:
        st.info("No specific sources found for this answer.")
