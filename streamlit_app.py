import os
import json
from datetime import datetime, timedelta, timezone
import praw
from groq import Groq
import streamlit as st
from langchain.schema import Document
import json
from scrapingbee import ScrapingBeeClient
from reddit_rag import get_reddit_rag 
# Get RAG instance r
rag = get_reddit_rag() 

# --- Clients ---


SCRAPINGBEE_API_KEY = "U3URPLPZWZ3QHVGEEP5HTXJ95873G9L58RJ3EHS4WSYTXOZAIE71L278CF589042BBMKNXZTRY23VYPF"

def fetch_recent_reddit_posts(hours=24, limit=100):
    documents = []

    client = ScrapingBeeClient(api_key=SCRAPINGBEE_API_KEY)

    response = client.get(
        "https://finance.yahoo.com/topic/latest-news/",
        params={
            "render_js": True,
            "ai_query": (
                "Extract all article headlines with their publication time and absolute URLs. "
                "Return as plain text, one article per line."
            )
        }
    )

    if response.status_code != 200:
        raise RuntimeError(f"ScrapingBee failed: {response.status_code} - {response.text}")

    documents.append(
        Document(
            page_content=response.text.strip(),
            metadata={
                "source": "yahoo_finance",
                "type": "news",
            }
        )
    )

    return documents

rag.vector_store.add_documents(fetch_recent_reddit_posts())

    
def market_copilot(query):
    return reddit_news_chatbot(query)

def reddit_news_chatbot(query, lookback_hours=24):
    # Import here to avoid circular imports
    
    # Query RAG system
    
    result = rag.query(query, lookback_hours=lookback_hours)
    
    return result


question = st.chat_input("Type your question and press Enter...")
st.write("Questions or feedback? Email hello@stockdoc.biz.")

if question:
  response=market_copilot(question)
  st.write(response)
