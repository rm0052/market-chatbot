import os
import json
from datetime import datetime, timedelta, timezone
import praw
from groq import Groq
import streamlit as st
from langchain.schema import Document
import json
from reddit_rag import get_reddit_rag 
# Get RAG instance r
rag = get_reddit_rag() 

# --- Clients ---

reddit = praw.Reddit(
    client_id="Ly24yiY7yWoF5CboIG217w",
    client_secret="iffbq4WIHgjEFUvsFFzAdqzbSsFYNQ",
    username="Basic-Cry-2405",
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent="market-copilot-reddit-agent"
)

SUBREDDITS = ["stocks","investing","pennystocks","Options","SecurityAnalysis","DividendInvesting","cryptocurrency","cryptomarkets","Bitcoin","wallstreetbets"]


def fetch_recent_reddit_posts(hours=24, limit=100):
    subreddit = reddit.subreddit("+".join(SUBREDDITS))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    documents = []
    for submission in subreddit.new(limit=limit):
        post_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if post_time < cutoff:
            continue
        if submission.stickied or submission.over_18:
            continue
        content = f""" Title: {submission.title} URL: {submission.url} Content: {submission.selftext[:2000]} """       
        documents.append( Document( page_content=content.strip(), metadata={ "source": "reddit", "subreddit": submission.subreddit.display_name, "created_utc": submission.created_utc, } ) )
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
