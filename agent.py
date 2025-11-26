#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
import os

# Load .env file (for local development)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables!")

if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in environment variables!")

if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY in environment variables!")

import json
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain.tools import tool, Tool

# Initialize Groq LLM and search tool
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=2)

# Define WASH sector domain keywords and topics
WASH_DOMAINS = [
    "sanitation", "hygiene", "wash", "water sanitation hygiene", 
    "toilet", "handwashing", "clean water", "waste management",
    "sewage", "wastewater", "public health", "waterborne diseases",
    "open defecation", "sanitation facilities", "hygiene promotion",
    "water supply", "water treatment", "solid waste", "liquid waste",
    "faecal sludge", "menstrual hygiene", "community-led total sanitation",
    "water quality", "water safety", "sanitation systems", "hygiene education"
]

# Create embeddings for semantic similarity
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create documents for WASH domains
wash_documents = [Document(page_content=domain, metadata={"domain": "wash"}) for domain in WASH_DOMAINS]

# Create vector store
vector_store = FAISS.from_documents(wash_documents, embeddings)

def is_wash_related_query(query, threshold=0.3):
    """
    Check if the query is related to WASH sector using semantic similarity
    """
    try:
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Search for similar documents
        similar_docs = vector_store.similarity_search_with_score(query, k=3)

        # Check if any similarity score meets the threshold
        for doc, score in similar_docs:
            if score >= threshold:
                return True

        # Additional keyword-based check for safety
        query_lower = query.lower()
        wash_keywords = ['sanitation', 'hygiene', 'wash', 'toilet', 'handwash', 'water', 'waste', 'sewage']
        return any(keyword in query_lower for keyword in wash_keywords)

    except Exception as e:
        print(f"Error in similarity check: {e}")
        # Fallback to keyword matching
        query_lower = query.lower()
        wash_keywords = ['sanitation', 'hygiene', 'wash', 'toilet', 'handwash', 'water', 'waste', 'sewage']
        return any(keyword in query_lower for keyword in wash_keywords)

# Global variable to store search results
search_results_cache = {}

# Create the filtered search tool using the @tool decorator
@tool
def filtered_search_tool(query: str):
    """Search for information specifically about sanitation, hygiene, and WASH sector."""
    # First check if the query is WASH-related
    if not is_wash_related_query(query):
        return json.dumps({
            "search_performed": False, 
            "results": [], 
            "message": "Query not related to sanitation, hygiene, or WASH sector."
        })

    # If WASH-related, perform the search
    try:
        result = search_tool.invoke({"query": query})
        
        # Store the search results in cache for source extraction
        search_results_cache[query] = result
        
        # Format the results for the agent
        formatted_results = []
        for item in result:
            formatted_item = {
                "title": item.get("title", "No title"),
                "url": item.get("url", "No URL"),
                "content": item.get("content", "No content")[:200] + "..." if len(item.get("content", "")) > 200 else item.get("content", "No content")
            }
            formatted_results.append(formatted_item)
        
        return json.dumps({
            "search_performed": True, 
            "results": formatted_results,
            "message": f"Found {len(result)} search results."
        })
        
    except Exception as e:
        return json.dumps({
            "search_performed": False, 
            "results": [], 
            "message": f"Search failed: {str(e)}"
        })

# Create tool instance
filtered_tool = Tool(
    name="wash_search_tool",
    description="Search for information specifically about sanitation, hygiene, and WASH sector",
    func=filtered_search_tool  # Use the tool function directly
)

# Enhanced system prompt with domain restriction
system_prompt = """You are a specialized AI chatbot focused exclusively on sanitation, hygiene, and WASH (Water, Sanitation, and Hygiene) sector. 

Your knowledge is strictly limited to:
- Sanitation systems and technologies
- Hygiene practices and promotion
- Water supply and treatment
- Waste management (solid and liquid)
- Public health aspects of WASH
- Community WASH programs
- WASH-related policies and research

If a query is not related to these topics, you must respond: "I'm specialized in sanitation, hygiene, and WASH sector and don't have knowledge about this topic."

For WASH-related queries, provide accurate, helpful information using your built-in knowledge. Only use the search tool when specifically instructed to search for current information."""

# Create two different agents - one with search and one without
agent_with_search = create_react_agent(
    model=groq_llm,
    tools=[filtered_tool],
    prompt=system_prompt
)

# Agent without search tools
agent_without_search = create_react_agent(
    model=groq_llm,
    tools=[],  # No search tools
    prompt=system_prompt
)

def get_agent_response(query, allow_web_search=True):
    """
    Get response from the WASH-specialized agent
    """

    # Determine WASH relevance once at the start
    wash_related_flag = is_wash_related_query(query)

    # If the query is not WASH-related
    if not wash_related_flag:
        return {
            "answer": "I'm specialized in sanitation, hygiene, and WASH sector and don't have knowledge about this topic.",
            "sources": [],
            "search_performed": False,
            "is_wash_related": False
        }

    # Clear previous search results for this query
    if query in search_results_cache:
        del search_results_cache[query]

    # Choose the appropriate agent
    if allow_web_search:
        agent_to_use = agent_with_search
        print("Using agent WITH web search")
    else:
        agent_to_use = agent_without_search
        print("Using agent WITHOUT web search")

    # Process the query
    state = {"messages": [HumanMessage(content=query)]}
    try:
        response = agent_to_use.invoke(state)
        messages = response.get("messages", [])
        ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

        final_response = ai_messages[-1] if ai_messages else "No response generated."

        # Blocked / unrelated WASH response
        if "Query not related to sanitation" in final_response or \
           "I can only search for WASH-related" in final_response:
            
            return {
                "answer": "I'm specialized in sanitation, hygiene, and WASH sector and don't have knowledge about this topic.",
                "sources": [],
                "search_performed": False,
                "is_wash_related": wash_related_flag
            }

        # Extract search results
        sources = []
        search_was_performed = False
        
        if allow_web_search and query in search_results_cache:
            search_results = search_results_cache[query]
            search_was_performed = True

            for i, result in enumerate(search_results):
                if i >= 2:
                    break
                sources.append({
                    "title": result.get("title", "No title"),
                    "url": result.get("url", "No URL"),
                    "content": (
                        result.get("content", "No content")[:100] + "..."
                        if len(result.get("content", "")) > 100
                        else result.get("content", "No content")
                    )
                })

        return {
            "answer": final_response,
            "sources": sources,
            "search_performed": search_was_performed,
            "is_wash_related": wash_related_flag
        }

    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": [],
            "search_performed": False,
            "is_wash_related": wash_related_flag
        }

# Test the agent
print("Testing the WASH-specialized agent...")