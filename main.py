#!/usr/bin/env python
# coding: utf-8

# In[11]:


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Import from your existing agent code
from agent import get_agent_response, is_wash_related_query, WASH_DOMAINS

app = FastAPI(
    title="WASH Bot API",
    description="Specialized AI chatbot for Water, Sanitation, and Hygiene sector",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    allow_web_search: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    search_performed: bool
    is_wash_related: bool

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        message="WASH Bot API is running successfully"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy", 
        message="WASH Bot API is running successfully"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        response = get_agent_response(
            query=chat_request.query,
            allow_web_search=chat_request.allow_web_search
        )
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/check-wash-related")
async def check_wash_related(query: str):
    try:
        is_related = is_wash_related_query(query)
        return {"query": query, "is_wash_related": is_related}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking query: {str(e)}")

@app.get("/wash-domains")
async def get_wash_domains():
    return {"domains": WASH_DOMAINS, "count": len(WASH_DOMAINS)}


# In[ ]:




