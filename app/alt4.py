import os
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import chromadb
import numpy as np
import json
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

# ====================
# Configuration (Best Practice: Use environment variables)
# ====================
# It is recommended to set your credentials as environment variables
# especially in a production environment, for enhanced security.
# You can set the following environment variables:
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ctmatchinggpt.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "af6c5f2c43294f1e9287a50d652c637e")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "GPTbgsw-openAIservice-Voiceseatbooking4")
CHROMA_PERSIST_DIR = "idea_vector_store"
COLLECTION_NAME = "ideas_v2"

# ====================
# Pydantic Models
# ====================
class IdeaItem(BaseModel):
    Id: int
    Title: str
    Desc: str

class MatchRequest(BaseModel):
    NewIdea: IdeaItem
    TopK: int = 10 # Number of initial candidates to retrieve

class MatchedIdea(BaseModel):
    IdeaId: int
    SimilarityPercent: int

class MatchResponse(BaseModel):
    IsSuccess: bool = True
    Message: str = "SUCCESS"
    ErrorMessage: Optional[str] = None
    MatchedIdeasCount: int = 0
    MatchedIdeas: List[MatchedIdea] = []

class AddIdeasRequest(BaseModel):
    Ideas: List[IdeaItem]

class AddIdeasResponse(BaseModel):
    IsSuccess: bool = True
    Message: str
    IdsAdded: List[int]

# ====================
# Global App State (managed by lifespan)
# ====================
# This dictionary will hold our initialized models and clients.
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    print("Application startup: Initializing models and vector store...")
    
    # 1. Initialize Embeddings and LLM
    embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint="https://ctmatchinggpt.openai.azure.com/",
    api_key="af6c5f2c43294f1e9287a50d652c637e",
    azure_deployment="text-embedding-3-large",
    model="text-embedding-3-large",
    api_version="2024-02-01"
    )

    llm = AzureChatOpenAI(
                    azure_deployment="GPTbgsw-openAIservice-Voiceseatbooking4",
                    api_key="2u2cSvJIlkFgj6BKsabPTVeIS4zcFlCu49yk2JxzrmUkIDTycp9qJQQJ99BHACYeBjFXJ3w3AAABACOGoOsh", 
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    azure_endpoint="https://bgsw-openaiservice-voiceseatbooking-p-eus-001.openai.azure.com/openai/deployments/bgsw-openAIservice-Voiceseatbooking/chat/completions?api-version=2025-01-01-preview", 
                    temperature=0)

    # 2. Initialize Persistent ChromaDB Client
    # This client connects to a database stored on disk.
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # 3. Create LangChain Chroma instance from the client
    # This allows us to use LangChain's familiar interface.
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"} # Use cosine distance for similarity
    )  
    app_state["vectorstore"] = vectorstore
    app_state["llm"] = llm
    
    # 4. Initialize Caching
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Initialization complete.")
    
    yield
    # Runs on shutdown
    print("Application shutdown.")
    app_state.clear()


# ====================
# FastAPI App Setup
# ====================
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ====================
# Helper Functions
# ====================
@cache(expire=3600) # Cache results for 1 hour
async def get_batched_contextual_validation(new_idea: IdeaItem, candidate_ideas: str, llm: AzureChatOpenAI) -> dict:
    """
    Use a single LLM call to validate a batch of candidate ideas.
    `candidate_ideas` is a JSON string to make it cacheable.
    """
    prompt = f"""
    Analyze the New Idea against each of the Old Ideas provided in the JSON list.
    For each Old Idea, determine if it is contextually similar to the New Idea in its core purpose, target domain, or the problem it solves.

    New Idea:
    - Title: {new_idea.Title}
    - Description: {new_idea.Desc}

    Old Ideas (JSON Array):
    {candidate_ideas}

    Your response MUST be a single, valid JSON object (no extra text or markdown).
    The JSON object should have keys as the idea IDs (as strings) and values as a boolean: `true` if contextually similar, `false` otherwise.
    Example response:
    {{
      "101": true,
      "102": false,
      "103": true
    }}
    """
    try:
        response = await llm.ainvoke(prompt)
        # Using json.loads to parse the string response from the LLM into a dictionary
        return json.loads(response.content)
    except Exception as e:
        print(f"Error parsing LLM response for contextual validation: {e}")
        # Fallback: if LLM fails, conservatively assume no contextual match
        return {}

# ====================
# API Endpoints
# ====================
@app.post("/addIdeas", response_model=AddIdeasResponse)
async def add_ideas(request: AddIdeasRequest, vectorstore: Chroma = Depends(lambda: app_state["vectorstore"])):
    """
    Endpoint to add new ideas to the persistent vector store.
    """
    if not request.Ideas:
        raise HTTPException(status_code=400, detail="Ideas list cannot be empty.")

    docs = []
    ids = []
    for idea in request.Ideas:
        text = f"{idea.Title}. {idea.Desc}"
        docs.append(
            Document(
                page_content=text,
                metadata={"id": idea.Id, "title": idea.Title, "description": idea.Desc}
            )
        )
        ids.append(str(idea.Id))
    try:
        vectorstore.add_documents(docs, ids=ids)
        return AddIdeasResponse(Message=f"Successfully added {len(ids)} ideas.", IdsAdded=[idea.Id for idea in request.Ideas])
    except Exception as e:
        # This can happen if an ID already exists. Chroma can be configured to handle this.
        raise HTTPException(status_code=500, detail=f"Failed to add ideas to vector store: {str(e)}")


@app.post("/findMatches", response_model=MatchResponse)
async def find_matches(request: MatchRequest):
    """
    This is the main, optimized endpoint for finding similar ideas.
    It queries the existing vector store and uses a batched LLM call for validation.
    """
    vectorstore = app_state["vectorstore"]
    llm = app_state["llm"]
    new_idea = request.NewIdea
    final_matches = []
    
    try:
        # -----------------
        # Step 1: Query the persistent vector store
        # -----------------
        query_text = f"{new_idea.Title}. {new_idea.Desc}"
        # We use similarity_search_with_relevance_scores which uses cosine similarity if configured
        # The score is between 0 and 1, where 1 is most similar.
        results = await vectorstore.asimilarity_search_with_relevance_scores(query_text, k=request.TopK)
        
        if not results:
             return MatchResponse(MatchedIdeasCount=0, MatchedIdeas=[])
        
        # -----------------
        # Step 2: Prepare for Batched LLM Validation
        # -----------------
        candidate_ideas_for_llm = []
        # Store original scores to use later
        original_scores = {}

        for doc, score in results:
            similarity_percent = int(round(score * 100))
            idea_id = doc.metadata["id"]
            original_scores[idea_id] = similarity_percent
            
            # Pre-filter to avoid sending clearly irrelevant ideas to the LLM
            if similarity_percent >= 30:
                candidate_ideas_for_llm.append({
                    "id": idea_id,
                    "title": doc.metadata["title"],
                    "description": doc.metadata["description"]
                })

        # -----------------
        # Step 3: Perform Batched Contextual Validation (if any candidates)
        # -----------------
        contextual_relevance = {}
        if candidate_ideas_for_llm:
            # Convert to JSON string for caching key
            candidates_json_str = json.dumps(candidate_ideas_for_llm, sort_keys=True)
            contextual_relevance = await get_batched_contextual_validation(new_idea, candidates_json_str, llm)

        # -----------------
        # Step 4: Adjust Scores and Finalize Matches
        # -----------------
        SIMILARITY_THRESHOLD = 30 # Final filter for results to be returned
        
        for idea_id, original_percent in original_scores.items():
            is_relevant = contextual_relevance.get(str(idea_id), False) # LLM result for this ID
            
            if is_relevant:
                # If contextually relevant, keep the high score
                adjusted_score = original_percent
            else:
                # If NOT contextually relevant, significantly penalize the score
                # but only if it was sent to the LLM for validation.
                if any(c["id"] == idea_id for c in candidate_ideas_for_llm):
                    adjusted_score = max(0, original_percent - 40)
                else: # Was not sent to LLM, keep original score
                    adjusted_score = original_percent

            if adjusted_score >= SIMILARITY_THRESHOLD:
                final_matches.append(MatchedIdea(IdeaId=idea_id, SimilarityPercent=adjusted_score))

        # Sort by final score
        final_matches.sort(key=lambda x: x.SimilarityPercent, reverse=True)
        
        return MatchResponse(
            MatchedIdeasCount=len(final_matches),
            MatchedIdeas=final_matches
        )

    except Exception as e:
        print(f"Error during match finding: {str(e)}")
        # Use a more robust error response
        return MatchResponse(
            IsSuccess=False,
            Message="FAILURE",
            ErrorMessage=str(e),
            MatchedIdeasCount=0,
            MatchedIdeas=[]
        )
