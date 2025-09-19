import os
import json
import asyncio
import traceback
import re
from typing import List, Optional, TypedDict, Dict, Any
import uvicorn
# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# --- Pydantic Models for API ---
from pydantic import BaseModel, Field
# --- LangChain & LangGraph Imports ---
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph, END
import chromadb


# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# --- Load Environment Variables ---


# --- ChromaDB Configuration ---
CHROMA_PERSIST_DIR = "idea_vector_store"
COLLECTION_NAME = "ideas_v3" # Changed version to avoid conflicts

# --- LangGraph Agent Configuration ---
CANDIDATE_POOL_SIZE = 50

# --- MODIFICATION: Updated Business Logic Constants for Dynamic Filtering ---
# Defines the IdeaStatus codes to be excluded from similarity search based on the input parameters.
EXCLUDED_STATUSES_S1_IC0 = [3, 5]  # Excluded statuses when source=1 and IsChildIdea=0
EXCLUDED_STATUSES_S0_IC1 = [5]     # Excluded statuses when source=0 and IsChildIdea=1


# ==============================================================================
# 2. API DATA MODELS (Pydantic)
# ==============================================================================

class IdeaItem(BaseModel):
    Id: int
    Title: str
    Description: str
    IdeaStatus: int
    IsChildIdea: bool

class MatchedIdea(BaseModel):
    IdeaId: int
    SimilarityPercent: int
    ReasonForSimilarity: Optional[str] = None

class AddIdeasRequest(BaseModel):
    Ideas: List[IdeaItem]

class AddIdeasResponse(BaseModel):
    Message: str
    IdsAdded: List[int]
    IdsSkipped: List[int] # --- NEW: Field to report IDs that already existed

# --- NEW: Pydantic models for the update endpoint ---
class UpdateIdeasRequest(BaseModel):
    Ideas: List[IdeaItem]

class UpdateIdeasResponse(BaseModel):
    Message: str
    IdsUpdated: List[int]
    IdsNotFound: List[int]

class DeleteIdeasRequest(BaseModel):
    Ids: List[int]

class DeleteIdeasResponse(BaseModel):
    Message: str
    IdsDeleted: List[int]

class MatchRequest(BaseModel):
    NewIdea: IdeaItem = Field(..., description="The new idea to find matches for.")
    TopK: int = 10
    MinSimilarity: int = Field(
        default=0,
        ge=0,
        le=100,
        description="The minimum similarity percentage (0-100) to include. e.g., 80 means matches >= 80%."
    )
    source: bool

class MatchResponse(BaseModel):
    IsSuccess: bool = True
    Message: str = "SUCCESS"
    ErrorMessage: Optional[str] = None
    MatchedIdeasCount: int = 0
    MatchedIdeas: List[MatchedIdea]

# ==============================================================================
# 3. LANGGRAPH AGENT STATE AND CONFIGURATION
# ==============================================================================

class AgentState(TypedDict):
    """State for the idea similarity agent"""
    new_idea: IdeaItem
    source: bool
    query_embedding: Optional[List[float]]
    vector_candidates: List[Dict[str, Any]]
    similarity_threshold: int
    top_k: int
    final_matches: List[MatchedIdea]
    error_message: Optional[str]
    current_step: str

# ==============================================================================
# 4. AZURE OPENAI INITIALIZATION
# ==============================================================================

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint="https://bgsw-openaiservice-voiceseatbooking-p-eus-001.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
    api_key="2u2cSvJIlkFgj6BKsabPTVeIS4zcFlCu49yk2JxzrmUkIDTycp9qJQQJ99BHACYeBjFXJ3w3AAABACOGoOsh",
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
    temperature=0
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
vector_store = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ==============================================================================
# 5. LANGGRAPH AGENT NODES
# ==============================================================================

# --- Nodes are unchanged ---

def create_embedding_node(state: AgentState) -> AgentState:
    try:
        print(f"Creating embedding for idea: {state['new_idea'].Title}")
        text_to_embed = f"{state['new_idea'].Title}. {state['new_idea'].Description}"
        embedding = embedding_model.embed_query(text_to_embed)
        state["query_embedding"] = embedding
        state["current_step"] = "embedding_created"
        return state
    except Exception as e:
        print(f"Error in create_embedding_node: {str(e)}")
        state["error_message"] = f"Failed to create embedding: {str(e)}"
        state["current_step"] = "error"
        return state
def vector_search_node(state: AgentState) -> AgentState:
    try:
        print("Performing vector similarity search with dynamic business logic filters")
        if not state["query_embedding"]:
            state["error_message"] = "No query embedding available"
            state["current_step"] = "error"
            return state

        source = state['source']
        is_child_idea_of_new = state['new_idea'].IsChildIdea
        
        conditions = []
        print(f"Building filter for source={source}, new_idea.is_child={is_child_idea_of_new}")

        # Business Rule 1: source=false - Include ALL ideas with ALL statuses, including child ideas
        if source is False:
            print("Rule Applied: source=false, including ALL ideas with ALL statuses and ALL child ideas")
            # No filters needed - include everything
            
        # Business Rule 2: source=true - Only approved ideas (status 4+) with IsChildIdea=0, exclude child ideas
        elif source is True:
            # Only include ideas with status >= 4 (approved)
            conditions.append({"IdeaStatus": {"$gte": 4}})
            print("Rule Applied: source=true, including only approved ideas (status >= 4)")
            
            # Exclude child ideas (only include parent ideas where IsChildIdea=0)
            conditions.append({"IsChildIdea": {"$eq": 0}})  # Changed from False to 0 for consistency
            print("Rule Applied: source=true, excluding all child ideas (IsChildIdea must be 0)")

        # Build the final filter
        metadata_filter = None
        if len(conditions) > 1:
            metadata_filter = {"$and": conditions}
        elif len(conditions) == 1:
            metadata_filter = conditions[0]
        else:
            print("No specific filter conditions. Including all ideas.")

        print(f"Final metadata filter for ChromaDB: {metadata_filter}")

        fetch_count = min(CANDIDATE_POOL_SIZE, state["top_k"] * 3)

        search_kwargs = {"k": fetch_count}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        results_with_scores = vector_store.similarity_search_by_vector_with_relevance_scores(
            state["query_embedding"],
            **search_kwargs
        )

        candidates = []
        for doc, score in results_with_scores:
            try:
                similarity_percent = int((1 - score) * 100)
                metadata = doc.metadata
                candidates.append({
                    'idea_id': metadata.get('id', 0),
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'initial_similarity': similarity_percent,
                    'content': doc.page_content,
                    'idea_status': metadata.get('IdeaStatus', 'Unknown'),
                    'is_child_idea': metadata.get('IsChildIdea', 'Unknown')
                })
            except Exception as e:
                print(f"Error processing search result: {str(e)}")
                continue

        state["vector_candidates"] = candidates
        state["current_step"] = "vector_search_completed"
        print(f"Found {len(candidates)} vector candidates after dynamic filtering")
        
        # Debug: Print summary of what was included
        if candidates:
            statuses = [c.get('idea_status', 'Unknown') for c in candidates]
            child_flags = [c.get('is_child_idea', 'Unknown') for c in candidates]
            print(f"Debug - Included statuses: {set(statuses)}")
            print(f"Debug - Child idea flags: {set(child_flags)}")
        
        return state
        
    except Exception as e:
        print(f"Error in vector_search_node: {str(e)}")
        print(traceback.format_exc())
        state["error_message"] = f"Vector search failed: {str(e)}"
        state["current_step"] = "error"
        return state


def llm_reranking_node(state: AgentState) -> AgentState:
    try:
        print("Performing LLM-based semantic reranking")
        candidates = state["vector_candidates"]
        new_idea = state["new_idea"]

        if not candidates:
            state["final_matches"] = []
            state["current_step"] = "reranking_completed"
            return state

        system_prompt = """You are an expert at analyzing conceptual similarity between ideas.
        Your task is to score the similarity between a new idea and a list of existing ideas on a scale of 0-100%.
        Consider these aspects: conceptual overlap, problem domain, solution approach, and target audience.
        For each comparison, provide a similarity score (0-100%) and a brief explanation."""

        human_prompt = f"NEW IDEA TO MATCH:\nTitle: {new_idea.Title}\nDescription: {new_idea.Description}\n\nEXISTING IDEAS TO COMPARE:\n"
        for i, candidate in enumerate(candidates[:15]):
            human_prompt += f"\n{i+1}. ID: {candidate['idea_id']}\n   Title: {candidate['title']}\n   Description: {candidate['description']}"

        human_prompt += """\n\nPlease analyze each existing idea and respond with a JSON array of objects. Each object must contain:
        - "idea_id": the ID number (as an integer)
        - "similarity_score": your assessed similarity score (as an integer from 0-100)
        - "reasoning": your brief explanation for the score"""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = llm.invoke(messages)
        response_text = response.content

        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            llm_results = json.loads(json_match.group(0)) if json_match else []
            final_matches = [
                MatchedIdea(
                    IdeaId=result['idea_id'],
                    SimilarityPercent=result['similarity_score'],
                    ReasonForSimilarity=result.get('reasoning', '')
                ) for result in llm_results
            ]
            final_matches.sort(key=lambda x: x.SimilarityPercent, reverse=True)
            state["final_matches"] = final_matches
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Failed to parse LLM response as JSON: {str(e)}")
            fallback_matches = [
                MatchedIdea(
                    IdeaId=c['idea_id'],
                    SimilarityPercent=c['initial_similarity'],
                    ReasonForSimilarity="Vector similarity (LLM parsing failed)"
                ) for c in candidates
            ]
            fallback_matches.sort(key=lambda x: x.SimilarityPercent, reverse=True)
            state["final_matches"] = fallback_matches[:state["top_k"]]

        state["current_step"] = "reranking_completed"
        print(f"LLM reranking completed. Found {len(state['final_matches'])} potential matches.")
        return state
    except Exception as e:
        print(f"Error in llm_reranking_node: {str(e)}")
        fallback_matches = [
            MatchedIdea(
                IdeaId=c['idea_id'],
                SimilarityPercent=c['initial_similarity'],
                ReasonForSimilarity="Fallback due to agent error"
            ) for c in state["vector_candidates"]
        ]
        fallback_matches.sort(key=lambda x: x.SimilarityPercent, reverse=True)
        state["final_matches"] = fallback_matches[:state["top_k"]]
        state["current_step"] = "reranking_completed_with_fallback"
        return state

# ==============================================================================
# 6. LANGGRAPH WORKFLOW DEFINITION
# ==============================================================================

def create_idea_similarity_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("create_embedding", create_embedding_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("llm_reranking", llm_reranking_node)

    workflow.set_entry_point("create_embedding")
    workflow.add_edge("create_embedding", "vector_search")
    workflow.add_edge("vector_search", "llm_reranking")
    workflow.add_edge("llm_reranking", END)

    app = workflow.compile()
    return app

similarity_agent = create_idea_similarity_agent()

# ==============================================================================
# 7. FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================

app = FastAPI(
    title="AI Agent-Based Idea Similarity API",
    version="3.4", # --- MODIFIED: Incremented version number
    description="An intelligent agent for finding conceptually similar ideas, with dynamic business logic filtering."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODIFIED: Updated add_ideas to perform ADD-ONLY logic ---
@app.post("/addIdeas", response_model=AddIdeasResponse)
async def add_ideas(request: AddIdeasRequest):
    """
    Adds new ideas to the vector store.
    If an idea with the same 'Id' already exists, it will be SKIPPED.
    Use the PUT /updateIdeasMetadata endpoint to update existing ideas.
    """
    try:
        ideas_to_add = []
        ids_to_add = []
        ids_skipped = []
        
        # Check for existing ideas before attempting to add
        all_ids_in_request = [str(idea.Id) for idea in request.Ideas]
        existing_docs = vector_store.get(ids=all_ids_in_request)
        existing_ids = set(map(int, existing_docs['ids']))

        for idea in request.Ideas:
            if idea.Id in existing_ids:
                ids_skipped.append(idea.Id)
            else:
                doc = Document(
                    page_content=f"{idea.Title}. {idea.Description}",
                    metadata={
                        "id": idea.Id,
                        "title": idea.Title,
                        "description": idea.Description,
                        "IdeaStatus": idea.IdeaStatus,
                        "IsChildIdea": idea.IsChildIdea
                    }
                )
                ideas_to_add.append(doc)
                ids_to_add.append(str(idea.Id))
        
        if ideas_to_add:
            vector_store.add_documents(documents=ideas_to_add, ids=ids_to_add)
            print(f"Added {len(ideas_to_add)} new documents with IDs: {ids_to_add}")
        
        message = f"Successfully processed request. Added {len(ids_to_add)} new ideas. Skipped {len(ids_skipped)} existing ideas."
        
        return AddIdeasResponse(
            Message=message,
            IdsAdded=[int(id_str) for id_str in ids_to_add],
            IdsSkipped=ids_skipped
        )
    except Exception as e:
        print(f"Error adding ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add ideas: {str(e)}")

# --- NEW: Endpoint to update metadata for existing ideas ---
@app.put("/updateIdeasMetadata", response_model=UpdateIdeasResponse)
async def update_ideas_metadata(request: UpdateIdeasRequest):
    """
    Updates the content and metadata for existing ideas in the vector store.
    The embedding will be re-calculated if the Title or Description changes.
    If an idea 'Id' is not found, it will be reported and skipped.
    """
    try:
        ids_updated = []
        ids_not_found = []

        for idea in request.Ideas:
            # First, check if the document exists. update_document doesn't provide this feedback.
            existing_doc = vector_store.get(ids=[str(idea.Id)])
            if not existing_doc['ids']:
                ids_not_found.append(idea.Id)
                continue

            # If it exists, create the updated document and update it
            updated_doc = Document(
                page_content=f"{idea.Title}. {idea.Description}",
                metadata={
                    "id": idea.Id,
                    "title": idea.Title,
                    "description": idea.Description,
                    "IdeaStatus": idea.IdeaStatus,
                    "IsChildIdea": idea.IsChildIdea
                }
            )
            # Langchain Chroma's `update_document` is a bit hidden.
            # The underlying client's `update` or `upsert` is better.
            # Here we use `add_documents` with existing IDs which acts as an upsert.
            # This is the most reliable method with the LangChain wrapper.
            vector_store.add_documents(documents=[updated_doc], ids=[str(idea.Id)])
            ids_updated.append(idea.Id)

        print(f"Updated {len(ids_updated)} documents. Not found: {len(ids_not_found)} documents.")
        
        message = f"Successfully processed request. Updated {len(ids_updated)} ideas. Could not find {len(ids_not_found)} ideas."
        return UpdateIdeasResponse(
            Message=message,
            IdsUpdated=ids_updated,
            IdsNotFound=ids_not_found
        )
    except Exception as e:
        print(f"Error updating ideas metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update ideas: {str(e)}")

@app.delete("/deleteIdeas", response_model=DeleteIdeasResponse)
async def delete_ideas(request: DeleteIdeasRequest):
    try:
        if not request.Ids:
            return DeleteIdeasResponse(Message="No IDs provided for deletion.", IdsDeleted=[])

        # Convert IDs to string for ChromaDB
        ids_to_delete = [str(id_val) for id_val in request.Ids]
        
        # This is the correct way to delete from Chroma
        vector_store.delete(ids=ids_to_delete)
        
        print(f"Deletion request processed for IDs: {request.Ids}")

        return DeleteIdeasResponse(
            Message=f"Successfully deleted {len(request.Ids)} ideas from the vector store.",
            IdsDeleted=request.Ids
        )
    except Exception as e:
        print(f"Error deleting ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete ideas: {str(e)}")

@app.post("/findMatches", response_model=MatchResponse)
async def match_ideas(request: MatchRequest):
    try:
        print(f"Matching request for idea: {request.NewIdea.Title}")
        print(f"Parameters - source: {request.source}, IsChildIdea: {request.NewIdea.IsChildIdea}, TopK: {request.TopK}, MinSimilarity: {request.MinSimilarity}%")

        initial_state = AgentState(
            new_idea=request.NewIdea,
            source=request.source,
            query_embedding=None,
            vector_candidates=[],
            similarity_threshold=request.MinSimilarity,
            top_k=request.TopK,
            final_matches=[],
            error_message=None,
            current_step="starting"
        )

        final_state = await asyncio.to_thread(similarity_agent.invoke, initial_state)

        if final_state.get("error_message"):
            return MatchResponse(
                IsSuccess=False,
                Message="FAILED",
                ErrorMessage=final_state["error_message"],
                MatchedIdeasCount=0,
                MatchedIdeas=[]
            )

        all_matches_from_agent = final_state.get("final_matches", [])

        filtered_matches = [
            match for match in all_matches_from_agent
            if match.SimilarityPercent >= request.MinSimilarity
        ]

        final_results = filtered_matches[:request.TopK]

        print(f"Agent returned {len(all_matches_from_agent)} potential matches.")
        print(f"Returning {len(final_results)} matches after filtering for similarity >= {request.MinSimilarity}% and applying TopK limit.")

        return MatchResponse(
            IsSuccess=True,
            Message="SUCCESS",
            MatchedIdeasCount=len(final_results),
            MatchedIdeas=final_results
        )

    except Exception as e:
        error_msg = f"Agent execution failed: {str(e)}"
        print(f"Error in match_ideas: {error_msg}")
        print(traceback.format_exc())
        return MatchResponse(
            IsSuccess=False,
            Message="FAILED",
            ErrorMessage=error_msg,
            MatchedIdeasCount=0,
            MatchedIdeas=[]
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "ready"}

@app.get("/")
async def root():
    return {
        "message": "AI Agent-Based Idea Similarity API",
        "version": "3.4", # --- MODIFIED: Incremented version
        "endpoints": {
            "add_new_ideas": "POST /addIdeas",
            "update_existing_ideas": "PUT /updateIdeasMetadata", # --- NEW
            "delete_ideas": "DELETE /deleteIdeas",
            "match_ideas": "POST /findMatches",
            "health": "GET /health"
        }
    }

    print("Starting AI Agent-Based Idea Similarity API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
