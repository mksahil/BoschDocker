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

# --- NEW: LangChain Azure AI Search Imports ---
from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.models import VectorizedQuery

# --- REMOVED: ChromaDB specific imports ---
# import chromadb


# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# --- Load Environment Variables ---
# NOTE: Make sure these are set in your App Service configuration
AZURE_AI_SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
AZURE_AI_SEARCH_API_KEY = os.environ["AZURE_AI_SEARCH_API_KEY"]
AZURE_AI_SEARCH_INDEX_NAME = os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", "ideas-index-v1")


# --- LangGraph Agent Configuration ---
CANDIDATE_POOL_SIZE = 50

# --- Business Logic Constants (unchanged) ---
EXCLUDED_STATUSES_S1_IC0 = [3, 5]
EXCLUDED_STATUSES_S0_IC1 = [5]


# ==============================================================================
# 2. API DATA MODELS (Pydantic) - Unchanged
# ==============================================================================
class IdeaItem(BaseModel):
    Id: int
    Title: str
    Description: str
    IdeaStatus: int
    IsChildIdea: bool

class IdeaUpdateItem(BaseModel):
    Id: int
    Title: Optional[str] = None
    Description: Optional[str] = None
    IdeaStatus: Optional[int] = None
    IsChildIdea: Optional[bool] = None

class MatchedIdea(BaseModel):
    IdeaId: int
    SimilarityPercent: int
    ReasonForSimilarity: Optional[str] = None

class AddIdeasRequest(BaseModel):
    Ideas: List[IdeaItem]

class AddIdeasResponse(BaseModel):
    Message: str
    IdsAdded: List[int]

# MODIFIED: AddIdeasResponse doesn't need IdsSkipped as AzureSearch upserts
class UpdateIdeasRequest(BaseModel):
    Ideas: List[IdeaUpdateItem]

class UpdateIdeasResponse(BaseModel):
    Message: str
    IdsUpdated: List[int]
    IdsNotFound: List[int] # Kept for consistency, but upsert might change need

class DeleteIdeasRequest(BaseModel):
    Ids: List[int]

class DeleteIdeasResponse(BaseModel):
    Message: str
    IdsDeleted: List[int]

class MatchRequest(BaseModel):
    NewIdea: IdeaItem
    TopK: int = 10
    MinSimilarity: int = Field(default=0, ge=0, le=100)
    source: bool

class MatchResponse(BaseModel):
    IsSuccess: bool = True
    Message: str = "SUCCESS"
    ErrorMessage: Optional[str] = None
    MatchedIdeasCount: int = 0
    MatchedIdeas: List[MatchedIdea]

# ==============================================================================
# 3. LANGGRAPH AGENT STATE AND CONFIGURATION (Unchanged)
# ==============================================================================
class AgentState(TypedDict):
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
# 4. AZURE OPENAI & AI SEARCH INITIALIZATION
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

# --- NEW: Initialize Azure AI Search Vector Store ---
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_AI_SEARCH_ENDPOINT,
    azure_search_key=AZURE_AI_SEARCH_API_KEY,
    index_name=AZURE_AI_SEARCH_INDEX_NAME,
    embedding_function=embedding_model.embed_query,
)
print(f"Connected to Azure AI Search index: '{AZURE_AI_SEARCH_INDEX_NAME}'")

# ==============================================================================
# 5. LANGGRAPH AGENT NODES
# ==============================================================================

# create_embedding_node is unchanged

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
        
# --- MODIFIED: vector_search_node to use Azure AI Search and OData filters ---
def vector_search_node(state: AgentState) -> AgentState:
    try:
        print("Performing vector similarity search with dynamic business logic filters on Azure AI Search")
        if not state["query_embedding"]:
            state["error_message"] = "No query embedding available"
            state["current_step"] = "error"
            return state

        source = state['source']
        
        # --- NEW: Build OData filter string for Azure AI Search ---
        odata_filter = ""
        print(f"Building filter for source={source}")

        if source is True:
            # Rule: source=true, including only approved ideas (status >= 4) AND parent ideas
            # OData syntax: "(IdeaStatus ge 4) and (IsChildIdea eq false)"
            odata_filter = "(IdeaStatus ge 4) and (IsChildIdea eq false)"
            print(f"Rule Applied: source=true. OData filter: {odata_filter}")
        else: # source is False
            print("Rule Applied: source=false, including ALL ideas. No filter applied.")

        fetch_count = min(CANDIDATE_POOL_SIZE, state["top_k"] * 3)
        
        # Azure Search LangChain integration uses different search kwargs
        search_kwargs = {"k": fetch_count}
        if odata_filter:
            # We perform a pure vector search first, then Azure applies the filter.
            # This is called "pre-filtering".
            search_kwargs["filters"] = odata_filter

        print(f"Azure AI Search parameters: {search_kwargs}")
        
        # Using the standard similarity search method
        results_with_scores = vector_store.similarity_search_with_relevance_scores(
            query=f"{state['new_idea'].Title}. {state['new_idea'].Description}", # Pass text query, it will be embedded
            **search_kwargs
        )

        print(f"Raw Azure AI Search results count: {len(results_with_scores)}")

        candidates = []
        for doc, score in results_with_scores:
            try:
                # Score is cosine similarity, so we can convert it directly
                similarity_percent = int(score * 100)
                metadata = doc.metadata
                print(f"Result - ID: {metadata.get('id')}, Status: {metadata.get('IdeaStatus')}, IsChild: {metadata.get('IsChildIdea')}, Score: {similarity_percent}%")
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
        return state
        
    except Exception as e:
        print(f"Error in vector_search_node: {str(e)}")
        print(traceback.format_exc())
        state["error_message"] = f"Vector search failed: {str(e)}"
        state["current_step"] = "error"
        return state

# llm_reranking_node is unchanged
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
# 6. LANGGRAPH WORKFLOW DEFINITION (Unchanged)
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
    version="4.0-AzureAISearch",
    description="An intelligent agent using Azure AI Search for persistent, scalable idea matching."
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- MODIFIED: addIdeas endpoint simplified for Azure Search ---
@app.post("/addIdeas", response_model=AddIdeasResponse)
async def add_ideas(request: AddIdeasRequest):
    """
    Adds or updates ideas in the Azure AI Search index.
    Azure Search performs an "upsert" by default, so if an Id exists, it will be overwritten.
    """
    try:
        docs_to_add = []
        ids_added = []
        for idea in request.Ideas:
            doc = Document(
                page_content=f"{idea.Title}. {idea.Description}",
                metadata={
                    "id": str(idea.Id), # The key field must be a string
                    "title": idea.Title,
                    "description": idea.Description,
                    "IdeaStatus": idea.IdeaStatus,
                    "IsChildIdea": idea.IsChildIdea
                }
            )
            docs_to_add.append(doc)
            ids_added.append(idea.Id)
        
        if docs_to_add:
            # add_documents with AzureSearch performs an upsert
            vector_store.add_documents(documents=docs_to_add)
            print(f"Upserted {len(docs_to_add)} documents with IDs: {ids_added}")
        
        return AddIdeasResponse(
            Message=f"Successfully added or updated {len(ids_added)} ideas.",
            IdsAdded=ids_added,
        )
    except Exception as e:
        print(f"Error adding ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add ideas: {str(e)}")

# --- MODIFIED: updateIdeasMetadata is now redundant but kept for API compatibility ---
# It can be simplified to just call the same logic as addIdeas.
@app.put("/updateIdeasMetadata", response_model=UpdateIdeasResponse)
async def update_ideas_metadata(request: UpdateIdeasRequest):
    """
    Updates metadata for existing ideas. Since Azure Search upserts, this
    is now functionally similar to adding ideas with the same ID.
    This endpoint is maintained for API compatibility.
    """
    try:
        # NOTE: Azure Search doesn't support partial updates via the LangChain API easily.
        # The easiest path is to treat this as a full "add/overwrite" operation.
        # This requires the client to know the full state of the object.
        # For a true partial update, you would need to fetch, merge, and then update.
        # But since the `IdeaUpdateItem` can be partial, we cannot do a simple upsert.
        # Let's keep your original logic but adapt it for Azure Search.

        ids_updated = []
        ids_not_found = []
        docs_to_update = []

        # We cannot efficiently get documents by ID with the current LangChain wrapper.
        # For this to work, we'll just upsert. If the user provides partial data,
        # ONLY that data will be in the new document. This is a behavioral change.
        # The simplest/best approach is to merge this endpoint's functionality into /addIdeas
        # and tell users to send the full IdeaItem object for updates.

        # Simplified approach: Treat update as a full replacement.
        # This requires client to send all fields for the update.
        full_ideas_to_update = []
        for idea_update in request.Ideas:
            # This is a simplification. A real implementation might fetch the doc first.
            if idea_update.Title is None or idea_update.Description is None or idea_update.IdeaStatus is None or idea_update.IsChildIdea is None:
                 raise HTTPException(status_code=400, detail=f"Update for ID {idea_update.Id} is partial. Please provide all fields (Title, Description, IdeaStatus, IsChildIdea) for an update operation with this backend.")

            full_ideas_to_update.append(IdeaItem(**idea_update.dict()))
            ids_updated.append(idea_update.Id)

        add_request = AddIdeasRequest(Ideas=full_ideas_to_update)
        await add_ideas(add_request)

        return UpdateIdeasResponse(
            Message=f"Successfully processed update request for {len(ids_updated)} ideas.",
            IdsUpdated=ids_updated,
            IdsNotFound=[] # Since it's an upsert, nothing is "not found"
        )
    except HTTPException as he:
        raise he # Re-raise client errors
    except Exception as e:
        print(f"Error updating ideas metadata: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to update ideas: {str(e)}")


@app.delete("/deleteIdeas", response_model=DeleteIdeasResponse)
async def delete_ideas(request: DeleteIdeasRequest):
    try:
        if not request.Ids:
            return DeleteIdeasResponse(Message="No IDs provided for deletion.", IdsDeleted=[])
        
        # Azure Search requires the key to be a string
        ids_to_delete = [str(id_val) for id_val in request.Ids]
        vector_store.delete(ids=ids_to_delete)
        print(f"Deletion request processed for IDs: {request.Ids}")

        return DeleteIdeasResponse(
            Message=f"Successfully deleted {len(request.Ids)} ideas from the vector store.",
            IdsDeleted=request.Ids
        )
    except Exception as e:
        print(f"Error deleting ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete ideas: {str(e)}")

# findMatches and other endpoints remain the same...
@app.post("/findMatches", response_model=MatchResponse)
async def match_ideas(request: MatchRequest):
    try:
        print(f"Matching request for idea: {request.NewIdea.Title}")
        print(f"Parameters - source: {request.source}, IsChildIdea: {request.NewIdea.IsChildIdea}, TopK: {request.TopK}, MinSimilarity: {request.MinSimilarity}%")
        initial_state = AgentState(
            new_idea=request.NewIdea, source=request.source, query_embedding=None,
            vector_candidates=[], similarity_threshold=request.MinSimilarity, top_k=request.TopK,
            final_matches=[], error_message=None, current_step="starting"
        )
        final_state = await asyncio.to_thread(similarity_agent.invoke, initial_state)
        if final_state.get("error_message"):
            return MatchResponse(
                IsSuccess=False, Message="FAILED", ErrorMessage=final_state["error_message"],
                MatchedIdeasCount=0, MatchedIdeas=[]
            )
        all_matches_from_agent = final_state.get("final_matches", [])
        filtered_matches = [m for m in all_matches_from_agent if m.SimilarityPercent >= request.MinSimilarity]
        final_results = filtered_matches[:request.TopK]
        print(f"Agent returned {len(all_matches_from_agent)} matches. Returning {len(final_results)} after filtering.")
        return MatchResponse(IsSuccess=True, Message="SUCCESS", MatchedIdeasCount=len(final_results), MatchedIdeas=final_results)
    except Exception as e:
        error_msg = f"Agent execution failed: {str(e)}"
        print(f"Error in match_ideas: {error_msg}\n{traceback.format_exc()}")
        return MatchResponse(IsSuccess=False, Message="FAILED", ErrorMessage=error_msg, MatchedIdeasCount=0, MatchedIdeas=[])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "ready", "vector_store": "Azure AI Search"}

@app.get("/")
async def root():
    return { "message": "AI Agent-Based Idea Similarity API", "version": "4.0-AzureAISearch" }
