# --- 1. Imports and Setup (Merged from both files) ---
import httpx
import re
import os
import json
import operator
import asyncio
import traceback
import logging
import pytz
import pyodbc
import chromadb
import contextvars
from time import time
from functools import wraps
from datetime import datetime
from typing import List, Dict, Optional, Any, TypedDict, Annotated

# Pydantic and FastAPI
from pydantic import BaseModel, Field
from fastapi import Depends, FastAPI, Body, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

# LangChain and LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Database and other utilities
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# --- 2. Logging and Global Configuration ---

# Logging Setup
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- SEAT BOOKING Configuration ---
connection_string_windows = (
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=rbeidashboarddbdev.database.windows.net;'
    r'DATABASE=RBEI-Dashboard-Dev;'
    r'UID=rbeicloudadmin;'
    r'PWD=Rbeisql@123;'
)

# --- IDEA SIMILARITY Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bgsw-openaiservice-voiceseatbooking-p-eus-001.openai.azure.com/...")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "2u2cSvJIlkFgj6BKsabPTVeIS4zcFlCu49yk2JxzrmUkIDTycp9qJQQJ99BHACYeBjFXJ3w3AAABACOGoOsh")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "GPTbgsw-openAIservice-Voiceseatbooking4")

# Seat Booking Models
class ChatRequest(BaseModel):
    EmployeeQueryMessage: str

class ChatResponse(BaseModel):
    item: str
    status: str

class BookingResult(BaseModel):
    success: bool
    message: str
    booking_details: Optional[Dict[str, Any]] = None

class DayBookingStatus(BaseModel):
    date: datetime
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    workspace_id: Optional[int] = None

class SeatAvailabilityResult(BaseModel):
    is_available: bool
    status_name: str
    workspace_id: Optional[int] = None

class CancelResult(BaseModel):
    success: bool
    message: str
    allocation_id: int

class BookingHistoryItem(BaseModel):
    seat: str
    floor: str
    building: str
    bookType: int
    unitId: int
    status: int
    isAvailable: bool
    allocationID: int
    fromDate: str


# --- 4. Global Variables & Context ---

# Context variable for dynamically setting employee_id per request (Seat Booking)
_current_employee_id = contextvars.ContextVar("employee_id", default=None)

# In-memory user data store (for associate info)
user_data_store: Dict[str, Dict[str, Any]] = {}

# In-memory conversation state store (for LangGraph messages history)
conversation_state_store: Dict[str, Any] = {}

# --- 5. FastAPI App Initialization ---
app = FastAPI(title="Bosch Combined AI Services API", version="1.0")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Caching for Idea Similarity
FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")

# Security scheme for token-based auth
security = HTTPBearer()

# --- 6. Global Object Initialization (Models, Vector Stores, etc.) ---

# LLM for Seat Booking
llm_seat_booking = AzureChatOpenAI(
                    azure_deployment="GPTbgsw-openAIservice-Voiceseatbooking4",
                    api_key="2u2cSvJIlkFgj6BKsabPTVeIS4zcFlCu49yk2JxzrmUkIDTycp9qJQQJ99BHACYeBjFXJ3w3AAABACOGoOsh",
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    azure_endpoint="https://bgsw-openaiservice-voiceseatbooking-p-eus-001.openai.azure.com/openai/deployments/bgsw-openAIservice-Voiceseatbooking/chat/completions?api-version=2025-01-01-preview",
                    temperature=0)


# --- 7. Seat Booking Service: Helper Functions ---

def get_db_connection():
    """Establishes a connection to the MSSQL database using SQLAlchemy."""
    try:
        connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string_windows})
        engine = create_engine(connection_url)
        connection = engine.connect()
        return connection
    except (pyodbc.Error, SQLAlchemyError) as e:
        print(f"Error connecting to the database: {e}")
        logger.error(f"Error connecting to the database: {e}")
        return None

async def log_conversation_to_db(session_id: str, message: str, sender: str,employee_id:int):
    """
    Asynchronously logs a conversation message to the MSSQL database using SQLAlchemy.
    """
    sql = text("INSERT INTO ChatConversations (SessionID, MessageText, Sender,employee_id) VALUES (:session_id, :message, :sender,:employee_id)")
    conn = None
    trans = None
    try:
        conn = get_db_connection()
        if conn:
            trans = await asyncio.to_thread(conn.begin)
            await asyncio.to_thread(
                conn.execute,
                sql,
                {"session_id": session_id, "message": message, "sender": sender,"employee_id":employee_id}
            )
            await asyncio.to_thread(trans.commit)
            print(f"Successfully logged message for SessionID: {session_id} and employee_id {employee_id}" )
            logger.info(f"Successfully logged message for SessionID: {session_id} and employee_id {employee_id}" )
    except (pyodbc.Error, SQLAlchemyError) as e:
        print(f"Database Error: Could not log message for SessionID {session_id}. Error: {e}")
        logger.info(f"Database Error: Could not log message for SessionID {session_id}. Error: {e}")
        if trans:
            try:
                await asyncio.to_thread(trans.rollback)
            except Exception as rollback_error:
                print(f"Error during rollback: {rollback_error}")
                logger.error(f"Error during rollback: {rollback_error}")
    finally:
        if conn:
            conn.close()

import requests
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
def validate_token(Authorization: str, client_id: str = "CD3054C5-6D98-47E9-BF73-43F26E8ED476") -> dict:
    url = "https://boschassociatearena.com/api/Token/ValidateToken"
    headers = {
        "Authorization": f"Bearer {Authorization}",
        "clientID": client_id
    }
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "IsSuccess": False,
            "Message": "Request failed",
            "ErrorMessage": str(e),
            "ResponseData": []
        }

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
def decrypt_text(text_to_decrypt: str, client_id: str = "CD3054C5-6D98-47E9-BF73-43F26E8ED476") -> str:
    url = "https://boschassociatearena.com/api/Token/DecryptClientData"
    params = {"TextToDecrypt": text_to_decrypt}
    headers = {"clientID": client_id}
    try:
        response = requests.get(url, params=params, headers=headers, verify=False)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

def parse_user_data(decrypted_data: str) -> dict:
    try:
        parts = decrypted_data.split(',')
        if len(parts) >= 3:
            return {
                "employee_id": parts[0].strip(),
                "employee_name": parts[1].strip(),
                "employee_code": parts[2].strip()
            }
        else:
            raise ValueError("Invalid user data format")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse user data: {str(e)}"
        )

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
async def get_new_access_token():
    """Fetches a new access token from the authentication server."""
    url = "https://flexibook.boschassociatearena.com/connect/token"
    payload = {
        'client_id': 'C851F411-2BFA-4578-A4EF-D420EC6CBB64',
        'client_secret': '90D18C4E-32E1-4DBF-B30F-CC91F47ADCF6',
        'grant_type': 'password', 'username': '11446688'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload, headers=headers)
            response.raise_for_status()
            token = response.json().get('access_token')
            print("Successfully retrieved Flexibook access token.")
            logger.info("Successfully retrieved Flexibook access token.")
            return token
    except Exception as e:
        print(f"Error getting new Flexibook access token: {str(e)}")
        return None

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
async def get_associate_info(access_token, employee_id_param: Optional[str] = None):
    """Gets associate information using their employee ID."""
    employee_id = employee_id_param or _current_employee_id.get()
    if not employee_id:
        raise ValueError("Employee ID not available in context or provided.")

    print(f"Fetching info for employee_id: {employee_id}")
    logger.info(f"Fetching info for employee_id: {employee_id}")

    try:
        employee_id_clean = int(str(employee_id).strip().replace('"', '').replace("'", ''))
    except ValueError:
        print(f"Invalid employee_id: {employee_id}")
        logger.error(f"Invalid employee_id: {employee_id}")
        return None

    url = f"https://flexibook.boschassociatearena.com/api/flexi/GetAssociate?searchValue={employee_id_clean}"
    headers = {'Authorization': f'Bearer {access_token}'}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, dict):
                 user_data_store[employee_id] = {
                     "collected_info": {
                         "employee_name": data.get('flexibleUserName'),
                         "employee_email": data.get('email'),
                         "associate_id_val": data.get('associateId')
                     },
                     "associate_api_data": data
                 }
            print(f"Successfully fetched associate info for {employee_id}.")
            logger.info(f"Successfully fetched associate info for {employee_id}.")
            return data
    except Exception as e:
        print(f"Error getting associate info: {str(e)}")
        logger.error(f"Error getting associate info: {str(e)}")
        return None

def extract_unit_id(associate_info, building_name, floor):
    print("----------------------------inside extract_unit_id------------------")
    print("--------------building_name----------",building_name)
    print("--------------floor----------",floor)
    if not associate_info or 'availableSeatList' not in associate_info:
        return None
    floor_word_to_number = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "ground": "0", "first": "1", "second": "2", "third": "3",
        "fourth": "4", "forth": "4", "fifth": "5"
    }
    normalized_floor = floor.lower().strip()
    for word, number in floor_word_to_number.items():
       normalized_floor = re.sub(rf'\b{word}\b', number, normalized_floor)
    normalized_floor = re.sub(r'\b(floor|fl|f)\b', '', normalized_floor, flags=re.IGNORECASE).strip()
    normalized_floor = re.sub(r'\b(no|number|#)\b', '', normalized_floor, flags=re.IGNORECASE).strip()
    normalized_floor = re.sub(r'(st|nd|rd|th)\b', '', normalized_floor).strip()

    print("normalized_floor:", normalized_floor)
    if normalized_floor in ['g', 'ground', '0']:
       possible_formats = ["G Floor", "G", "Ground Floor", "Ground", "0"]
    else:
      possible_formats = [
        f"Floor {normalized_floor}", f"{normalized_floor} Floor",
        f"F{normalized_floor}", f"{normalized_floor}"
      ]
    print("Looking for floor formats:", possible_formats)
    available_floors = [f['floor'] for f in associate_info.get('availableSeatList', [])]
    print("Available floors in API response:", available_floors)
    logger.info(f"Available floors in API response: {available_floors}")

    for floor_info in associate_info['availableSeatList']:
        floor_name = floor_info['floor'].lower()
        if any(format.lower() == floor_name for format in possible_formats):
            print(f"Found matching floor: {floor_info['floor']}")
            logger.info(f"Found matching floor: {floor_info['floor']}")
            print(":::::::::::::::::::::::::::::::floor info::::::::::::::::::::::::",floor_info)
            print("building_name in floor_info",building_name in floor_info)
            print("floor_info[building_name]",floor_info[building_name])
            if building_name in floor_info:
                value = floor_info[building_name]
                match = re.search(r"/space/flex/\d+/(\d+)\?", value)
                print("match:",match)
                if match: return match.group(1)
    for floor_info in associate_info['availableSeatList']:
        floor_digits = re.search(r'\d+', floor_info['floor'])
        if floor_digits and floor_digits.group(0) == normalized_floor:
            print(f"Found floor by number match: {floor_info['floor']}")
            print("line 274 building_name:", building_name)
            building_name= floor_info.get(building_name.lower(), None)
            print("line 276 building_name:", building_name)
            if(building_name):
              print("floor_info:", floor_info[building_name.lower()])
              print(" line 279 building_name:", building_name)
              if building_name.lower() in floor_info:
                   value = building_name
                   print("value:",value)
                   match = re.search(r"/space/flex/\d+/(\d+)\?", value)
                   if match: return match.group(1)
    return None

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
async def get_floor_booking_info(access_token: str, unit_id: str, date_to_check: datetime):
    """Gets booking information for a floor (unit) on a specific date."""
    print(f"Getting booking info for unit {unit_id} on {date_to_check.strftime('%Y-%m-%d')}")
    logger.info(f"Getting booking info for unit {unit_id} on {date_to_check.strftime('%Y-%m-%d')}")
    formatted_date_for_api = date_to_check.strftime("%Y%%2F%m%%2F%d")
    print("line 291 ",formatted_date_for_api)
    url = f"https://flexibook.boschassociatearena.com/api/Flexi/GetBookingForWeb4Days?unitId={unit_id}&typeID=5&dateCheck={formatted_date_for_api}"
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            print(":::::::::::::::::::::::::get_floor_booking_info response::::::::::::::::::::::::",response)
            response.raise_for_status()
            data = response.json()
            print("Successfully fetched floor booking info.")
            logger.info(f"Successfully fetched floor booking info for unit {unit_id} on {date_to_check.strftime('%Y-%m-%d')}")
            return data
    except Exception as e:
        print(f"line 399 Error getting booking info for unit {unit_id}: {str(e)}")
        logger.error(f"Error getting booking info for unit {unit_id}: {str(e)}")
        return None

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
async def book_seat(access_token, employee_id_param: Optional[str], workspace_id, booking_date: datetime, time_slot: str = 'full_day'):
    """Books a specific seat for an employee."""
    employee_id = employee_id_param or _current_employee_id.get()
    if not employee_id:
        raise ValueError("Employee ID not available in context or provided for booking.")

    logger.info(f"Attempting to book seat for employee {employee_id} on {booking_date.strftime('%Y-%m-%d')} at {time_slot}")
    logger.info(f"Workspace ID: {workspace_id}, Booking Date: {booking_date.strftime('%Y-%m-%d')}, Time Slot: {time_slot}")
    url = "https://flexibook.boschassociatearena.com/api/flexi/book4Days"

    bookType = 1 # Full Day
    if time_slot == 'first_half':
        bookType = 2
    elif time_slot == 'second_half':
        bookType = 3

    from_time = booking_date.replace(hour=8, minute=0, second=0, microsecond=0)
    to_time = booking_date.replace(hour=20, minute=0, second=0, microsecond=0)
    if time_slot == 'first_half':
        to_time = booking_date.replace(hour=12, minute=0)
    elif time_slot == 'second_half':
        from_time = booking_date.replace(hour=13, minute=0)

    from_date_str, to_date_str = from_time.isoformat(), to_time.isoformat()

    associate_info_data = user_data_store.get(employee_id, {}).get("collected_info", {})
    associate_name = associate_info_data.get("employee_name", "AI Bot User")
    associate_email = associate_info_data.get("employee_email", "")
    associate_id_val = associate_info_data.get("associate_id_val")

    if not associate_id_val:
        temp_associate_info = await get_associate_info(access_token, employee_id)
        if temp_associate_info:
            associate_id_val = user_data_store.get(employee_id, {}).get("collected_info", {}).get("associate_id_val")
        else:
             return BookingResult(success=False, message="Critical error: Could not retrieve associate ID.")

    payload = {
        "allocationMode": 5, "associateId": associate_id_val, "bookType": bookType,
        "Email": associate_email, "employeeNumber": employee_id, "from": "",
        "fromDate": from_date_str, "isMovingHere": False, "isValid": False, "modifiedBy": "AI bot",
        "remark": "Seat booked by LangGraph AI", "to": "", "toDate": to_date_str, "toName": associate_name,
        "workspaceId": int(workspace_id)
    }

    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            booking_response = response.json()
            print("booking_response:",booking_response)
            logger.info(f"Booking response for employee {employee_id}: {booking_response}")
            if booking_response.get('isValid', False):
                print("Booking successful via API.")
                logger.info(f"Booking successful for employee {employee_id} on {booking_date.strftime('%Y-%m-%d')} at {time_slot}")
                return BookingResult(
                    success=True, message="Seat booked successfully!",
                    booking_details={"workspace_id": workspace_id, "date": booking_date.strftime('%Y-%m-%d'), "time_slot": time_slot}
                )
            else:
                error_msg = booking_response.get('exceptionMessage', 'Unknown error from booking API')
                print(f"Booking failed: {error_msg}")
                logger.error(f"Booking failed for employee {employee_id}: {error_msg}")
                return BookingResult(success=False, message=f"Failed to book seat: {error_msg}")
    except Exception as e:
        print(f"Exception during booking: {str(e)}")
        logger.error(f"Exception during booking for employee {employee_id}: {str(e)}")
        return BookingResult(success=False, message=f"An exception occurred while booking: {str(e)}")

def check_seat_availability(floor_booking_data: Dict[str, Any], seat_number: str) -> SeatAvailabilityResult:
    """Checks if a specific seat is available from floor booking data."""
    if not floor_booking_data or "asscocicateInfoList" not in floor_booking_data:
        return SeatAvailabilityResult(is_available=False, status_name="Data not found")
    print(" line 486 :::::::::::::::::check seat availability called::::::::::::::::::::::::::::")
    logger.info(f"Checking seat availability for seat number: {seat_number}")
    workspaceName=""
    for seat in floor_booking_data["asscocicateInfoList"]:
        workspaceName=seat.get("worspaceName", "")
        print("workspace name:",workspaceName)
        logger.info(f"Checking seat: {workspaceName} against requested seat number: {seat_number}")
        print("seat number",seat_number)
        print(" Is worksapce name is null",workspaceName=='null')
        print(" Is worksapce name is None",workspaceName==None)
        if workspaceName!='null' and  workspaceName!=None:
            if workspaceName.lower()== seat_number.lower():
             print("  seat name matched")
             logger.info(f"Seat {workspaceName} matched with requested seat number: {seat_number}")
             if seat.get("workspaceStatusId") == 1: # 1 means available
                logger.info(f"Seat {workspaceName} is available.")
                return SeatAvailabilityResult(
                    is_available=True,
                    status_name="Available",
                    workspace_id=seat.get("workspaceId")
                )
             else: # Other statuses: booked, blocked etc.
                logger.info(f"Seat {workspaceName} is not available. Status ID: {seat.get('workspaceStatusId')}")
                return SeatAvailabilityResult(is_available=False, status_name="Not Available")
        else:
            print("workspace name is ethire null or none")
    print("workspaceNames",workspaceName)
    return SeatAvailabilityResult(is_available=False, status_name="Seat not found")

def format_booking_results_table(results: List[DayBookingStatus]) -> str:
    """Formats booking results into a markdown table."""
    if not results:
        return "No booking attempts were made."

    header = "| Date       | Status    | Details                                  |"
    separator = "|------------|-----------|------------------------------------------|"
    rows = [header, separator]

    for res in results:
        status_icon = "✅ Success" if res.success else "❌ Failed"
        rows.append(f"| {res.date.strftime('%Y-%m-%d')} | {status_icon} | {res.message} |")

    return "\n".join(rows)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
async def get_booking_history(access_token: str, employee_id_param: Optional[str]) -> List[BookingHistoryItem]:
    """Retrieves the booking history for an employee."""
    employee_id = employee_id_param or _current_employee_id.get()
    if not employee_id:
        raise ValueError("Employee ID not available in context or provided for history.")

    url = "https://flexibook.boschassociatearena.com/api/flexi/GetBookingHistory"
    payload = {"keySearch": employee_id, "pageIndex": 1, "pageSize": 50}
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            history_data = response.json()
            logger.info(f"Successfully fetched booking history for employee {employee_id}.")
            return [BookingHistoryItem(**item) for item in history_data]
    except Exception as e:
        print(f"Error getting booking history for employee {employee_id}: {str(e)}")
        logger.error(f"Error getting booking history for employee {employee_id}: {str(e)}")
        return []

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
async def cancel_booking_api(access_token: str, allocation_id: int) -> CancelResult:
    """The core API call to cancel a booking."""
    url = f"https://flexibook.boschassociatearena.com/api/flexi/Cancel?allocationId={allocation_id}"
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            try:
                cancel_response_data = response.json()
                is_successful = cancel_response_data.get('isValid', False) or \
                                "cancelled" in cancel_response_data.get('exceptionMessage', "").lower() or \
                                "success" in cancel_response_data.get('message', "").lower()

                if is_successful or (response.status_code == 200 and not cancel_response_data):
                     return CancelResult(success=True, message=cancel_response_data.get('exceptionMessage', "Booking successfully cancelled."), allocation_id=allocation_id)
                else:
                    return CancelResult(success=False, message=cancel_response_data.get('exceptionMessage', f"Cancellation response unclear: {response.text[:100]}"), allocation_id=allocation_id)
            except json.JSONDecodeError:
                if 200 <= response.status_code < 300:
                    return CancelResult(success=True, message="Booking successfully cancelled.", allocation_id=allocation_id)
                else:
                    return CancelResult(success=False, message=f"Cancellation failed with status {response.status_code}. Response: {response.text[:100]}", allocation_id=allocation_id)
    except httpx.HTTPStatusError as e:
        try:
            error_details = e.response.json()
            message = error_details.get("exceptionMessage", error_details.get("Message", str(e)))
        except json.JSONDecodeError: message = e.response.text or str(e)
        return CancelResult(success=False, message=f"Failed to cancel booking: {message[:100]}", allocation_id=allocation_id)
    except Exception as e:
        return CancelResult(success=False, message=f"Error cancelling booking: {str(e)}", allocation_id=allocation_id)

def seats_match(seat1: str, seat2: str) -> bool:
    """Compares two seat numbers for a match, ignoring case and whitespace."""
    return re.sub(r'\s*-\s*', '-', seat1.strip().upper()) == re.sub(r'\s*-\s*', '-', seat2.strip().upper())

from collections import defaultdict
async def get_Zonal_info_list(access_token):
    """Gets zonal booking information."""
    url = f"https://flexibook.boschassociatearena.com/api/flexi/GetListZonalBooking"
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data
    except Exception as e:
        print(f"Error getting zonal info: {str(e)}")
        return None

def preprocess_zonals(zonal_list):
    seat_to_zonals = defaultdict(list)
    for z in zonal_list:
        try:
            from_date = datetime.fromisoformat(z["blockFromDate"])
            to_date = datetime.fromisoformat(z["blockToDate"])
        except:
            continue
        emp_list = set(e.strip() for e in z["listEmployeeNumber"].split(",") if e.strip())
        seat_list = set(s.strip() for s in z["listOfSeat"].split(",") if s.strip())
        for seat_id in seat_list:
            seat_to_zonals[seat_id].append({
                "emp_set": emp_list,
                "from_date": from_date,
                "to_date": to_date
            })
    return seat_to_zonals

def get_filtered_workspace_list(book_date, current_employee_number, get_booking_for_web_list, zonal_list):
    if isinstance(book_date, str):
        book_date = datetime.strptime(book_date, "%Y-%m-%d")
    seat_to_zonals = preprocess_zonals(zonal_list)

    result = []
    for seat in get_booking_for_web_list["asscocicateInfoList"]:
        workspace_id = str(seat.get("workspaceId"))
        worspace_name = seat.get("worspaceName")
        is_flexible = seat.get("isFlexible")
        is_valid = seat.get("isValid")
        work_space_status_id = seat.get("workspaceStatusId")
        you_cannot_book_this_zonnal = False
        zonals = seat_to_zonals.get(workspace_id, [])
        for z in zonals:
            if z["from_date"] <= book_date <= z["to_date"]:
                if str(current_employee_number) not in z["emp_set"]:
                    you_cannot_book_this_zonnal = True
                    break
        result.append({
            "workspaceId": int(workspace_id),
            "worspaceName": worspace_name,
            "isFlexible": is_flexible,
            "isValid": is_valid,
            "youCannotBookThisZonnal": you_cannot_book_this_zonnal,
            "workspaceStatusId": work_space_status_id
        })
    return result

async def _book_single_day(date_obj: datetime, unit_id: str, seat_no: str, timeslot: str, access_token: str) -> DayBookingStatus:
    """Helper coroutine to check and book one day."""
    employee_id = _current_employee_id.get()

    floor_booking_data = await get_floor_booking_info(access_token, unit_id, date_obj)
    if not floor_booking_data:
        return DayBookingStatus(date=date_obj, success=False, message="Failed to get availability for this date.")

    seat_availability = check_seat_availability(floor_booking_data, seat_no)
    if not seat_availability.is_available:
        return DayBookingStatus(date=date_obj, success=False, message=f"Seat not available ({seat_availability.status_name}).")

    booking_api_result = await book_seat(access_token, employee_id, seat_availability.workspace_id, date_obj, timeslot)
    return DayBookingStatus(date=date_obj, success=booking_api_result.success, message=booking_api_result.message)

def dict_to_readable_string(location_dict: Dict[str, str]) -> str:
    """Helper function to convert location dictionary to readable string format."""
    return ", ".join([f"{code}: {name}" for code, name in location_dict.items()])

def get_allowed_locations_internal(associate_info: Dict[str, Any]) -> Dict[str, str]:
    if not associate_info:
        return {}
    
    location_info = {
        'office_location_code': associate_info.get('officeLocationCode', '').upper(),
        'office_location_name': associate_info.get('officeLocationName', '').lower(),
        'location_name': associate_info.get('locationName', '').lower()
    }
    
    return location_info


# --- 9. LangGraph Agent for Seat Booking ---

# Agent Tools
@tool
async def get_allowed_buildings() -> dict:
    """
    Retrieves a list of all buildings the associate is authorized to book a seat in.
    This tool orchestrates multiple API calls to get the result.
    """
    print("Tool 'get_allowed_buildings' called.")
    access_token = await get_new_access_token()
    if not access_token:
        return {"error": "Failed to get authentication token."}

    employee_id = _current_employee_id.get()
    if not employee_id:
        return {"error": "Employee ID not found in context. Cannot retrieve buildings."}

    associate_info = await get_associate_info(access_token, employee_id)
    if not associate_info:
        return {"error": f"Failed to get associate information for employee {employee_id}."}

    if 'availableSeatList' not in associate_info:
        return {"error": "No available seat list found in associate information."}

    allowed_buildings = set()
    for floor_info in associate_info.get('availableSeatList', []):
        for key in floor_info.keys():
            if key.lower() not in ['floor', 'iscob']:
                allowed_buildings.add(key)
    if not allowed_buildings:
        return {"message": "No allowed buildings found for the associate."}
    return {"allowed_buildings": sorted(list(allowed_buildings))}

@tool
async def get_allowed_locations_tool() -> dict:
    """
    Retrieves the allowed office locations for the associate.
    This tool orchestrates multiple API calls to get the result.
    """
    print("Tool 'get_allowed_locations_tool' called.")
    logger.info("Tool 'get_allowed_locations_tool' called.")
    access_token = await get_new_access_token()
    if not access_token:
        logger.error("Failed to get authentication token.")
        return {"error": "Failed to get authentication token."}

    employee_id = _current_employee_id.get()
    if not employee_id:
        logger.error("Employee ID not found in context. Cannot retrieve locations.")
        return {"error": "Employee ID not found in context. Cannot retrieve locations."}

    associate_info = await get_associate_info(access_token, employee_id)
    if not associate_info:
        logger.error(f"Failed to get associate information for employee {employee_id}.")
        print(f"Failed to get associate information for employee {employee_id}.")
        return {"error": f"Failed to get associate information for employee {employee_id}."}

    location_data = get_allowed_locations_internal(associate_info)

    if not any(location_data.values()):
        logger.info(f"No location information found for employee {employee_id}.")
        print(f"No location information found for employee {employee_id}.")
        return {"message": "No location information found for the associate."}

    return {"allowed_locations": location_data}

class GetAvailableSeatsInput(BaseModel):
    building: str = Field(Description="The building number, for example, '903'.")
    floor: str = Field(Description="The floor number, for example, '1', 'Ground Floor', or 'five'.")
    date: str = Field(Description="The desired date for booking in YYYY-MM-DD format.")

@tool(args_schema=GetAvailableSeatsInput)
async def get_available_seats(building: str, floor: str, date: str) -> dict:
    """
    Checks for available seats for a given building, floor, and date.
    """
    print(f"Tool 'get_available_seats' called with: building={building}, floor={floor}, date={date}")
    access_token = await get_new_access_token()
    if not access_token:
        return {"error": "Failed to get authentication token."}
    try:
        zonal_list = await get_Zonal_info_list(access_token)
    except Exception as e:
        return {"error": "Failed to retrieve zonal information."}
    try:
        booking_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}

    employee_id = _current_employee_id.get()
    if not employee_id:
        return {"error": "Employee ID not found in context. Cannot get available seats."}

    associate_info = user_data_store.get(employee_id, {}).get("associate_api_data")
    if not associate_info:
        associate_info = await get_associate_info(access_token, employee_id)
    if not associate_info:
        return {"error": f"Failed to get associate information for employee {employee_id}."}

    unit_id = extract_unit_id(associate_info, building, floor)
    if not unit_id:
        available_floors = [f['floor'] for f in associate_info.get('availableSeatList', [])]
        return {"error": f"Could not find a matching unit for building '{building}' and floor '{floor}'. Available floors are: {available_floors}"}
    
    booking_info = await get_floor_booking_info(access_token, unit_id, booking_date)
    if not booking_info or "asscocicateInfoList" not in booking_info:
        return {"error": "Could not retrieve booking information for the specified floor and date."}

    seats_with_zonal_info = get_filtered_workspace_list(date, employee_id, booking_info, zonal_list)
    available_seats = [
        {"seat_no": seat['worspaceName'], "workspace_id": seat['workspaceId']}
        for seat in seats_with_zonal_info
        if (seat.get("workspaceStatusId") == 1 and not seat.get("youCannotBookThisZonnal") and
            seat.get('worspaceName') is not None and
            str(seat.get('worspaceName')).strip() != "")]
    
    if not available_seats:
        return {"message": "No available seats found for the specified criteria."}
    
    top_10_seats = available_seats[:10]
    return {"available_seats": top_10_seats}

class BookSeatMultiDateInput(BaseModel):
    dates: List[str] = Field(Description="A list of dates in YYYY-MM-DD format to book the seat for.")
    building: str = Field(Description="The building number, for example, '903'.")
    floor: str = Field(Description="The floor number, for example, '1' or 'Ground'.")
    seat_no: str = Field(Description="The specific seat number to book, e.g., 'L1-048'.")
    timeslot: str = Field(Description="The timeslot. Must be 'full_day', 'first_half', or 'second_half'.")

@tool(args_schema=BookSeatMultiDateInput)
async def book_seat_for_multiple_dates(dates: List[str], building: str, floor: str, seat_no: str, timeslot: str) -> dict:
    """
    Attempts to book a specific seat across multiple dates concurrently.
    """
    logger.info(f"Tool 'book_seat_for_multiple_dates' called for seat {seat_no} on dates: {dates}")
    try:
        dates_to_book_dt = [datetime.fromisoformat(d) for d in dates]
    except ValueError:
        return {"error": "One or more dates are in an invalid format. Please use YYYY-MM-DD."}

    access_token = await get_new_access_token()
    if not access_token:
        return {"error": "System error: Could not get authentication token."}

    employee_id = _current_employee_id.get()
    if not employee_id:
        return {"error": "Employee ID not found in context."}

    associate_api_info = await get_associate_info(access_token, employee_id)
    if not associate_api_info:
        return {"error": "System error: Failed to get associate information."}

    unit_id = extract_unit_id(associate_api_info, building, floor)
    if not unit_id:
        return {"error": f"Could not find a matching unit for building '{building}' and floor '{floor}'."}
    
    booking_tasks = [_book_single_day(date_obj, unit_id, seat_no, timeslot, access_token) for date_obj in dates_to_book_dt]
    multi_day_results = await asyncio.gather(*booking_tasks)
    summary_table = format_booking_results_table(multi_day_results)
    overall_success = any(res.success for res in multi_day_results)
    return {
        "success": overall_success,
        "summary": f"Completed multi-day booking process for seat {seat_no}.",
        "results_table": summary_table
    }

class ViewBookingHistoryInput(BaseModel):
    limit: int = Field(5, Description="The number of recent bookings to display.")

@tool(args_schema=ViewBookingHistoryInput)
async def view_booking_history(limit: int = 5) -> dict:
    """
    Retrieves and displays the user's upcoming and recent booking history.
    """
    access_token = await get_new_access_token()
    if not access_token:
        return {"error": "Failed to get authentication token."}
    
    employee_id = _current_employee_id.get()
    if not employee_id:
        return {"error": "Employee ID not found in context."}

    all_bookings = await get_booking_history(access_token, employee_id)
    if not all_bookings:
        return {"message": "You have no booking history."}

    upcoming_bookings = [
        b for b in all_bookings
        if datetime.fromisoformat(b.fromDate).date() >= datetime.now().date() and b.status == 4
    ]
    upcoming_bookings.sort(key=lambda b: b.fromDate)

    if not upcoming_bookings:
        return {"message": "You have no upcoming bookings."}

    formatted_bookings = [
        f" - **Seat {booking.seat}** on **{datetime.fromisoformat(booking.fromDate).strftime('%Y-%m-%d')}** (Building: {booking.building}, Floor: {booking.floor})"
        for booking in upcoming_bookings[:limit]
    ]
    
    return {"booking_history": "\n".join(formatted_bookings)}

class CancelBookingInput(BaseModel):
    seat_number: str = Field(Description="The seat number to cancel, e.g., 'L1-048'.")
    date: str = Field(Description="The date of the booking to cancel in YYYY-MM-DD format.")

@tool(args_schema=CancelBookingInput)
async def cancel_booking(seat_number: str, date: str) -> dict:
    """
    Cancels a booking for a specific seat on a specific date.
    """
    access_token = await get_new_access_token()
    if not access_token:
        return {"error": "Failed to get authentication token."}

    try:
        target_date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}
    
    employee_id = _current_employee_id.get()
    if not employee_id:
        return {"error": "Employee ID not found in context."}

    all_bookings = await get_booking_history(access_token, employee_id)
    if not all_bookings:
        return {"message": "No bookings found to cancel."}

    found_booking = None
    for booking in all_bookings:
        try:
            booking_date = datetime.fromisoformat(booking.fromDate).date()
            if seats_match(booking.seat, seat_number) and booking_date == target_date_obj and booking.status == 4:
                found_booking = booking
                break
        except ValueError:
            continue
    
    if not found_booking:
        return {"message": f"No active booking found for seat {seat_number} on {date}."}

    cancellation_result = await cancel_booking_api(access_token, found_booking.allocationID)
    return cancellation_result.dict()

# Agent State and Graph Definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

tools = [get_available_seats, book_seat_for_multiple_dates, get_allowed_buildings, get_allowed_locations_tool, view_booking_history, cancel_booking]
tool_node = ToolNode(tools)
llm_with_tools = llm_seat_booking.bind_tools(tools)

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    return "call_tool" if state["messages"][-1].tool_calls else END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"call_tool": "tools", END: END})
workflow.add_edge("tools", "agent")

app_graph = workflow.compile()

def get_system_prompt() -> HumanMessage:
    """Generates the system prompt with the current date for the India time zone."""
    india_tz = pytz.timezone('Asia/Kolkata')
    current_india_time = datetime.now(india_tz)
    known_locations = {"ADU": "Adugodi", "KOR": "Koramangala", "COB-GTP": "COB - Global Technology Park", "KT1": "Coimbatore SEZ", "KT2": "Coimbatore SEZIL", "EC1": "Electronic City", "PBP": "Pune", "EC": "Electronic City", "Hyd": "Hyderabad", "COB": "Coimbatore"}
    known_buildings = ["601", "602", "603", "605", "BlockA", "BlockB", "BlockC", "102", "101", "106", "EC 360 BP", "HM-Vibha-Tower", "Hyderabad3", "Tower A", "Tower B & C", "ta", "tb","tc","901","903","905"]
    prompt_content = f"""
    You are a helpful assistant designed to book, cancel seats and view bookings history. Today's date is {current_india_time.strftime('%Y-%m-%d')}. Your primary goal is to guide the user through their request by following a clear, step-by-step process.
    ### Greet the user and determine their intent: booking, canceling, or viewing history. ###
    ---
    ### --- Booking a Seat --- ###
    **Step 1:** Identify and Validate Building
     - A. ask the user to specify a building.
     - B. Use the `get_allowed_buildings` tool for the permissible Buildings.
     - C. Compare the user's input against the list of allowed buildings and for your reference the Buildings examples: {', '.join(known_buildings)} knowledge base ,Attempt to find a match with permissible Buildings, even if the user's input is a synonym or a close variation (e.g., "ta or towerA" should match "SEZL-TA").
     - D. **If a match is found:** Proceed to the booking path.
     - E. **If no match is found:** Inform the user and show them the list of available buildings for the selected location.
    **Step 2:** Identify Booking seat Type
    - Ask the user whether they want to book a **workspace** or a **cabin**.
    - Based on their selection, follow the appropriate path.
    ---
    #### Workspace Booking Flow ####
    **Step 3:** Ask for floor, date(s), and timeslot (first half, second half, full day).
    **Follow one of the two paths:**
      #### Path 1.1: User knows the Workspace number ####
      - **Step 4:** Ask for the Workspace number.
      - **Step 5:** Use the `book_seat_for_multiple_dates` tool with the validated building, and other details.
      #### Path 1.2: User doesn't know the Workspace number ####
      - **Step 4:** If the user does't know the Workspace number and wants to know the available Workspaces
      - **Step 5:** Use the `get_available_seats` tool with the validated building, floor, and date, to get only top 10 available Workspaces.
      - **Step 6:** Present the available Workspaces to the user and ask them to choose a Workspace or take if they have any specific Workspace.
      - **Step 7:** Use the `book_seat_for_multiple_dates` tool with the validated building, and other details to finalize the booking.
    ---
    #### Cabin Booking Flow ####
    **Step 3:** Ask for floor, date(s), and timeslot (first half, second half, full day).
    **Step 4:** Ask for the desired cabin number.
    **Step 5:** Do **not** format cabin numbers. Keep them as provided by the user (e.g., "568", "664").
    **Step 6:** Use the `book_seat_for_multiple_dates` tool with the validated building, and other details to finalize the booking.
    ---
    **Step If user mentioned location:** Identify and Validate Location Only if the user mentioned
    - A. If the user mentioned their desired office location.
    - B. Use the `get_allowed_locations_tool` to fetch the list of permissible locations.
    - C. Compare the user's input against the fetched list and for your reference the location examples: {dict_to_readable_string(known_locations)} " knowledge base. Attempt to find a match with permissible locations, even if the user's input is a synonym or a close variation (e.g., "Koramangala" should match "KOR").
    - D. **If a confident match is found:** Confirm the matched location with the user (e.g., "Got it, you've selected Koramangala. Is that correct?"). Upon confirmation, proceed to the next step.
    - E. **If no match is found:** Inform the user that you couldn't recognize the location and present them with the list of available locations returned by the tool.
    ---
    ### --- Canceling a Seat --- ###
    **Step 1:** Gather Information
    - Ask the user for the **seat number** and the **date** of the booking they wish to cancel.
    **Step 2:** Execute Cancellation
    - Use the cancel_booking tool with the provided seat number and date.
    **Step 3:** Report Result
    - Inform the user of the outcome of the cancellation attempt.
    ---
    ### --- Viewing Booking History --- ###
    **Step 1:** Acknowledge Request
    - Confirm user's request to see their bookings.
    **Step 2:** Retrieve History
    - Use `view_booking_history` tool.
    **Step 3:** Display History
    - Show a nicely formatted list of upcoming bookings.
    ---
    **Important Rules:**
    - **Seat Number Interpretation:** If the user provides a seat number **without a prefix** (e.g., "736"), ask if it's a *workspace* or a *cabin*. For **Workspace**, format as `L<FloorNumber>-<SeatCode>`. For **Cabin**, use the number as-is. If a prefix is present (e.g., "L1-736"), assume it is a workspace.
    - **Workspace Formatting Rules**: Always convert and return seat numbers in the standard format: L<FloorNumber>-<SeatCode>. Examples: “l0 b034” → L0-B034, “l1 001” → L1-001, “l12024” → L12-024.
    - **Cabin Formatting Rules**: Do **not** apply any formatting. Use directly: "568", "664".
    - **Allowed Booking Days**: Bookings are allowed for **today and up to the current week only**.
    - **Non Booking Related Queries**: Politely state that you specialize in seat booking.
    - **Communication**: Be friendly, clear, and guide the user step-by-step.
    """
    return HumanMessage(content=prompt_content)


# --- 10. API Endpoints (Merged) ---
@app.post("/chat", response_model=ChatResponse, tags=["Seat Booking"])
async def chat(
    request: ChatRequest = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    Session_id: Optional[str] = Header(None, alias="Session_id")
    ):
    if not Session_id:
        raise HTTPException(status_code=400, detail="A 'Session_id' header is required for the chat.")
    
    message_text = request.EmployeeQueryMessage.strip()
    authorization = credentials.credentials
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization is required.")
    
    token_context_reset_token = None
    try:
        encrypted_data = validate_token(authorization)
        response_data = encrypted_data.get('ResponseData', [])
        print(response_data)
        newdata = response_data[0] if response_data and isinstance(response_data, list) else None
        decrypted_data = decrypt_text(newdata)
        print(decrypted_data)
        user_info = parse_user_data(decrypted_data)
        employee_id = user_info["employee_id"]
        print(employee_id)
        try:
          employee_id = str(int(str(employee_id).strip().replace('"', '').replace("'", '')))
        #   await log_conversation_to_db(Session_id, message_text, 'user', employee_id=int(employee_id))
        except ValueError:
           error_message = f"Internal chatbot error: {str(e)}"
        #    await log_conversation_to_db(Session_id, error_message, 'ai_error', employee_id=int(employee_id))
           raise HTTPException(status_code=400, detail=f"Invalid employee_id format in token: {employee_id}")
       
        token_context_reset_token = _current_employee_id.set(employee_id)
        
        if Session_id not in conversation_state_store:
            current_messages = [get_system_prompt()]
        else:
            current_state = conversation_state_store[Session_id]
            current_messages = current_state["messages"]

        current_messages.append(HumanMessage(content=message_text))
        initial_state = {"messages": current_messages}
        result = await app_graph.ainvoke(initial_state)
        print(result)
        conversation_state_store[Session_id] = result
        final_bot_response = result['messages'][-1].content
        # await log_conversation_to_db(Session_id, final_bot_response, 'ai', employee_id=int(employee_id))
        
    except HTTPException as e:
        raise e
    except Exception as e:
        error_message = f"Internal chatbot error:"
        # await log_conversation_to_db(Session_id, error_message, 'ai_error', employee_id=(employee_id))
        raise HTTPException(status_code=500, detail=error_message)
    
    finally:
        if token_context_reset_token:
            _current_employee_id.reset(token_context_reset_token)
    
    return ChatResponse(item=final_bot_response, status="True")


@app.get("/")
async def root():
    return {"message": f"LangGraph Arena Bosch Chatbot API: live14"}
