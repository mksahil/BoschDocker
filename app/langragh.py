# OPTIMISED CODE FOR BETTER ERROR HANDLING FOR EXTERNAL API'S
# --- 1. Imports and Setup ---
import httpx
import re
import os
import json
import operator
from datetime import datetime
from typing import List, Dict, Optional, Any, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import contextvars
from fastapi import Depends, FastAPI, Body, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
from functools import wraps
from time import time
from datetime import datetime
from typing import Dict, Optional, Any
from pydantic import BaseModel
import pyodbc
from sqlalchemy.engine import URL
from sqlalchemy import create_engine

import logging


logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# # Connection string for Windows Authentication
# connection_string_windows = (
#     r'DRIVER={ODBC Driver 17 for SQL Server};'
#     r'SERVER=(local)\SQLEXPRESS;'  # Or 'localhost\InstanceName' for a named instance
#     r'DATABASE=BoschDB;'
#     r'Trusted_Connection=yes;'
# )


# Connection string for SQL Server Authentication
connection_string_windows = (
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=boschserver.database.windows.net;'  # Or 'localhost\InstanceName'
    r'DATABASE=seatbooking;'
    r'UID=useradmin;'
    r'PWD=MDsahil@123;'
)

# Pydantic models

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

from fastapi import  HTTPException
import requests

# --- Context variable for dynamically setting employee_id per request ---
_current_employee_id = contextvars.ContextVar("employee_id", default=None)

# In-memory user data store (for associate info like name, email, associate_id_val, etc.)
user_data_store: Dict[str, Dict[str, Any]] = {}

# In-memory conversation state store (for LangGraph messages history per session)
# This remains essential for the agent's turn-by-turn conversational context.
conversation_state_store: Dict[str, Any] = {} # Stores AgentState here


# --- NEW: Database Connection and Logging Functions ---
import pyodbc
import asyncio
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

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
    Runs the blocking DB call in a separate thread to avoid blocking FastAPI's event loop.
    """
    sql = text("INSERT INTO ChatConversations (SessionID, MessageText, Sender,employee_id) VALUES (:session_id, :message, :sender,:employee_id)")
    conn = None
    trans = None
    try:
        conn = get_db_connection()
        if conn:
            # Run the blocking database operations in a separate thread
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
        # Log the error but don't crash the main application
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


from tenacity import retry, stop_after_attempt, wait_exponential,retry_if_exception_type
import httpx
# --- Token service functions  ---
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
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
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
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
        print("print parse user data")
        parts = decrypted_data.split(',')
        print("parts",parts)
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


# API Helper functions
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import httpx

# Apply the decorator to a function that makes a network call
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
)
async def get_new_access_token():
    """Fetches a new access token from the authentication server."""
    url = "https://flexibook.boschassociatearena.com/connect/token"
    payload = {
        'client_id': 'C851F411-2BFA-4578-A4EF-D420EC6CBB64',
        'client_secret': '90D18C4E-32E1-4DBF-B30F-CC91F47ADCF6',
        'grant_type': 'password', 'username': '11446688' # This username might also need to be dynamic for real scenarios
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
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
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
            # Store user details in user_data_store keyed by employee_id
            if data and isinstance(data, dict):
                 user_data_store[employee_id] = {
                     "collected_info": {
                         "employee_name": data.get('flexibleUserName'),
                         "employee_email": data.get('email'),
                         "associate_id_val": data.get('associateId')
                     },
                     "associate_api_data": data # Store full data for unit_id extraction later
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
    # print("associate_info:",associate_info)
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
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
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
            print(response)
            response.raise_for_status()
            data = response.json()
            print("line 298, response floor booking info",response.json)
            print("Successfully fetched floor booking info.")
            logger.info(f"Successfully fetched floor booking info for unit {unit_id} on {date_to_check.strftime('%Y-%m-%d')}")
            return data
    except Exception as e:
        print(f"Error getting booking info for unit {unit_id}: {str(e)}")
        logger.error(f"Error getting booking info for unit {unit_id}: {str(e)}")
        return None


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
)
async def book_seat(access_token, employee_id_param: Optional[str], workspace_id, booking_date: datetime, time_slot: str = 'full_day'):
    """Books a specific seat for an employee."""
    employee_id = employee_id_param or _current_employee_id.get()
    if not employee_id:
        raise ValueError("Employee ID not available in context or provided for booking.")

    # print(f"--- Attempting to book seat for {employee_id} ---")
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
        # Fallback if associate info wasn't pre-fetched or retrieved for this employee_id
        temp_associate_info = await get_associate_info(access_token, employee_id) # Pass employee_id
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
    print(" line 380 :::::::::::::::::check seat availability called::::::::::::::::::::::::::::")
    logger.info(f"Checking seat availability for seat number: {seat_number}")
    # print("line 381 floor_booking_data ",floor_booking_data["asscocicateInfoList"])
    workspaceName=""
    for seat in floor_booking_data["asscocicateInfoList"]:
        # print("line 383  :::::::::::seat:::::::::::",seat)
        # print("line 384", seat.get("worspaceName", ""), seat_number)
        workspaceName=seat.get("worspaceName", "")
        print("workspace name:",workspaceName)
        logger.info(f"Checking seat: {workspaceName} against requested seat number: {seat_number}")
        print("seay number",seat_number)
        if workspaceName!='null' and  workspaceName!=None:
            if workspaceName.lower()== seat_number.lower():
             print(" line 378 seat name matched")
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
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
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
    wait=wait_exponential(multiplier=1, min=2, max=60), # Waits 2s, 4s, 8s...
    stop=stop_after_attempt(5), # Retries up to 5 times
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)) # Retry on network or server errors
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

# --- 3. LangGraph Agent Implementation ---

# --- Agent Tools ---
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
            # Any key that isn't 'floor' or 'iscob' is a building number
            # print("line 491 key",key)
            if key.lower() not in ['floor', 'iscob']:
                allowed_buildings.add(key)
    if not allowed_buildings:
        return {"message": "No allowed buildings found for the associate."}
    return {"allowed_buildings": sorted(list(allowed_buildings))}


def get_allowed_locations_internal(associate_info: Dict[str, Any]) -> Dict[str, str]:
    if not associate_info:
        return {}
    
    location_info = {
        'office_location_code': associate_info.get('officeLocationCode', '').upper(),
        'office_location_name': associate_info.get('officeLocationName', '').lower(),
        'location_name': associate_info.get('locationName', '').lower()
    }
    
    return location_info

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

    # Use the provided function to extract location information
    location_data = get_allowed_locations_internal(associate_info)
    
    if not any(location_data.values()):
        logger.info(f"No location information found for employee {employee_id}.")
        print(f"No location information found for employee {employee_id}.") 
        return {"message": "No location information found for the associate."}
        
    return {"allowed_locations": location_data}

class GetAvailableSeatsInput(BaseModel):
    building: str = Field(description="The building number, for example, '903'.")
    floor: str = Field(description="The floor number, for example, '1', 'Ground Floor', or 'five'.")
    date: str = Field(description="The desired date for booking in YYYY-MM-DD format.")

@tool(args_schema=GetAvailableSeatsInput)
async def get_available_seats(building: str, floor: str, date: str) -> dict:
    """
    Checks for available seats for a given building, floor, and date.
    This tool orchestrates multiple API calls to get the result.
    """
    print(f"Tool 'get_available_seats' called with: building={building}, floor={floor}, date={date}")
    try:
        booking_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}

    access_token = await get_new_access_token()
    if not access_token:
        return {"error": "Failed to get authentication token."}

    employee_id = _current_employee_id.get()
    if not employee_id:
        return {"error": "Employee ID not found in context. Cannot get available seats."}

    # Try to get associate info from cache first, then fetch if not found
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

    available_seats = [
        {"seat_no": seat['worspaceName'], "workspace_id": seat['workspaceId']}
        for seat in booking_info["asscocicateInfoList"]
        if seat.get("workspaceStatusId") == 1
    ]
     
    top_6_seats = available_seats[:10] 
    if not available_seats:
        return {"message": "No available seats found for the specified criteria."}
    return {"available_seats": top_6_seats}

# --- Helper function to process a single date ---
async def _book_single_day(date_obj: datetime, unit_id: str, seat_no: str, timeslot: str, access_token: str) -> DayBookingStatus:
    """Helper coroutine to check and book one day."""
    employee_id = _current_employee_id.get() # Context is available here

    floor_booking_data = await get_floor_booking_info(access_token, unit_id, date_obj)
    # print("line 599 floor_booking_data",floor_booking_data)
    if not floor_booking_data:
        return DayBookingStatus(date=date_obj, success=False, message="Failed to get availability for this date.")

    seat_availability = check_seat_availability(floor_booking_data, seat_no)
    print("line 602 seat availabilty",seat_availability)
    if not seat_availability.is_available:
        return DayBookingStatus(date=date_obj, success=False, message=f"Seat not available ({seat_availability.status_name}).")

    booking_api_result = await book_seat(access_token, employee_id, seat_availability.workspace_id, date_obj, timeslot)
    return DayBookingStatus(date=date_obj, success=booking_api_result.success, message=booking_api_result.message)

class BookSeatMultiDateInput(BaseModel):
    dates: List[str] = Field(description="A list of dates in YYYY-MM-DD format to book the seat for.")
    building: str = Field(description="The building number, for example, '903'.")
    floor: str = Field(description="The floor number, for example, '1' or 'Ground'.")
    seat_no: str = Field(description="The specific seat number to book across all dates, e.g., 'L1-048'.")
    timeslot: str = Field(description="The timeslot for all bookings. Must be 'full_day', 'first_half', or 'second_half'.")

@tool(args_schema=BookSeatMultiDateInput)
async def book_seat_for_multiple_dates(dates: List[str], building: str, floor: str, seat_no: str, timeslot: str) -> dict:
    """
    Attempts to book a specific seat across multiple dates concurrently.
    Use this when the user knows the exact seat they want for several days.
    """
    print(f"--- TOOL: book_seat_for_multiple_dates for seat {seat_no} ---")
    logger.info(f"Tool 'book_seat_for_multiple_dates' called for seat {seat_no} on dates: {dates}, building: {building}, floor: {floor}, timeslot: {timeslot}")
    
    try:
        dates_to_book_dt = [datetime.fromisoformat(d) for d in dates]
        logger.info(f"Parsed dates for booking: {dates_to_book_dt}")
    except ValueError:
        logger.error("Invalid date format in input. Expected YYYY-MM-DD.")
        print("Invalid date format in input. Expected YYYY-MM-DD.") 
        return {"error": "One or more dates are in an invalid format. Please use YYYY-MM-DD."}

    access_token = await get_new_access_token() # This will be cached
    if not access_token:
        logger.error("Failed to get authentication token.")
        return {"error": "System error: Could not get authentication token."}

    employee_id = _current_employee_id.get()
    if not employee_id:
        logger.error("Employee ID not found in context.")
        return {"error": "Employee ID not found in context."}

    # This call is also cached
    associate_api_info = await get_associate_info(access_token, employee_id)
    if not associate_api_info:
        logger.error(f"Failed to get associate information for employee {employee_id}.")
        return {"error": "System error: Failed to get associate information."}

    unit_id = extract_unit_id(associate_api_info, building, floor)
    print("unit_id",unit_id)
    if not unit_id:
        logger.error(f"Could not find a matching unit for building '{building}' and floor '{floor}'.")
        return {"error": f"Could not find a matching unit for building '{building}' and floor '{floor}'."}
    
    # Create concurrent tasks
    booking_tasks = [
        _book_single_day(date_obj, unit_id, seat_no, timeslot, access_token)
        for date_obj in dates_to_book_dt
    ]
    
    # Run all tasks in parallel
    multi_day_results = await asyncio.gather(*booking_tasks)
    print("multi booking day results")

    summary_table = format_booking_results_table(multi_day_results)
    overall_success = any(res.success for res in multi_day_results)
    logger.info(f"Multi-day booking process completed for seat {seat_no}. Overall success: {overall_success}")
    return {
        "success": overall_success,
        "summary": f"Completed multi-day booking process for seat {seat_no}.",
        "results_table": summary_table
    }

# --- Cancellation and History Tools ---
class ViewBookingHistoryInput(BaseModel):
    limit: int = Field(5, description="The number of recent bookings to display.")

@tool(args_schema=ViewBookingHistoryInput)
async def view_booking_history(limit: int = 5) -> dict:
    """
    Retrieves and displays the user's upcoming and recent booking history.
    """
    print(f"Tool 'view_booking_history' called with limit: {limit}")
    access_token = await get_new_access_token()
    if not access_token:
        return {"error": "Failed to get authentication token."}
    
    employee_id = _current_employee_id.get()
    if not employee_id:
        return {"error": "Employee ID not found in context. Cannot view booking history."}

    all_bookings = await get_booking_history(access_token, employee_id)
    if not all_bookings:
        return {"message": "You have no booking history."}

    # Filter for upcoming bookings first
    upcoming_bookings = [
        b for b in all_bookings
        if datetime.fromisoformat(b.fromDate).date() >= datetime.now().date() and b.status == 4
    ]
    # Sort by date
    upcoming_bookings.sort(key=lambda b: b.fromDate)

    if not upcoming_bookings:
        return {"message": "You have no upcoming bookings."}

    # Format for display
    formatted_bookings = []
    for booking in upcoming_bookings[:limit]:
        booking_date = datetime.fromisoformat(booking.fromDate).strftime('%Y-%m-%d')
        formatted_bookings.append(
            f" - **Seat {booking.seat}** on **{booking_date}** (Building: {booking.building}, Floor: {booking.floor})"
        )
    
    return {"booking_history": "\n".join(formatted_bookings)}

class CancelBookingInput(BaseModel):
    seat_number: str = Field(description="The seat number to cancel, e.g., 'L1-048'.")
    date: str = Field(description="The date of the booking to cancel in YYYY-MM-DD format.")

@tool(args_schema=CancelBookingInput)
async def cancel_booking(seat_number: str, date: str) -> dict:
    """
    Cancels a booking for a specific seat on a specific date.
    """
    print(f"Tool 'cancel_booking' called for seat: {seat_number} on date: {date}")
    access_token = await get_new_access_token()
    if not access_token:
        logger.error("Failed to get authentication token.")
        return {"error": "Failed to get authentication token."}

    try:
        target_date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        logger.error("Invalid date format provided for cancellation.")
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}
    
    employee_id = _current_employee_id.get()
    if not employee_id:
        logger.error("Employee ID not found in context. Cannot cancel booking.")
        return {"error": "Employee ID not found in context. Cannot cancel booking."}

    all_bookings = await get_booking_history(access_token, employee_id)
    if not all_bookings:
        logger.info("No bookings found for cancellation.")
        return {"message": "No bookings found to cancel."}

    found_booking = None
    for booking in all_bookings:
        try:
            booking_date = datetime.fromisoformat(booking.fromDate).date()
            # Check for both seat match and status (4 for active bookings)
            if seats_match(booking.seat, seat_number) and booking_date == target_date_obj and booking.status == 4:
                found_booking = booking
                break
        except ValueError:
            continue
    
    if not found_booking:
        logger.info(f"No active booking found for seat {seat_number} on {date}.")
        return {"message": f"No active booking found for seat {seat_number} on {date}."}

    # Proceed with cancellation
    cancellation_result = await cancel_booking_api(access_token, found_booking.allocationID)
    logger.info(f"Cancellation result for seat {seat_number} on {date}: {cancellation_result}")
    return cancellation_result.dict()

# --- Agent State and Graph Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- Combined Tool List ---
tools = [
    get_available_seats,
    book_seat_for_multiple_dates,
    get_allowed_buildings,
    get_allowed_locations_tool,
    view_booking_history,
    cancel_booking
    ]
tool_node = ToolNode(tools)

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
                    azure_deployment="GPT4",
                    api_key="af6c5f2c43294f1e9287a50d652c637e", 
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    azure_endpoint="https://ctmatchinggpt.openai.azure.com/", 
                    temperature=0)

llm_with_tools = llm.bind_tools(tools)

# Agent node: invokes the LLM with the current conversation messages
def agent_node(state: AgentState):
    print("---AGENT NODE---")
    messages_to_send = state["messages"]
    print("::::::::::::::::::::::::::::::::::::::::::Aget Node::::::::::::::::::::::::::::::::::")
    print("messages_to_send::::::::::",messages_to_send)
    try:
        print(f"Invoking LLM with {len(messages_to_send)} messages.")
        response = llm_with_tools.invoke(messages_to_send)
        # print(":::::::::::::::::::::::llm response::::::::::::::::::::::::::::::::::::",response)
        return {"messages": [response]}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print("Rate limit hit. Tenacity will retry...")
            raise
        else:
            return {"messages": [AIMessage(content=f"An unexpected API error occurred: {e}")]}

# Conditional edge: decides whether to continue with tool call or end
def should_continue(state: AgentState) -> str:
    # If the last message from the agent has tool calls, route to tools
    return "call_tool" if state["messages"][-1].tool_calls else END

# Define the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"call_tool": "tools", END: END})
workflow.add_edge("tools", "agent")

# Compile the LangGraph application
app_graph = workflow.compile() 

# --- FastAPI App Initialization ---
app = FastAPI(title="LangGraph Arena Bosch Chatbot API", version="live5")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def dict_to_readable_string(location_dict: Dict[str, str]) -> str:
    """Helper function to convert location dictionary to readable string format."""
    return ", ".join([f"{code}: {name}" for code, name in location_dict.items()])
    
# System prompt for the chatbot
known_locations = {
    "ADU": "Adugodi", "KOR": "Koramangala", "COB-GTP": "COB - Global Technology Park",
    "KT1": "Coimbatore SEZ", "KT2": "Coimbatore SEZIL", "EC1": "Electronic City",
    "HMVT": "HM-Vibha Tower", "PBP": "Pune",
    "EC": "Electronic City", "Hyd": "Hyderabad", "COB": "Coimbatore"
}

known_buildings = [
"601",
"602",
"603",
"605",
"BlockA",
"BlockB",
"BlockC",
"102",
"101",
"106",
"EC 360 BP",
"HM-Vibha-Tower",
"Hyderabad3",
"Tower A",
"Tower B & C",
"ta",
"tb","tc","901","903","905"]

import pytz

def get_system_prompt() -> HumanMessage:
    """
    Generates the system prompt with the current date for the India time zone.
    This ensures the date is fresh for every new conversation.
    """
    india_tz = pytz.timezone('Asia/Kolkata')
    # This will now be calculated every time the function is called
    current_india_time = datetime.now(india_tz) 
    print("current_india_time:",current_india_time)
    prompt_content = f"""
    You are a helpful assistant designed to book, cancel, and view seat bookings.
    Today's date is {current_india_time.strftime('%Y-%m-%d')}.
    Your primary goal is to guide the user through their request by following a clear, step-by-step process.
    ### Greet the user and determine their intent: booking, canceling, or viewing history. ###
 
    ---
    ### --- Booking a Seat --- ###
 
    **Step 1: Identify and Validate Building**
    - A. ask the user to specify a building.
    - B. Use the `get_allowed_buildings` tool for the permissible Buildings.
    - C. Compare the user's input against the list of allowed buildings and for your reference the Buildings examples: {', '.join(known_buildings)} knowledge base ,Attempt to find a match with permissible Buildings, even if the user's input is a synonym or a close variation (e.g., "ta or towerA" should match "SEZL-TA").
    - D. **If a match is found:** Proceed to the booking path.
    - E. **If no match is found:** Inform the user and show them the list of available buildings for the selected location.
     
    **Step 2:** Ask for the desired floor, date or list of dates and the desired timeslot(first half,second half,fullday) .
 
    **Follow one of the two booking paths below:**
      #### Path 1.1: User knows the seat number####
    - **Step 3:** Ask the user for the specific seat number.
    - **Step 4:** Use the `book_seat_for_multiple_dates` tool with the validated building, and other details.
    
     #### Path 1.2: User does't know the seat number####
    - **Step 3:** If the user does't know the seat and wants to know the available seats
    - **Step 4:** Use the `get_available_seats` tool with the validated building, floor, and date, to get only top 10 available seats.
    - **Step 5:** Present the available seats to the user and ask them to choose a seat or take if they have any specific seat.
    - **Step 6:** Use the `book_seat_for_multiple_dates` tool with the validated building, and other details to finalize the booking.
   
    **Step If user mentioned location: Identify and Validate Location Only if the user mentioned**
    - A. If the user mentioned their desired office location.
    - B. Use the `get_allowed_locations_tool` to fetch the list of permissible locations.
    - C. Compare the user's input against the fetched list and for your reference the location examples: {dict_to_readable_string(known_locations)} " knowledge base. Attempt to find a match with permissible locations, even if the user's input is a synonym or a close variation (e.g., "Koramangala" should match "KOR").
    - D. **If a confident match is found:** Confirm the matched location with the user (e.g., "Got it, you've selected Koramangala. Is that correct?"). Upon confirmation, proceed to the next step.
    - E. **If no match is found:** Inform the user that you couldn't recognize the location and present them with the list of available locations returned by the tool.
 
    --- 
    ### --- Canceling a Seat --- ###
 
    **Step 1: Gather Information**
    - Ask the user for the **seat number** and the **date** of the booking they wish to cancel.
 
    **Step 2: Execute Cancellation**
    - Use the `cancel_booking` tool with the provided seat number and date.
 
    **Step 3: Report Result**
    - Inform the user of the outcome of the cancellation attempt.
 
    ---
    ### --- Viewing Booking History --- ###
 
    **Step 1: Acknowledge Request**
    - When the user asks to see their bookings, acknowledge the request.
 
    **Step 2: Retrieve History**
    - Use the `view_booking_history` tool.
 
    **Step 3: Display History**
    - Present the formatted list of upcoming bookings to the user.
    ---
 
    **Important Rules:**
    - **Seat Number Formatting**:Users may provide seat numbers in any format (e.g., “l0 bo34”, “L12bo24”, “l3a054”).Always convert and return seat numbers in the standard format:L<FloorNumber>-<SeatCode>
         For example:
                      “l0 b034” → L0-B034
                       “l1 001” → L1-001
                       “l12024” → L12-024
                       “l3a054” → L3-A054
                       “l03177” → L03-177
                       “l17a057” → L17-A057
                       “l0C085” → L0-C085
                     
    - **Communication**: Be friendly and clear, and present results cleanly, do not show the steps numbers to the users.
    -**Allowed booking days**: The user can book a seat for today and up to this week only .
    """
    return HumanMessage(content=prompt_content)

# --- MODIFIED: /chat endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    Session_id: Optional[str] = Header(None, alias="Session_id")
    ):
    print("-----------------Received Chat Request------------------")
    if not Session_id:
        raise HTTPException(status_code=400, detail="A 'Session_id' header is required for the chat.")
    
    message_text = request.EmployeeQueryMessage.strip()

    authorization = credentials.credentials
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization is required.")
    
    token_context_reset_token = None
    try:
        # (Your token validation and user info parsing logic remains the same)
        encrypted_data = validate_token(authorization)
        response_data = encrypted_data.get('ResponseData', [])
        newdata = response_data[0] if response_data and isinstance(response_data, list) else None
        decrypted_data =  decrypt_text(newdata)
        user_info = parse_user_data(decrypted_data)
        employee_id = user_info["employee_id"]
          # atharwa pune
        # employee_id = 35581584 
      
        # abhishek cob gtp
        # employee_id = 35489300 
     
        # Aparajeeta hydrabad3
        # employee_id = 31563433 
        try:
          employee_id = str(int(str(employee_id).strip().replace('"', '').replace("'", '')))
          await log_conversation_to_db(Session_id, message_text, 'user', employee_id=int(employee_id))
        except ValueError:
           print(f"Invalid employee_id from token: {employee_id}")
           raise HTTPException(status_code=400, detail=f"Invalid employee_id format in token: {employee_id}")
        
        token_context_reset_token = _current_employee_id.set(employee_id)
        print(f"Employee ID set in context: {employee_id}")

        # (Your conversation state management logic remains the same)
        if Session_id not in conversation_state_store:
            print(f"Initializing new session for Session_id: {Session_id}")
            current_messages = [get_system_prompt()]
        else:
            print(f"Resuming session for Session_id: {Session_id}")
            current_state = conversation_state_store[Session_id]
            current_messages = current_state["messages"]

        current_messages.append(HumanMessage(content=message_text))
        initial_state = {"messages": current_messages}
        result = await app_graph.ainvoke(initial_state)
        conversation_state_store[Session_id] = result
        final_bot_response = result['messages'][-1].content
        
        # --- NEW: Log the final AI response to the database ---
        await log_conversation_to_db(Session_id, final_bot_response, 'ai', employee_id=int(employee_id))
    except HTTPException as e:
        raise e 
    except Exception as e:
        error_message = f"Internal chatbot error: {str(e)}"
        print(f"Unhandled error during chat processing: {e}")
        # --- NEW: Log error messages as well for analysis ---
        await log_conversation_to_db(Session_id, error_message, 'ai_error', employee_id=int(employee_id))
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        if token_context_reset_token:
            _current_employee_id.reset(token_context_reset_token)
            print(f"Employee ID context reset for {employee_id}.")
    return ChatResponse(item=final_bot_response, status="True")

@app.get("/")
async def root():
    return {"message": f"LangGraph Arena Bosch Chatbot API: live5"}
