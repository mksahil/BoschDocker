import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import httpx
from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
import requests

llm = AzureChatOpenAI(
                    azure_deployment="GPTbgsw-openAIservice-Voiceseatbooking4",
                    api_key="2u2cSvJIlkFgj6BKsabPTVeIS4zcFlCu49yk2JxzrmUkIDTycp9qJQQJ99BHACYeBjFXJ3w3AAABACOGoOsh", 
                    model="gpt-4o",
                    api_version="2024-12-01-preview",
                    azure_endpoint="https://bgsw-openaiservice-voiceseatbooking-p-eus-001.openai.azure.com/openai/deployments/bgsw-openAIservice-Voiceseatbooking/chat/completions?api-version=2025-01-01-preview", 
                    temperature=0)


Dn4yI0dHukIc2ih6lDxHbQTUSGLLWxqprwrarERPEHQljn7d7yoxJQQJ99BFACYeBjFXJ3w3AAABACOGZwJO


eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJITnU5TTNvNUdMNDc3MXJJeUphS0RBPT0iLCJhdWQiOlsiRWZvRTYvVGxDeDhvcWVsbFB5MUZLV3lHeXJtZytKblE2eGl1Rzk0aVRqRm1KbkFJKytCZDJJYVZNVFJjTEErciIsIkVmb0U2L1RsQ3g4b3FlbGxQeTFGS1d5R3lybWcrSm5RNnhpdUc5NGlUakZtSm5BSSsrQmQySWFWTVRSY0xBK3IiXSwibmFtZSI6IlRaWVF1bytCNXFHSmNlYWlia2x3ZWFWUjVEVE9aUmo4cUNSL3BPVWhVd05GbEl3L3B6WXVKenFjV3FuU2RvdkUiLCJnaXZlbl9uYW1lIjoiZU1nTVpSemdBenlKUXBEZ1hKUGlLZz09IiwiaWF0IjoxNzUyMjI4NTQzLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL2V4cGlyYXRpb24iOiI5LzQvMjAyNiAxMDowOTowMyBBTSIsImV4cCI6MTc4ODUxNjU0M30.Hf-rjvKHGorp1ktaaVd5PRG9aNYkJ5q6kxKqZrm15e9FEvQfRIJlQBJD_NGuL8s10lhPIpR_9UmKjJw81sCp9wY5XpyBTUUAYIMaE2VMVBh2aBr-qdZrrPKaUcLgD-YzkX3waO4BIhkyfnQkLsNZVri8ufqQDw8W59Qyw_XM0-s13tgTnxq6wEfsowCvILIuMNqkvRO-rXM_LTaEvBAiGAyAga9PMmF4k9yTq-gpTtBxQpCLWL9ckU1G4e8gxlV61kQHwiY9woMjibiXv57gHnLMdJsn_UMkk7Zz9smaEtLLUVxyLrfFAzXCCijzYGRb8Rjxrhn85jWl-E0muFuaGg



https://bosch-chartbot.azurewebsites.net/docs#/default/chat_chat_post

vaishnavi
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJITnU5TTNvNUdMNDc3MXJJeUphS0RBPT0iLCJhdWQiOlsiRWZvRTYvVGxDeDhvcWVsbFB5MUZLV3lHeXJtZytKblE2eGl1Rzk0aVRqRm1KbkFJKytCZDJJYVZNVFJjTEErciIsIkVmb0U2L1RsQ3g4b3FlbGxQeTFGS1d5R3lybWcrSm5RNnhpdUc5NGlUakZtSm5BSSsrQmQySWFWTVRSY0xBK3IiXSwibmFtZSI6Ilkzd0lBcDRaZmo3Wm9kczllYmZjV3FwZXl6VzdMVGo2eDErTHh1ejBVREMrMzFocHd6b0UzaS9oRlhjR0NZbkYiLCJnaXZlbl9uYW1lIjoiZU1nTVpSemdBenlKUXBEZ1hKUGlLZz09IiwiaWF0IjoxNzUyNDg1MzUzLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL2V4cGlyYXRpb24iOiI5LzcvMjAyNiA5OjI5OjEzIEFNIiwiZXhwIjoxNzg4NzczMzUzfQ.yjk0FcN-u6fckiISHNiub2vAYX9XokvKsl2Ceov-joHZ2vXgmwhRqsJhfhCaBIiiHOY6VZ3nMVSChafPN0GmmtHUTEcAJ7JuVII8dc5DWHllrI3-S2kANH4WvLLYJxXaqeuqkRN3mQbUmvjCQjK0J3mAZ1AdBY3uI38EwR0XloMkRkd_rf5MKL9_aFVXx84lYx_WnJKNLNfPMgtpdlY4uchY-UEksQADGBzfZvrC0ojWKe9LYPVZ0orxLxYVu3FNpCBLOsfCJlO0ig7tCj7rCTHF8c5ulEG3652lgoE2tesm-EUnmCF7t-8mU9_Jgihj5dA6WxHUk_FnnwokDfw97g


# FastAPI App Initialization
app = FastAPI(title="Agile Arena Bosch Chatbot API", version="UATToken_service") # Version updated for all fixes

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Configuration
llm = AzureChatOpenAI(
                    azure_deployment="gpt-4o-mini",
                    api_key="", 
                    model="gpt-4o-mini",
                    api_version="2024-02-15-preview",
                    azure_endpoint="https://agsopenaiservice.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview", 
                    temperature=0,)

# In-memory user data store
user_data_store: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    EmployeeQueryMessage: str


class SeatAvailability(BaseModel):
    seat_number: str
    is_available: bool
    workspace_id: str
    status_name: str

class ChatResponse(BaseModel):
    item: str
    status: bool

class BookingResult(BaseModel):
    success: bool
    message: str
    booking_details: Optional[Dict[str, str]] = None

class DayBookingStatus(BaseModel):
    date: datetime
    success: bool
    message: str
    formatted_date: Optional[str] = None
    workspace_id: Optional[str] = None
    details: Optional[Dict[str, str]] = None

class BookingHistoryItem(BaseModel):
    seat: str
    floor: str
    building: str
    location: str
    time: str
    bookType: int
    unitId: int
    typeOfUnit: int
    svgId: str
    status: int
    isAvailable: bool
    checkInTime: Optional[str] = None
    allocationID: int
    fromDate: str

class CancelResult(BaseModel):
    success: bool
    message: str
    allocation_id: Optional[int] = None

# --- Constants ---
LOCATION_FIELDS = ["building_no", "seat_number", "floor"]
BOOKING_INFO_FIELDS = ["booking_days_description"]
# Fields for specific cancellation intent
CANCEL_INFO_FIELDS = ["cancel_seat_number", "cancel_date_description"]
# Fields for view history intent
VIEW_HISTORY_FIELDS = ["history_count"]
RESET_COMMANDS = ["clear", "reset", "start over", "start", "new booking"]
GREETING_COMMANDS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

# --- Date Helper Functions (assumed correct from provided code) ---
def format_dates_for_display(dates: List[Any]) -> str:
    if not dates: return "No dates specified"
    
    # Handle both datetime objects and ISO date strings
    date_objects = []
    for d in dates:
        if isinstance(d, str):
            try:
                date_objects.append(datetime.fromisoformat(d.split('T')[0]))
            except ValueError:
                continue # Skip if string is not a valid date format
        elif isinstance(d, datetime):
            date_objects.append(d)

    if not date_objects: return "Invalid date format provided"

    date_objects = sorted(list(set(date_objects)))
    display_format_dates = [d.strftime('%B %d, %Y') for d in date_objects]
    if len(date_objects) == 1: return f"Single day: {display_format_dates[0]}"
    return f"Multiple days ({len(date_objects)}): {', '.join(display_format_dates)}"

def format_date_for_table(date_obj: datetime) -> str:
    return date_obj.strftime('%d.%b.%Y')

def parse_booking_days(booking_days_description: str) -> List[datetime]:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    result_dates = []
    description = booking_days_description.lower().strip()

    # today and next X days
    today_and_next_pattern = re.search(r"today and next (\d+) days?", description)
    if today_and_next_pattern:
        days_count = int(today_and_next_pattern.group(1))
        result_dates.append(today)
        for i in range(1, days_count + 1): result_dates.append(today + timedelta(days=i))
        return result_dates

    # next X days (implicitly includes today if today is a weekday and not past)
    next_days_pattern = re.search(r"next (\d+) days?", description)
    if next_days_pattern:
        days_count = int(next_days_pattern.group(1))
        # Start from today
        for i in range(days_count): result_dates.append(today + timedelta(days=i))
        return result_dates
    
    if "this week" in description:
        start_of_week = today - timedelta(days=today.weekday()) # Monday of current week
        for i in range(5): # Monday to Friday
            day_in_week = start_of_week + timedelta(days=i)
            if day_in_week >= today: # Only include today or future days
                result_dates.append(day_in_week)
        if result_dates: return sorted(list(set(result_dates)))

    if "next week" in description:
        start_of_next_week = today - timedelta(days=today.weekday()) + timedelta(weeks=1) # Monday of next week
        for i in range(5): # Monday to Friday
            result_dates.append(start_of_next_week + timedelta(days=i))
        return sorted(list(set(result_dates)))

    day_mapping = {
        "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6
    }
    # Day range e.g. "Monday to Wednesday"
    day_range_pattern = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\s+(?:to|-)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)", description)
    if day_range_pattern:
        start_day_name, end_day_name = day_range_pattern.group(1), day_range_pattern.group(2)
        start_day, end_day = day_mapping.get(start_day_name), day_mapping.get(end_day_name)

        if start_day is not None and end_day is not None:
            days_until_start = (start_day - today.weekday() + 7) % 7
            start_date = today + timedelta(days=days_until_start)
            
            current_loop_day = start_day
            day_offset_count = 0
            while True:
                result_dates.append(start_date + timedelta(days=day_offset_count))
                if current_loop_day == end_day:
                    break
                current_loop_day = (current_loop_day + 1) % 7
                day_offset_count += 1
                if day_offset_count > 14: break # Safety break for long/invalid ranges
            return result_dates if result_dates else [] # fallback to empty list if range parsing yields nothing

    # Specific dates "15th July", "July 15", "15th of July"
    current_year = today.year
    current_month = today.month
    date_pattern = r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    month_mapping = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
                     "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
                     "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
    
    found_specific_dates = False
    reference_month = None

    matches = list(re.finditer(date_pattern, description))
    for match in matches:
        day = int(match.group(1))
        month_name = match.group(2)
        month = month_mapping.get(month_name.lower())
        if month and 1 <= day <= 31:
            target_year = current_year
            if month < current_month or (month == current_month and day < today.day):
                target_year = current_year + 1
            
            try:
                date_obj = datetime(year=target_year, month=month, day=day)
                if date_obj not in result_dates: result_dates.append(date_obj)
                reference_month = month
                found_specific_dates = True
            except ValueError:
                pass
    
    if not found_specific_dates or (found_specific_dates and re.search(r",\s*(\d{1,2})(?:st|nd|rd|th)?(?!\s+(?:of\s+)?(?:jan|feb|...))", description)):
        effective_month = reference_month if reference_month else current_month

        for day_match in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", description):
            is_part_of_full_date = False
            for m in matches:
                if m.start() <= day_match.start() and m.end() >= day_match.end():
                    is_part_of_full_date = True
                    break
            if is_part_of_full_date:
                continue

            day = int(day_match.group(1))
            if 1 <= day <= 31:
                target_year, target_month = current_year, effective_month
                try: temp_date_obj = datetime(year=target_year, month=target_month, day=day)
                except ValueError: continue

                if temp_date_obj < today:
                    target_month +=1
                    if target_month > 12:
                        target_month, target_year = 1, target_year + 1
                
                try:
                    date_obj = datetime(year=target_year, month=target_month, day=day)
                    if date_obj not in result_dates: result_dates.append(date_obj)
                    found_specific_dates = True
                except ValueError:
                    pass
                        
    if found_specific_dates:
        return sorted(list(set(result_dates)))

    # Fallbacks if no other patterns matched
    if not result_dates:
        if "today" in description:
            result_dates.append(today)
        elif "tomorrow" in description:
            result_dates.append(today + timedelta(days=1))
        # FIX #1: REMOVED the final else block that defaulted to 'today', which caused random dates to populate.
        # Now, if no date is found, an empty list is returned.
            
    return sorted(list(set(result_dates)))

# --- API Helper Functions ---
async def get_new_access_token():
    url = "https://associ-connec-dev-flexi-webapp01.azurewebsites.net/connect/token"
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
            return response.json().get('access_token')
    except Exception as e:
        print(f"Error getting new access token: {str(e)}")
        return None

async def get_associate_info(access_token, employee_id):
    url = f"https://associ-connec-dev-flexi-webapp01.azurewebsites.net/api/flexi/GetAssociate?searchValue={employee_id}"
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, dict) and employee_id in user_data_store:
                user_data_store[employee_id]["collected_info"]["employee_name"] = data.get('flexibleUserName')
                user_data_store[employee_id]["collected_info"]["employee_email"] = data.get('email')
            return data
    except Exception as e:
        print(f"Error getting associate info: {str(e)}")
        return None

async def get_floor_booking_info(access_token: str, unit_id: str, date_to_check: datetime):
    formatted_date_for_api = date_to_check.strftime("%Y%%2F%m%%2F%d")
    url = f"https://associ-connec-dev-flexi-webapp01.azurewebsites.net/api/Flexi/GetBookingForWeb4Days?unitId={unit_id}&typeID=5&dateCheck={formatted_date_for_api}"
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error getting booking info for unit {unit_id} on {date_to_check.strftime('%Y-%m-%d')}: {str(e)}")
        return None

async def book_seat(access_token, employee_id, workspace_id, booking_date: Optional[datetime] = None):
    url = "https://associ-connec-dev-flexi-webapp01.azurewebsites.net/api/flexi/book4Days"
    target_date = booking_date if booking_date else datetime.now()
    from_time = target_date.replace(hour=8, minute=0, second=0, microsecond=0)
    to_time = target_date.replace(hour=20, minute=0, second=0, microsecond=0)
    from_date_str, to_date_str = from_time.isoformat(), to_time.isoformat()

    associate_info_data = user_data_store.get(employee_id, {}).get("collected_info", {})
    associate_name = associate_info_data.get("employee_name", "AI Bot User")
    associate_email = associate_info_data.get("employee_email", "")

    temp_associate_info = await get_associate_info(access_token, employee_id)
    if not temp_associate_info:
         return BookingResult(success=False, message="Failed to retrieve associate information for booking.")
    associate_id_val = temp_associate_info.get('associateId', 0)
    if not associate_id_val:
        return BookingResult(success=False, message="Associate ID not found. Cannot proceed with booking.")

    payload = {
        "allocationMode": 5, "associateId": associate_id_val, "bookType": 1, "createdBy": "AI bot",
        "Email": associate_email, "employeeNumber": int(employee_id), "exceptionMessage": "", "from": "",
        "fromDate": from_date_str, "isMovingHere": False, "isValid": False, "modifiedBy": "AI bot",
        "remark": "seat booked by AI", "selectedBusinessUnitCode": "", "selectedBusinessUnitId": 0,
        "selectedDepartmentCode": "", "selectedDepartmentId": 0, "selectedSectionCode": "",
        "selectedSectionId": 0, "to": "", "toDate": to_date_str, "toName": associate_name,
        "workspaceId": int(workspace_id)
    }
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            booking_response = response.json()
            if booking_response.get('isValid', False):
                return BookingResult(success=True, message="Seat booked successfully",
                    booking_details={"from_date": from_date_str, "to_date": to_date_str, "workspace_id": workspace_id,
                                     "employee_id": employee_id, "booked_date": target_date.strftime('%Y-%m-%d')})
            else:
                return BookingResult(success=False, message=f"Failed to book seat: {booking_response.get('exceptionMessage', 'Unknown error')}")
    except Exception as e:
        return BookingResult(success=False, message=f"Error booking seat: {str(e)}")

async def get_booking_history(access_token: str, employee_id: str) -> List[BookingHistoryItem]:
    url = "https://associ-connec-dev-flexi-webapp01.azurewebsites.net/api/flexi/GetBookingHistory"
    payload = {"keySearch": employee_id, "pageIndex": 1, "pageSize": 50}
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            history_data = response.json()
            return [BookingHistoryItem(**item) for item in history_data]
    except Exception as e:
        print(f"Error getting booking history for employee {employee_id}: {str(e)}")
        return []

async def cancel_booking_api(access_token: str, allocation_id: int) -> CancelResult:
    url = f"https://associ-connec-dev-flexi-webapp01.azurewebsites.net/api/flexi/Cancel?allocationId={allocation_id}"
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

def extract_unit_id(associate_info, building_no, floor):
    if not associate_info or 'availableSeatList' not in associate_info:
        return None
    # FIX #3: Added cardinal number words to the mapping dictionary.
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

    for floor_info in associate_info['availableSeatList']:
        floor_name = floor_info['floor'].lower()
        if any(format.lower() == floor_name for format in possible_formats):
            print(f"Found matching floor: {floor_info['floor']}")
            if building_no in floor_info:
                value = floor_info[building_no]
                match = re.search(r"/space/flex/\d+/(\d+)\?", value)
                if match: return match.group(1)
    for floor_info in associate_info['availableSeatList']:
        floor_digits = re.search(r'\d+', floor_info['floor'])
        if floor_digits and floor_digits.group(0) == normalized_floor:
            print(f"Found floor by number match: {floor_info['floor']}")
            if building_no in floor_info:
                value = floor_info[building_no]
                match = re.search(r"/space/flex/\d+/(\d+)\?", value)
                if match: return match.group(1)
    return None

def check_seat_availability(booking_info, seat_number):
    if not booking_info or 'asscocicateInfoList' not in booking_info: return None
    normalized_seat = seat_number.upper().strip()
    for workspace in booking_info['asscocicateInfoList']:
        if workspace.get('worspaceName', '').upper().strip() == normalized_seat:
            return SeatAvailability(seat_number=workspace['worspaceName'], is_available=workspace.get('workspaceStatusId') == 1,
                                    workspace_id=str(workspace.get('workspaceId', '')), status_name=workspace.get('statusName', ''))
    return SeatAvailability(seat_number=seat_number, is_available=False, workspace_id="", status_name="Seat not found")

def parse_dates_fallback(message: str, today_iso_date: str) -> List[str]:
    """
    Fallback date parser for common patterns when LLM fails
    """
    dates = []
    today_date = datetime.strptime(today_iso_date, "%Y-%m-%d")
    current_year = today_date.year
    
    message_lower = message.lower()
    
    date_range_pattern = r'(\d{1,2})(?:th|st|nd|rd)?\s+(\w+)\s*[-–—to]\s*(\d{1,2})(?:th|st|nd|rd)?\s+(\w+)'
    range_match = re.search(date_range_pattern, message_lower)
    
    if range_match:
        start_day, start_month, end_day, end_month = range_match.groups()
        try:
            start_date = parse_date_string(f"{start_day} {start_month} {current_year}")
            end_date = parse_date_string(f"{end_day} {end_month} {current_year}")
            
            if start_date and end_date:
                current_date = start_date
                while current_date <= end_date:
                    dates.append(current_date.strftime("%Y-%m-%d"))
                    current_date += timedelta(days=1)
                return dates
        except:
            pass
    
    single_month_range = r'(\d{1,2})(?:th|st|nd|rd)?\s*[-–—to]\s*(\d{1,2})(?:th|st|nd|rd)?\s+(\w+)'
    single_match = re.search(single_month_range, message_lower)
    
    if single_match:
        start_day, end_day, month = single_match.groups()
        try:
            start_date = parse_date_string(f"{start_day} {month} {current_year}")
            end_date = parse_date_string(f"{end_day} {month} {current_year}")
            
            if start_date and end_date:
                current_date = start_date
                while current_date <= end_date:
                    dates.append(current_date.strftime("%Y-%m-%d"))
                    current_date += timedelta(days=1)
                return dates
        except:
            pass
    
    if "today and tomorrow" in message_lower:
        dates.append(today_iso_date)
        dates.append((today_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    elif "tomorrow and day after" in message_lower:
        dates.append((today_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        dates.append((today_date + timedelta(days=2)).strftime("%Y-%m-%d"))
    elif "next" in message_lower and "day" in message_lower:
        numbers = re.findall(r'\d+', message_lower)
        if numbers:
            num_days = int(numbers[0])
            for i in range(num_days):
                dates.append((today_date + timedelta(days=i)).strftime("%Y-%m-%d"))
    elif "today" in message_lower and "tomorrow" not in message_lower:
        dates.append(today_iso_date)
    elif "tomorrow" in message_lower and "today" not in message_lower:
        dates.append((today_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    
    return dates

def parse_date_string(date_str: str) -> Optional[datetime]:
    from dateutil import parser
    try:
        return parser.parse(date_str)
    except:
        months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        parts = date_str.lower().split()
        if len(parts) >= 3:
            try:
                day_str = re.sub(r'(st|nd|rd|th)', '', parts[0])
                day = int(day_str)
                month_str = parts[1]
                year = int(parts[2])
                month = months.get(month_str)
                if month:
                    return datetime(year, month, day)
            except (ValueError, IndexError):
                pass
        return None

def format_booking_results_table(booking_results_list: List[DayBookingStatus], collected_info: Dict[str, Any]) -> str:
    table_header = "| Date | Building | Floor | Seat Number | Status |\n"
    table_separator = "|------|----------|-------|-------------|--------|\n"
    table_rows = []
    for result in booking_results_list:
        date_formatted = result.date.strftime("%d.%b.%Y")
        status_message = f"✅ Booked" if result.success else f"❌ {result.message}"
        row = f"| {date_formatted} | {collected_info.get('building_no', 'N/A')} | {collected_info.get('floor', 'N/A')} | {collected_info.get('seat_number', 'N/A')} | {status_message} |"
        table_rows.append(row)

    confirmation_table = table_header + table_separator + "\n".join(table_rows)
    user_details_parts = ["\n"]
    if collected_info.get("employee_name"): user_details_parts.append(f"**Employee:** {collected_info['employee_name']}")
    if collected_info.get("employee_email"): user_details_parts.append(f"({collected_info['employee_email']})")
    return confirmation_table + " ".join(user_details_parts)

async def process_multi_date_booking(employee_id: str, dates_list: List[str], collected_info: Dict[str, Any]):
    # Convert ISO strings to datetime objects
    dates_to_book_dt = [datetime.fromisoformat(d) for d in dates_list]

    multi_day_results: List[DayBookingStatus] = []
    access_token = await get_new_access_token()
    if not access_token:
        for date_obj in dates_to_book_dt: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="System error: Auth token failed."))
        return {"success": False, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info), "message": "System error: Could not get access token."}

    if not collected_info.get("employee_name") or not collected_info.get("employee_email"):
        await get_associate_info(access_token, employee_id)

    associate_api_info = user_data_store[employee_id].get("associate_api_data")
    if not associate_api_info:
        associate_api_info = await get_associate_info(access_token, employee_id)
        if associate_api_info:
            user_data_store[employee_id]["associate_api_data"] = associate_api_info
        else:
            for date_obj in dates_to_book_dt: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="System error: Failed to get associate info."))
            return {"success": False, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info), "message": "Failed to retrieve associate information."}
    
    unit_id = extract_unit_id(associate_api_info, collected_info["building_no"], collected_info["floor"])
    if not unit_id:
        for date_obj in dates_to_book_dt: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="Invalid Building/Floor or unit ID not found."))
        return {"success": False, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info), "message": "Invalid Building/Floor or unit ID not found."}

    seat_num_to_check = collected_info["seat_number"]

    if seat_num_to_check and len(seat_num_to_check) > 2 and "-" not in seat_num_to_check:
        formatted_seat_num = f"{seat_num_to_check[:2]}-{seat_num_to_check[2:]}"
        print(f"Reformatted seat number from {seat_num_to_check} to {formatted_seat_num}")
        seat_num_to_check = formatted_seat_num
        collected_info["seat_number"] = seat_num_to_check

    for date_obj in dates_to_book_dt:
        floor_booking_data = await get_floor_booking_info(access_token, unit_id, date_obj)
        if not floor_booking_data:
            multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="Failed to get availability for this date."))
            continue
        seat_availability = check_seat_availability(floor_booking_data, seat_num_to_check)
        if not seat_availability or not seat_availability.is_available:
            multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message=f"Seat not available ({seat_availability.status_name if seat_availability else 'Not found'})."))
            continue
        if seat_availability.workspace_id:
            booking_api_result = await book_seat(access_token, employee_id, seat_availability.workspace_id, date_obj)
            multi_day_results.append(DayBookingStatus(date=date_obj, success=booking_api_result.success, message=booking_api_result.message, details=booking_api_result.booking_details, workspace_id=seat_availability.workspace_id))
        else:
            multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="Workspace ID not found for booking."))
    overall_success = any(res.success for res in multi_day_results)
    return {"success": overall_success, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info)}

def get_status_name(status_code: int) -> str:
    status_map = {
        4: "Booked",  
        1: "Cancelled", 
    }
    return status_map.get(status_code, f"Unknown (Status {status_code})")

def filter_and_sort_cancellable_bookings(booking_history: List[BookingHistoryItem]) -> List[BookingHistoryItem]:
    cancellable: List[BookingHistoryItem] = []
    today = datetime.now().date()
    for booking in booking_history:
        try:
            from_date_dt = datetime.fromisoformat(booking.fromDate)
            if booking.status == 4 and from_date_dt.date() >= today:
                cancellable.append(booking)
        except Exception as e:
            print(f"Skipping booking item {booking.allocationID} due to parsing/filter error: {e}")
            continue
    cancellable.sort(key=lambda b: datetime.fromisoformat(b.fromDate))
    return cancellable

def generate_initial_greeting(employee_id: str) -> str:
    user_name = user_data_store.get(employee_id, {}).get("collected_info", {}).get("employee_name")
    greeting = f"Hello{(' ' + user_name) if user_name else ''}! "
    greeting += "I can help you book seat (I'll need the days, building, floor, and seat number), cancel an existing booking (I'll need the seat number and specific date), or view your booking history. Please specify what you'd like to do?"
    return greeting

def clear_user_flow_state(employee_id: str, intent_to_clear: Optional[str] = None):
    if employee_id not in user_data_store:
        return
    
    fields_to_pop = [
        "intent", "cancellation_step", "selected_allocation_id_for_cancel",
        "cancellable_bookings_cache", "awaiting_booking_confirmation",
        "booking_dates_iso", "offered_tomorrow", "offered_next_week",
        "specific_booking_details_for_cancel",
        "llm_parsed_dates_iso_list", "llm_parsed_cancel_date_iso", "selected_allocation_id"
    ]
    
    booking_related_fields = LOCATION_FIELDS + BOOKING_INFO_FIELDS
    cancel_related_fields = CANCEL_INFO_FIELDS
    view_history_related_fields = VIEW_HISTORY_FIELDS

    if intent_to_clear == "book_seat":
        fields_to_pop.extend(booking_related_fields)
    elif intent_to_clear == "cancel_seat":
        fields_to_pop.extend(cancel_related_fields)
    elif intent_to_clear == "view_booking_history":
        fields_to_pop.extend(view_history_related_fields)
    else: # Full reset
        fields_to_pop.extend(booking_related_fields)
        fields_to_pop.extend(cancel_related_fields)
        fields_to_pop.extend(view_history_related_fields)

    for field in set(fields_to_pop):
        user_data_store[employee_id]["collected_info"].pop(field, None)

async def get_llm_response_for_booking(message: str, conversation_history: List[Dict[str, str]], collected_info: Dict[str, Any]):
    system_prompt_parts = ["You are a helpful assistant for Bosch seat booking."]
    missing_fields_desc = []
    
    required_booking_fields_map = {
        "booking_days_description": "the day(s) for the booking (e.g., 'today', 'next Monday')",
        "building_no": "the building number (e.g., '903')",
        "floor": "the floor (e.g., '1st floor')",
        "seat_number": "the seat number (e.g., 'L1-001')"
    }
    
    for field_key, field_desc in required_booking_fields_map.items():
        if field_key not in collected_info or not collected_info[field_key]:
            missing_fields_desc.append(field_desc)
    
    if missing_fields_desc:
        system_prompt_parts.append(f"Ask the user for the first missing detail needed for a booking, which is: {missing_fields_desc[0]}.")
    else:
        # This case is no longer used, as the direct LLM ack is removed.
        system_prompt_parts.append("All required information for booking has been collected. Acknowledge this briefly and state that a summary will be shown.")
    
    system_prompt_parts.extend([
        "Your goal is to collect information for seat booking: booking days, building number, floor, and seat number.",
        "Ask only one question at a time. Be friendly but efficient.",
        "Do NOT ask for employee name or email; this is handled automatically.",
        "Do NOT ask for the booking time. All bookings are for standard work hours.",
        "Do NOT ask for confirmation to book. The system will show a summary for confirmation.",
        "Do NOT use phrases like 'please hold on', 'wait a moment', or similar. Respond directly.",
        "If you do not understand the user's input, ask them to clarify or rephrase."
    ])
    system_content = "\n".join(system_prompt_parts)
    messages = [SystemMessage(content=system_content)]
    for msg in conversation_history: messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message))
    try:
        response = llm.invoke(messages)    
        return response.content
    except Exception as e:
        print(f"LLM error in get_llm_response_for_booking: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Azure OpenAI: {str(e)}")

def normalize_seat_number(seat_number: str) -> str:
    return seat_number.replace("-", "").lower()

def seats_match(seat1: str, seat2: str) -> bool:
    return normalize_seat_number(seat1) == normalize_seat_number(seat2)

def get_allowed_buildings(associate_info: Dict[str, Any]) -> List[str]:
    if not associate_info or 'availableSeatList' not in associate_info:
        return []

    allowed_buildings = set()
    
    for floor_info in associate_info.get('availableSeatList', []):
        for key in floor_info.keys():
            if key.lower() in ['floor', 'iscob']:
                continue    
            
            if key.isdigit():
                allowed_buildings.add(key)
            else:
                match = re.search(r'\d+', key)
                if match:
                    allowed_buildings.add(match.group(0))
    return sorted(list(allowed_buildings))

def get_allowed_locations(associate_info: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract allowed location information from associate data.
    Returns a dictionary with location details for validation.
    """
    if not associate_info:
        return {}
    
    location_info = {
        'office_location_code': associate_info.get('officeLocationCode', '').upper(),
        'office_location_name': associate_info.get('officeLocationName', '').lower(),
        'location_name': associate_info.get('locationName', '').lower()
    }
    
    return location_info

def validate_user_location(user_location: str, associate_info: Dict[str, Any]) -> bool:
    """
    Validate if user provided location matches their authorized location.
    Returns True if valid, False otherwise.
    """
    if not user_location or not associate_info:
        return False
    
    allowed_locations = get_allowed_locations(associate_info)
    user_location_lower = user_location.lower().strip()
    
    # Check against all location fields
    return (
        user_location_lower == allowed_locations.get('office_location_name', '') or
        user_location_lower == allowed_locations.get('location_name', '') or
        user_location.upper() == allowed_locations.get('office_location_code', '')
    )

async def extract_info_with_llm(message: str, current_info: Dict[str, Any], today_iso_date: str) -> Dict[str, Any]:
    today_date = datetime.strptime(today_iso_date, "%Y-%m-%d")
    current_weekday = today_date.strftime('%A')

    llm_context_info = {
        k: v for k, v in current_info.items()
        if k not in ['cancellable_bookings_cache', 'booking_history_cache', 'associate_api_data', 'employee_name', 'employee_email']
    }
    system_prompt = f"""
    You are an expert AI assistant for a seat booking system. Your task is to analyze a user's message and conversation context, then output a single, valid JSON object.

    The JSON object must have two keys: "intent" and "parameters".
    Today's date is: {today_iso_date} ({current_weekday}).

    --- INTENTS ---
    - 'book_seat': User wants to reserve a seat (even if details are incomplete).
    - 'cancel_seat': User wants to cancel a booking (even if details are incomplete).
    - 'view_booking_history': User wants to see their booking history.
    - 'general_query': The request is completely unrelated to booking, canceling, or viewing history, or is just a greeting.

    --- PARAMETERS (by Intent) ---
    1. For 'book_seat':
   - 'booking_days_description': The user's raw text for dates (e.g., 'next 3 days', 'July 15th', 'next monday').
   - 'llm_parsed_dates_iso_list': A list of all dates in 'YYYY-MM-DD' format. Be very careful with date parsing:
     * "next monday" = the upcoming Monday (if today is Monday, then next Monday is 7 days away)
     * "tomorrow" = {(today_date + timedelta(days=1)).strftime('%Y-%m-%d')}
     * "today" = {today_iso_date}
     * "next week" = all weekdays (Mon-Fri) of next week starting from {(today_date + timedelta(days=7-today_date.weekday())).strftime('%Y-%m-%d')}
     * "this week" = remaining weekdays of the current week (including today if applicable)
     * Date ranges: "June 9-11" becomes ["{today_date.year}-06-09", "{today_date.year}-06-10", "{today_date.year}-06-11"]
     * Always assume current year unless specified otherwise
   - 'building_no': Building number (e.g., '903', from 'Kor903' extract '903').
   - 'floor': Floor description (e.g., 'floor one','2nd floor', '1st floor'). Normalize to consistent format if possible (e.g. '1st floor').
   - 'seat_number': Seat identifier, normalized to uppercase (e.g., 'L1A108', 'L1-001').
   - 'Location': associte officeLocationCode(KOR),officeLocationName(Koramangala),locationName(Bengaluru) (e.g., 'Bengaluru','KOR','Koramangala','Hyderabad')

    2. For 'cancel_seat':
   - 'cancel_seat_number': The seat to cancel, normalized to uppercase.
   - 'cancel_date_description': The user's raw text for the cancellation date.
   - 'llm_parsed_cancel_date_iso': The single cancellation date in 'YYYY-MM-DD' format.

    3. For 'view_booking_history':
   - No specific parameters needed

    --- RULES ---
    - IMPORTANT: Recognize the intent even if booking/cancellation details are incomplete.
    - Only extract parameters from the user's LATEST message. Do not carry over parameters from the conversational state unless they are being re-confirmed.
    - If a parameter is not mentioned in the latest message, do not include its key in the 'parameters' object.
    - Use 'general_query' only for messages completely unrelated to seat booking operations.
    - For building numbers: KOR903, Kor903, aud606, kormangala 903, hydrabad 888, cob432 → extract only the numeric part (903, 606,888,432).
    - For floor information: "floor one" or "first floor" should be extracted as "1st floor". "ground floor" as "ground floor".
    - Be very precise with date calculations. Consider the current day of the week when parsing relative dates.

    --- EXAMPLES ---
    User message: "book a seat for next monday in hydrabad 606 1st floor L1A108"
    Your output:
    {{
  "intent": "book_seat",
  "parameters": {{
    "booking_days_description": "next monday",
    "llm_parsed_dates_iso_list": ["{(today_date + timedelta(days=(7 - today_date.weekday()) % 7 or 7)).strftime('%Y-%m-%d')}"],
    "building_no": "606",
    "floor": "1st floor",
    "seat_number": "L1A108",
    "location":"hydrabad"
  }}
    }}

    User message: "book for tomorrow"
    Your output:
    {{
  "intent": "book_seat",
  "parameters": {{
    "booking_days_description": "tomorrow",
    "llm_parsed_dates_iso_list": ["{(today_date + timedelta(days=1)).strftime('%Y-%m-%d')}"]
  }}
    }}
    """
    human_prompt_content = f"""
    Analyze the following information and provide the JSON output as specified in your instructions.

    - Current conversational state: {json.dumps(llm_context_info)}
    - User's latest message: "{message}"
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt_content)
        ]
        response_llm = llm.invoke(messages)
        extracted_json_str = response_llm.content.strip()
        
        match_json = re.search(r"\{.*\}", extracted_json_str, re.DOTALL)
        if match_json:
            extracted_json_str = match_json.group(0)

        llm_output = json.loads(extracted_json_str)
        print(f"LLM Raw Output: {llm_output}")

        final_output = {}
        final_output["intent"] = llm_output.get("intent", "general_query")
        
        if "parameters" in llm_output and isinstance(llm_output["parameters"], dict):
            final_output.update(llm_output["parameters"])

        print(f"LLM Extracted and Processed Info: {final_output}")
        return final_output

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in extract_info_with_llm: {e}. Raw LLM response: {extracted_json_str}")
        return {"intent": "general_query"}
    except Exception as e:
        print(f"General Error in extract_info_with_llm: {e}")
        return {"intent": "general_query"}


# Token searvive
# Validate Token API
def validate_token(Authorization: str, client_id: str = "CD3054C5-6D98-47E9-BF73-43F26E8ED476") -> dict:
    """
    Sends a GET request to validate the token.

    Args:
        token (str): The token to be validated.
        client_id (str): The client ID to authenticate the request.

    Returns:
        dict: Parsed JSON response from the API.
    """
    url = "https://dev.boschassociatearena.com/api/Token/ValidateToken"
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

# Decyrpt Message

def decrypt_text(text_to_decrypt: str, client_id: str = "CD3054C5-6D98-47E9-BF73-43F26E8ED476") -> str:
    """
    Sends a GET request to decrypt the given text.

    Args:
        text_to_decrypt (str): The encrypted text.
        client_id (str): The client ID to be sent in the request header.

    Returns:
        str: Decrypted text if successful, else an error message.
    """
    url = "https://dev.boschassociatearena.com/api/Token/DecryptClientData"
    
    # Parameters sent in the URL
    params = {
        "TextToDecrypt": text_to_decrypt
    }

    # Headers including the client ID
    headers = {
        "clientID": client_id
    }

    try:
        response = requests.get(url, params=params, headers=headers, verify=False)
        response.raise_for_status()
        return response.text  # Assuming the response is plain text
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"


def parse_user_data(decrypted_data: str) -> dict:
    """
    Parse the decrypted user data string into a dictionary
    Expected format: "35017285,Kasireddy Sri Vaishnavi,KASC1KOR"
    """
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

from fastapi import FastAPI, Body, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
security = HTTPBearer()

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    Session_id: Optional[str] = Header(None, alias="Session_id")
):
    message_text = request.EmployeeQueryMessage.strip()
    
    authorization = credentials.credentials  # This gives you the token
    print(f"Token: {authorization}")
    print(f"Session ID: {Session_id}")

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization is required"
        )
    
    try:
        # Step 1: Validate Authorization and get encrypted data
        encrypted_data = validate_token(authorization)
        print(f"Encrypted data: {encrypted_data}")

        response_data = encrypted_data.get('ResponseData', [])
        if response_data and isinstance(response_data, list):
           newdata = response_data[0]
           print(f"New data: {newdata}")
        else:
           print("ResponseData is missing or empty")
           newdata = None
        
        # Step 2: Decrypt the client data
        decrypted_data =  decrypt_text(newdata)
        print(f"Decrypted data: {newdata}")
        
        # Step 3: Parse user information
        user_info = parse_user_data(decrypted_data)
        print(f"User info: {user_info}")
        
        # Now you can use the user_info in your chat logic
        # For example:
        employee_id = user_info["employee_id"]
        employee_name = user_info["employee_name"]
        employee_code = user_info["employee_code"]
    except HTTPException:
        raise HTTPException(
            status_code=401,
            detail="Authorization is required"
        )
    # employee_id = request.employee_id
    today_iso_date_str = datetime.now().strftime("%Y-%m-%d")
    
    if employee_id not in user_data_store:
        user_data_store[employee_id] = {"conversation_history": [], "collected_info": {}, "associate_api_data": None}
        access_token = await get_new_access_token()
        if access_token:
            api_data = await get_associate_info(access_token, employee_id)
            if api_data: user_data_store[employee_id]["associate_api_data"] = api_data

    current_conversation_data = user_data_store[employee_id]
    history = current_conversation_data["conversation_history"]
    collected_info = current_conversation_data["collected_info"]

    final_bot_response = ""
    is_flow_complete_for_response = False

    # --- 1. Initial Reset/Greeting Handling ---
    if message_text.lower() in RESET_COMMANDS:
        clear_user_flow_state(employee_id, None) 
        history.clear() 
        final_bot_response = generate_initial_greeting(employee_id)
        is_flow_complete_for_response = True 
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": final_bot_response})
        return  ChatResponse(item=final_bot_response, status=is_flow_complete_for_response)

    if message_text.lower() in GREETING_COMMANDS and not collected_info.get("intent"):
        final_bot_response = generate_initial_greeting(employee_id)
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(item=final_bot_response, status=is_flow_complete_for_response)

    # --- 2. Log User Message and Extract Intent & Parameters via LLM ---
    history.append({"role": "user", "content": message_text})
    
    previous_intent_before_llm = collected_info.get("intent")
    llm_context = {k: v for k, v in collected_info.items() if k not in ['cancellable_bookings_cache', 'booking_history_cache', 'associate_api_data']}
    
    extracted_llm_data = await extract_info_with_llm(message_text, llm_context, today_iso_date_str)
    
    newly_extracted_intent = extracted_llm_data.get("intent")

    # --- 3. Handle Intent Change (Interrupt Flow if Necessary) ---
    if newly_extracted_intent and newly_extracted_intent != previous_intent_before_llm:
        print(f"Intent changed from '{previous_intent_before_llm}' to '{newly_extracted_intent}'. Clearing state for '{previous_intent_before_llm}'.")
        clear_user_flow_state(employee_id, previous_intent_before_llm)
    
    collected_info.update(extracted_llm_data)
    current_intent = collected_info.get("intent") 

    user_confirms = any(w in message_text.lower() for w in ["yes", "yep", "yeah", "confirm", "proceed", "ok", "sure", "do it"])
    user_declines = any(w in message_text.lower() for w in ["no", "nope", "cancel", "stop", "don't", "do not", "never mind"])

    # --- 4. Process Confirmation Steps ---
    if collected_info.get("cancellation_step") == "awaiting_confirmation" and current_intent == "cancel_seat":
        if user_confirms:
            allocation_id = collected_info.get("selected_allocation_id")
            access_token = await get_new_access_token() 
            if not access_token:
                final_bot_response = "Sorry, I couldn't get authorization. Please try again later."
            elif allocation_id:
                # FIX #4.3: Corrected cancellation response logic
                cancel_result = await cancel_booking_api(access_token, allocation_id)
                if cancel_result.success:
                    final_bot_response = "Your booking has been successfully cancelled."
                else:
                    final_bot_response = f"Sorry, I was unable to cancel the booking. Reason: {cancel_result.message}"
            else:
                final_bot_response = "Error: No booking was selected for cancellation."
            is_flow_complete_for_response = True
            clear_user_flow_state(employee_id, "cancel_seat")
        elif user_declines:
            final_bot_response = "Okay, the booking will not be cancelled. Anything else?"
            is_flow_complete_for_response = True
            clear_user_flow_state(employee_id, "cancel_seat")
        else: 
            final_bot_response = "I didn't quite understand. Please confirm with 'yes' or 'no' to cancel this booking."
        
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(item=final_bot_response, status=is_flow_complete_for_response)

    elif collected_info.get("awaiting_booking_confirmation") and current_intent == "book_seat":
        if user_confirms:
            dates_to_book_iso = collected_info.get("llm_parsed_dates_iso_list", [])
            today = datetime.now().date()
            valid_dates_iso, ignored_dates_count = [], 0

            for date_str in dates_to_book_iso:
                try:
                    if datetime.fromisoformat(date_str.split('T')[0]).date() >= today:
                        valid_dates_iso.append(date_str)
                    else:
                        ignored_dates_count += 1
                except (ValueError, TypeError):
                    ignored_dates_count += 1
            
            warning_message = ""
            if ignored_dates_count > 0:
                plural_s = "s" if ignored_dates_count > 1 else ""
                warning_message = f"Please note: I cannot book seats for past dates. {ignored_dates_count} requested date{plural_s} were ignored.\n\n"

            if valid_dates_iso:
                booking_process_result = await process_multi_date_booking(employee_id, valid_dates_iso, collected_info)
                if booking_process_result["success"]: 
                    final_bot_response = warning_message + f"Your booking request for the valid dates has been processed.\n\n{booking_process_result['confirmation_table']}"
                else:
                    final_bot_response = warning_message + f"Your booking request for the valid dates was processed, but encountered issues:\n\n{booking_process_result['confirmation_table']}"
                # FIX #4.2: Removed history.clear() to preserve conversation context.
            else:
                final_bot_response = warning_message + "There were no valid future dates to book. Please try the booking process again with a valid date (today or later)."
            
            is_flow_complete_for_response = True
            clear_user_flow_state(employee_id, "book_seat")

        elif user_declines:
            final_bot_response = "Okay, I will not proceed with the booking. Is there anything else?"
            is_flow_complete_for_response = True
            clear_user_flow_state(employee_id, "book_seat")
        else: 
            final_bot_response = "I'm sorry, I didn't catch that. Do you want to proceed with the booking? (yes/no)"

        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(item=final_bot_response, status=is_flow_complete_for_response)

    # --- 5. Main Intent Processing Logic ---
    if current_intent == "cancel_seat":
        access_token = await get_new_access_token()
        if not access_token:
            final_bot_response = "Sorry, system error getting authorization. Please try again."
        else:
            seat_number = collected_info.get("cancel_seat_number")
            parsed_cancel_date_iso = collected_info.get("llm_parsed_cancel_date_iso")
            
            if not seat_number:
                final_bot_response = "To cancel a booking, which seat number is it?"
            elif not parsed_cancel_date_iso:
                final_bot_response = f"And for which date do you want to cancel seat {seat_number}?"
            else:
                try:
                    target_date_obj = datetime.strptime(parsed_cancel_date_iso, "%Y-%m-%d").date()
                    all_bookings = await get_booking_history(access_token, employee_id)
                    found_booking = None
                    for booking in all_bookings:
                        try:
                            booking_date = datetime.fromisoformat(booking.fromDate).date()
                            if (seats_match(booking.seat, seat_number) and booking_date == target_date_obj and booking.status == 4 and booking_date >= datetime.now().date()):
                                found_booking = booking
                                break
                        except ValueError: continue 
                    
                    if found_booking:
                        final_bot_response = (f"I found a booking for seat {found_booking.seat} on "
                                            f"{target_date_obj.strftime('%B %d, %Y')} (Building: {found_booking.building}, Floor: {found_booking.floor}). "
                                            f"Are you sure you want to cancel it? (yes/no)")
                        collected_info["selected_allocation_id"] = found_booking.allocationID
                        collected_info["cancellation_step"] = "awaiting_confirmation"
                    else:
                        final_bot_response = (f"I couldn't find an active, cancellable booking for seat {seat_number} on "
                                            f"{target_date_obj.strftime('%B %d, %Y')}. "
                                            f"Please double-check the details.")
                        clear_user_flow_state(employee_id, "cancel_seat")
                except ValueError:
                    final_bot_response = "The date provided seems invalid. Please provide a clear date like 'tomorrow' or 'July 15th'."
                    collected_info.pop("llm_parsed_cancel_date_iso", None)
                    collected_info.pop("cancel_date_description", None)

    elif current_intent == "view_booking_history":
        access_token = await get_new_access_token()
        if not access_token:
            final_bot_response = "Sorry, system error getting authorization. Please try again."
        else:
            booking_hist_items = await get_booking_history(access_token, employee_id)
            if not booking_hist_items:
                final_bot_response = "You have no booking history."
            else:
                booking_hist_items.sort(key=lambda item: datetime.fromisoformat(item.time), reverse=True)
                history_count_to_show = 5
                
                history_to_show = booking_hist_items[:history_count_to_show]
                response_parts = [f"Here are your {len(history_to_show)} most recent bookings:\n"]
                for item in history_to_show:
                    dt_obj = datetime.fromisoformat(item.fromDate)
                    status_name = get_status_name(item.status)
                    response_parts.append(
                        f"- Seat: {item.seat}, For: {dt_obj.strftime('%d %b %Y')}, "
                        f"Building: {item.building}, Floor: {item.floor}, Status: {status_name}"
                    )
                final_bot_response = "\n".join(response_parts)
        is_flow_complete_for_response = True
        clear_user_flow_state(employee_id, "view_booking_history")

    elif current_intent == "book_seat":
        print(current_conversation_data.get("associate_api_data"))
        
        # FIX #2: Hardened building validation logic
        user_provided_building = collected_info.get("building_no")
        if user_provided_building:
            associate_api_info = current_conversation_data.get("associate_api_data")
            if not associate_api_info:
                access_token = await get_new_access_token()
                if access_token:
                    api_data = await get_associate_info(access_token, employee_id)
                    if api_data:
                        current_conversation_data["associate_api_data"] = api_data
                        associate_api_info = api_data
            
            if not associate_api_info:
                final_bot_response = "I'm having trouble accessing your profile to validate the building. Please try again later."
                clear_user_flow_state(employee_id, "book_seat")
                history.append({"role": "assistant", "content": final_bot_response})
                return ChatResponse(item=final_bot_response, status=True)

            allowed_buildings = get_allowed_buildings(associate_api_info)
            if not allowed_buildings:
                final_bot_response = "It seems there are no buildings assigned to your profile. Please contact support."
                clear_user_flow_state(employee_id, "book_seat")
                history.append({"role": "assistant", "content": final_bot_response})
                return ChatResponse(item=final_bot_response, status=True)
            
            if user_provided_building not in allowed_buildings:
                allowed_buildings_str = ", ".join(allowed_buildings)
                final_bot_response = (f"I'm sorry, but you are not authorized to book in Building {user_provided_building}. "
                                      f"Your authorized buildings are: {allowed_buildings_str}. Please provide a valid building number.")
                collected_info.pop("building_no", None)
                history.append({"role": "assistant", "content": final_bot_response})
                return ChatResponse(item=final_bot_response, status=False)
        
        #  Location validation.
        user_provided_location = collected_info.get("location") 
        if user_provided_location:
          associate_api_info = current_conversation_data.get("associate_api_data")
          if not associate_api_info:
              access_token = await get_new_access_token()
              if access_token:
                  api_data = await get_associate_info(access_token, employee_id)
                  if api_data:
                      current_conversation_data["associate_api_data"] = api_data
                      associate_api_info = api_data
        
          if not associate_api_info:
              final_bot_response = "I'm having trouble accessing your profile to validate the location. Please try again later."
              clear_user_flow_state(employee_id, "book_seat")
              history.append({"role": "assistant", "content": final_bot_response})
              return ChatResponse(item=final_bot_response, status=True)

          if not validate_user_location(user_provided_location, associate_api_info):
              allowed_locations = get_allowed_locations(associate_api_info)
              location_options = []
            
              if allowed_locations.get('office_location_name'):
                  location_options.append(allowed_locations['office_location_name'].title())
              if allowed_locations.get('location_name'):
                  location_options.append(allowed_locations['location_name'].title())
              if allowed_locations.get('office_location_code'):
                  location_options.append(allowed_locations['office_location_code'])
            
              allowed_locations_str = ", ".join(location_options)
              final_bot_response = (f"I'm sorry, but you are not authorized to book in '{user_provided_location}'. "
                                  f"Your authorized location is: {allowed_locations_str}. Please provide a valid location.")
              collected_info.pop("location", None)  # Clear invalid location
              history.append({"role": "assistant", "content": final_bot_response})
              return ChatResponse(item=final_bot_response, status=True)

        all_location_fields_collected = all(collected_info.get(f) for f in LOCATION_FIELDS)
        has_booking_days = "booking_days_description" in collected_info and collected_info["booking_days_description"]

        if all_location_fields_collected and has_booking_days:
            # FIX #4.1: Removed the redundant LLM 'ack' call that produced "hold on" messages.
            
            parsed_dates = []
            if "llm_parsed_dates_iso_list" in collected_info:
                parsed_dates = [datetime.fromisoformat(d) for d in collected_info["llm_parsed_dates_iso_list"]]
            if not parsed_dates:
                parsed_dates = parse_booking_days(collected_info.get("booking_days_description", ""))
            
            current_time_for_rules = datetime.now()
            valid_dates_for_preview = []
            for date_obj in parsed_dates:
                if date_obj.date() >= current_time_for_rules.date():
                    valid_dates_for_preview.append(date_obj)
            
            if not valid_dates_for_preview:
                 final_bot_response = "The date(s) you requested are in the past. Please provide a valid date (today or in the future)."
                 clear_user_flow_state(employee_id, "book_seat")
                 is_flow_complete_for_response = True
            else:
                collected_info["llm_parsed_dates_iso_list"] = [d.isoformat() for d in valid_dates_for_preview]
                collected_info["awaiting_booking_confirmation"] = True

                preview_table = "| Date | Building | Floor | Seat Number |\n|------|----------|-------|-------------|\n"
                for date_obj in valid_dates_for_preview:
                    preview_table += f"| {date_obj.strftime('%d.%b.%Y')} | {collected_info['building_no']} | {collected_info['floor']} | {collected_info['seat_number']} |\n"
                
                final_bot_response = f"Here is a summary of your request:\n\n{preview_table}\n\n**Do you want to proceed with this booking?** (yes/no)"
                is_flow_complete_for_response = False
        else:
            final_bot_response = await get_llm_response_for_booking(message_text, history, collected_info)
            is_flow_complete_for_response = False

    # General query
    elif current_intent == "general_query" or not current_intent :
        final_bot_response = "I can help you book seat (I'll need the days, building, floor, and seat number), cancel an existing booking (I'll need the seat number and specific date), or view your booking history. Please specify what you'd like to do?"
        is_flow_complete_for_response = True 
        clear_user_flow_state(employee_id, "general_query")  
    
    else: 
        final_bot_response = "I'm not sure how to help with that. You can ask me to 'book a seat', 'cancel a booking', or 'view my history'."
        is_flow_complete_for_response = True 

    # --- 6. Log Bot Response and Return ---
    MAX_HISTORY_LEN = 20 
    if len(history) > MAX_HISTORY_LEN:
        user_data_store[employee_id]["conversation_history"] = history[-MAX_HISTORY_LEN:]

    history.append({"role": "assistant", "content": final_bot_response})
    return ChatResponse(item=final_bot_response, status=True)

@app.get("/")
async def root():
    return {"message": f"Bosch seat booking Chatbot API is running version: UATToken_service"}
