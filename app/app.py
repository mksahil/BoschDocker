import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import httpx
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

# FastAPI App Initialization
app = FastAPI(title="Agile Arena Bosch Chatbot API", version="3.14") # Version updated for prompt fix

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
                    api_key="da151a3ed5194c77880111ff94c40d4b", # Replace with your actual key or use environment variables
                    model="gpt-4o-mini",
                    api_version="2024-02-15-preview",
                    azure_endpoint="https://icubeai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview", # Replace with your actual endpoint
                    temperature=0,)

# In-memory user data store
user_data_store: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    employee_id: str

class SeatAvailability(BaseModel):
    seat_number: str
    is_available: bool
    workspace_id: str
    status_name: str

class ChatResponse(BaseModel):
    employee_id: str
    response: str
    is_complete: bool

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
            return result_dates if result_dates else [today] # fallback to today if range parsing yields nothing

    # Specific dates "15th July", "July 15", "15th of July"
    current_year = today.year
    current_month = today.month
    # Regex for "15th July", "15 July", "July 15", "15th of July"
    # Supports optional "of", day suffixes (st,nd,rd,th)
    date_pattern = r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    month_mapping = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5, # "may" is already short
                     "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
                     "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
    
    found_specific_dates = False
    reference_month = None # To handle "15th, 16th July" vs "July 15th, 16th"

    matches = list(re.finditer(date_pattern, description)) # Find all "Day Month" patterns first
    for match in matches:
        day = int(match.group(1))
        month_name = match.group(2)
        month = month_mapping.get(month_name.lower())
        if month and 1 <= day <= 31:
            # Determine year: if month is past, or same month but day is past, assume next year
            target_year = current_year
            if month < current_month or (month == current_month and day < today.day):
                target_year = current_year + 1
            
            try:
                date_obj = datetime(year=target_year, month=month, day=day)
                if date_obj not in result_dates: result_dates.append(date_obj)
                reference_month = month # Set reference month from first valid date
                found_specific_dates = True
            except ValueError: # Invalid date (e.g., Feb 30)
                pass
    
    # Handle "15th, 16th" after a month has been established, or "15th, 16th of this month" implicitly
    # This regex looks for numbers (days) that are NOT part of a "Day Month" pattern already processed
    if not found_specific_dates or (found_specific_dates and re.search(r",\s*(\d{1,2})(?:st|nd|rd|th)?(?!\s+(?:of\s+)?(?:jan|feb|...))", description)): # Check for trailing days
        # If no month found yet, or if there are comma-separated days following a "Day Month"
        # Use current month if no reference_month, otherwise use established reference_month
        effective_month = reference_month if reference_month else current_month

        for day_match in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", description):
            # Ensure this day_match isn't already part of a "Day Month" match
            is_part_of_full_date = False
            for m in matches:
                if m.start() <= day_match.start() and m.end() >= day_match.end():
                    is_part_of_full_date = True
                    break
            if is_part_of_full_date:
                continue

            day = int(day_match.group(1))
            if 1 <= day <= 31:
                target_year = current_year
                target_month = effective_month
                
                # Check if this day in this month/year is in the past
                try: temp_date_obj = datetime(year=target_year, month=target_month, day=day)
                except ValueError: continue # Skip invalid day for month (e.g. April 31)

                if temp_date_obj < today:
                    target_month +=1 # Try next month
                    if target_month > 12:
                        target_month = 1
                        target_year +=1
                
                try:
                    date_obj = datetime(year=target_year, month=target_month, day=day)
                    if date_obj not in result_dates: result_dates.append(date_obj)
                    found_specific_dates = True # Mark that we found some form of specific date
                except ValueError:
                    pass # Invalid date
                        
    if found_specific_dates:
        return sorted(list(set(result_dates))) # Return all collected specific dates

    # Fallbacks if no other patterns matched
    if not result_dates:
        if "today" in description:
            result_dates.append(today)
        elif "tomorrow" in description:
            result_dates.append(today + timedelta(days=1))
        else: # Default to today if nothing else matches
            result_dates.append(today)
            
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
    from_time = target_date.replace(hour=8, minute=0, second=0, microsecond=0) # Standard booking time
    to_time = target_date.replace(hour=20, minute=0, second=0, microsecond=0) # Standard booking time
    from_date_str, to_date_str = from_time.isoformat(), to_time.isoformat()

    associate_info_data = user_data_store.get(employee_id, {}).get("collected_info", {})
    associate_name = associate_info_data.get("employee_name", "AI Bot User")
    associate_email = associate_info_data.get("employee_email", "")

    temp_associate_info = await get_associate_info(access_token, employee_id) # Ensure fresh info for associateId
    if not temp_associate_info:
         return BookingResult(success=False, message="Failed to retrieve associate information for booking.")
    associate_id_val = temp_associate_info.get('associateId', 0)
    if not associate_id_val: # Check if associateId is valid
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
    payload = {"keySearch": employee_id, "pageIndex": 1, "pageSize": 50} # Fetch more to allow for recent N
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
            response = await client.get(url, headers=headers) # Changed to GET as per typical cancel patterns
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
    floor_word_to_number = {
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
    
    # Handle date ranges like "9th june - 12th june" or "june 9 - june 12"
    date_range_pattern = r'(\d{1,2})(?:th|st|nd|rd)?\s+(\w+)\s*[-–—to]\s*(\d{1,2})(?:th|st|nd|rd)?\s+(\w+)'
    range_match = re.search(date_range_pattern, message_lower)
    
    if range_match:
        start_day, start_month, end_day, end_month = range_match.groups()
        try:
            # Parse start and end dates
            start_date = parse_date_string(f"{start_day} {start_month} {current_year}")
            end_date = parse_date_string(f"{end_day} {end_month} {current_year}")
            
            if start_date and end_date:
                # Generate all dates in the range
                current_date = start_date
                while current_date <= end_date:
                    dates.append(current_date.strftime("%Y-%m-%d"))
                    current_date += timedelta(days=1)
                return dates
        except:
            pass
    
    # Handle single date ranges like "9th - 12th june"
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
    
    # Handle "today and tomorrow"
    if "today and tomorrow" in message_lower:
        dates.append(today_iso_date)
        dates.append((today_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    # Handle "tomorrow and day after"
    elif "tomorrow and day after" in message_lower:
        dates.append((today_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        dates.append((today_date + timedelta(days=2)).strftime("%Y-%m-%d"))
    # Handle "next X days"
    elif "next" in message_lower and "day" in message_lower:
        numbers = re.findall(r'\d+', message_lower)
        if numbers:
            num_days = int(numbers[0])
            for i in range(num_days):
                dates.append((today_date + timedelta(days=i)).strftime("%Y-%m-%d"))
    # Handle just "today"
    elif "today" in message_lower and "tomorrow" not in message_lower:
        dates.append(today_iso_date)
    # Handle just "tomorrow"
    elif "tomorrow" in message_lower and "today" not in message_lower:
        dates.append((today_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    
    return dates

def parse_date_string(date_str: str) -> Optional[datetime]:
    """
    Parse various date string formats into datetime object
    """
    from dateutil import parser
    try:
        return parser.parse(date_str)
    except:
        # Fallback manual parsing for common formats
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

    # Ensure employee name/email are fresh or fetched if missing, for booking payload
    if not collected_info.get("employee_name") or not collected_info.get("employee_email"):
        await get_associate_info(access_token, employee_id) # This updates user_data_store

    associate_api_info = user_data_store[employee_id].get("associate_api_data") # Use cached if available
    if not associate_api_info: # Fetch if not cached or forced refresh needed
        associate_api_info = await get_associate_info(access_token, employee_id)
        if associate_api_info:
            user_data_store[employee_id]["associate_api_data"] = associate_api_info
        else:
            for date_obj in dates_to_book_dt: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="System error: Failed to get associate info."))
            return {"success": False, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info), "message": "Failed to retrieve associate information."}
    print("building_no:", collected_info["building_no"],"floor: ", collected_info["floor"])
    unit_id = extract_unit_id(associate_api_info, collected_info["building_no"], collected_info["floor"])
    print("unit_id:", unit_id)  
    if not unit_id:
        for date_obj in dates_to_book_dt: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="Invalid Building/Floor or unit ID not found."))
        return {"success": False, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info), "message": "Invalid Building/Floor or unit ID not found."}

    seat_num_to_check = collected_info["seat_number"]

    if seat_num_to_check and len(seat_num_to_check) > 2 and "-" not in seat_num_to_check:
                                formatted_seat_num = f"{seat_num_to_check[:2]}-{seat_num_to_check[2:]}"
                                print(f"Reformatted seat number from {seat_num_to_check} to {formatted_seat_num}")
                                seat_num_to_check = formatted_seat_num
                                print("seat_num_to_check:", seat_num_to_check)  
                                collected_info["seat_number"] = seat_num_to_check # Update for consistency

    print("seat_num_to_check:", seat_num_to_check)  
    print("dates_list:", dates_to_book_dt)
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
            print("Booking API result:", booking_api_result)
            multi_day_results.append(DayBookingStatus(date=date_obj, success=booking_api_result.success, message=booking_api_result.message, details=booking_api_result.booking_details, workspace_id=seat_availability.workspace_id))
        else:
            multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="Workspace ID not found for booking."))
    overall_success = any(res.success for res in multi_day_results)
    return {"success": overall_success, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info)}

def get_status_name(status_code: int) -> str:
    status_map = {
        4: "Booked",  # Per user: This is cancellable
        1: "Cancelled", # Per user: This means already cancelled
        2: "Checked-In", # Assumption from prior context
        3: "Auto Cancelled", # Assumption from prior context
    }
    return status_map.get(status_code, f"Unknown (Status {status_code})")

def filter_and_sort_cancellable_bookings(booking_history: List[BookingHistoryItem]) -> List[BookingHistoryItem]:
    cancellable: List[BookingHistoryItem] = []
    today = datetime.now().date()
    for booking in booking_history:
        try:
            from_date_dt = datetime.fromisoformat(booking.fromDate)
            # Apply user's rule: Status 4 is "Booked" and cancellable if for today or future.
            if booking.status == 4 and from_date_dt.date() >= today:
                cancellable.append(booking)
        except Exception as e:
            print(f"Skipping booking item {booking.allocationID} due to parsing/filter error: {e}")
            continue
    # Sort by date, soonest first
    cancellable.sort(key=lambda b: datetime.fromisoformat(b.fromDate))
    return cancellable

def generate_initial_greeting(employee_id: str) -> str:
    user_name = user_data_store.get(employee_id, {}).get("collected_info", {}).get("employee_name")
    greeting = f"Hello{(' ' + user_name) if user_name else ''}! "
    greeting += "I can help you book a seat, cancel an existing booking, or show your booking history. What would you like to do?"
    return greeting

def clear_user_flow_state(employee_id: str, intent_to_clear: Optional[str] = None):
    """Clears flags and temporary data related to a completed or aborted flow."""
    if employee_id not in user_data_store:
        return
    
    # Common fields to clear after any flow completion/reset
    fields_to_pop = [
        "intent", "cancellation_step", "selected_allocation_id_for_cancel",
        "cancellable_bookings_cache", "awaiting_booking_confirmation",
        "booking_dates_iso", "offered_tomorrow", "offered_next_week",
        "specific_booking_details_for_cancel", # For specific cancellation flow
        "llm_parsed_dates_iso_list", "llm_parsed_cancel_date_iso", "selected_allocation_id"
    ]
    
    # Fields related to booking
    booking_related_fields = LOCATION_FIELDS + BOOKING_INFO_FIELDS
    # Fields related to specific cancellation
    cancel_related_fields = CANCEL_INFO_FIELDS
    # Fields related to view history
    view_history_related_fields = VIEW_HISTORY_FIELDS

    if intent_to_clear == "book_seat":
        fields_to_pop.extend(booking_related_fields)
    elif intent_to_clear == "cancel_seat":
        fields_to_pop.extend(cancel_related_fields)
    elif intent_to_clear == "view_booking_history":
        fields_to_pop.extend(view_history_related_fields)
    else: # Full reset of all flow-specific fields
        fields_to_pop.extend(booking_related_fields)
        fields_to_pop.extend(cancel_related_fields)
        fields_to_pop.extend(view_history_related_fields)

    for field in set(fields_to_pop): # Use set to avoid duplicates
        user_data_store[employee_id]["collected_info"].pop(field, None)

async def get_llm_response_for_booking(message: str, conversation_history: List[Dict[str, str]], collected_info: Dict[str, Any]):
    system_prompt_parts = ["You are a helpful assistant for Bosch seat booking."]
    missing_fields_desc = [] # For human-readable list of missing fields
    
    # Define required fields for booking and their user-friendly descriptions
    required_booking_fields_map = {
        "booking_days_description": "booking_days_description (e.g., 'today and next 2 days', 'next Monday', '15th July')",
        "building_no": "building number (e.g., '903')",
        "floor": "floor (e.g., 'Floor 1', '2nd floor')",
        "seat_number": "seat number (e.g., 'L1-001')"
    }
    
    # Check which required fields are missing
    for field_key, field_desc in required_booking_fields_map.items():
        if field_key not in collected_info or not collected_info[field_key]:
            missing_fields_desc.append(field_desc)
    
    if missing_fields_desc:
        system_prompt_parts.append(f"Ask for the first missing detail for booking: {missing_fields_desc[0]}.")
        if len(missing_fields_desc) > 1:
            system_prompt_parts.append(f"The other pending fields for booking are: {', '.join(missing_fields_desc[1:])}.")
    else:
        system_prompt_parts.append("All required information for booking (days, building, floor, seat number) seems to be collected. You should be preparing a summary.")
    
    system_prompt_parts.extend([
        "Do not ask for employee name or email; they will be fetched automatically from the API.",
        "Do NOT ask for the booking time. The booking will be for the standard working hours (8 AM to 8 PM).",
        "Focus only on collecting information for seat booking: booking days description, building number, floor, and seat number. Ask only one question at a time. Be friendly but efficient.",
        "If the user has provided all necessary information for booking, do not ask for it again.",
        "Do not ask for confirmation to book yet; that will be handled separately.",
        "Do not show internal API details. Respond concisely."
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
    """
    Normalize seat number by removing hyphens and converting to lowercase
    for flexible comparison.
    """
    return seat_number.replace("-", "").lower()

def seats_match(seat1: str, seat2: str) -> bool:
    """
    Compare two seat numbers flexibly by normalizing them first.
    This handles cases where one has hyphens and the other doesn't.
    """
    return normalize_seat_number(seat1) == normalize_seat_number(seat2)

async def extract_info_with_llm(message: str, current_info: Dict[str, Any], today_iso_date: str) -> Dict[str, Any]:
    """
    Analyzes the user message to extract intent and parameters using a structured LLM prompt
    that is less prone to content filtering.
    """
    today_date = datetime.strptime(today_iso_date, "%Y-%m-%d")

    llm_context_info = {
        k: v for k, v in current_info.items()
        if k not in ['cancellable_bookings_cache', 'booking_history_cache', 'associate_api_data', 'employee_name', 'employee_email']
    }
    system_prompt = f"""
You are an expert AI assistant for a seat booking system. Your task is to analyze a user's message and conversation context, then output a single, valid JSON object.

The JSON object must have two keys: "intent" and "parameters".
Today's date is: {today_iso_date}.

--- INTENTS ---
- 'book_seat': User wants to reserve a seat (even if details are incomplete).
- 'cancel_seat': User wants to cancel a booking (even if details are incomplete).
- 'view_booking_history': User wants to see their booking history.
- 'general_query': The request is completely unrelated to booking, canceling, or viewing history, or is just a greeting.

--- PARAMETERS (by Intent) ---
1. For 'book_seat':
   - 'booking_days_description': The user's raw text for dates (e.g., 'next 3 days', 'July 15th').
   - 'llm_parsed_dates_iso_list': A list of all dates in 'YYYY-MM-DD' format. Fully resolve all ranges. Example: "June 9-11" becomes ["{today_date.year}-06-09", "{today_date.year}-06-10", "{today_date.year}-06-11"].
   - 'building_no': Building number (e.g., '903').
   - 'floor': Floor description (e.g., '2nd floor').
   - 'seat_number': Seat identifier, normalized to uppercase (e.g., 'L1-001').

2. For 'cancel_seat':
   - 'cancel_seat_number': The seat to cancel, normalized to uppercase.
   - 'cancel_date_description': The user's raw text for the cancellation date.
   - 'llm_parsed_cancel_date_iso': The single cancellation date in 'YYYY-MM-DD' format.

3. For 'view_booking_history':
   - 'history_count': The number of bookings to show (e.g., '3').

--- RULES ---
- IMPORTANT: Recognize the intent even if booking/cancellation details are incomplete. For example:
  * "book a seat" → intent: "book_seat" (even without building/floor/seat details)
  * "cancel my booking" → intent: "cancel_seat" (even without seat number/date)
  * "show my history" → intent: "view_booking_history"
- Only extract parameters from the user's LATEST message.
- If a parameter is not mentioned in the latest message, do not include its key in the 'parameters' object.
- Use 'general_query' only for messages completely unrelated to seat booking operations (like greetings, weather questions, etc.).

--- EXAMPLES ---
User message: "book a seat"
Your output:
{{
  "intent": "book_seat",
  "parameters": {{}}
}}

User message: "cancel L1-004 on 10 Jun 2025"
Your output:
{{
  "intent": "cancel_seat",
  "parameters": {{
    "cancel_seat_number": "L1-004",
    "cancel_date_description": "10 Jun 2025",
    "llm_parsed_cancel_date_iso": "2025-06-10"
  }}
}}

User message: "cancel my booking"
Your output:
{{
  "intent": "cancel_seat",
  "parameters": {{}}
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
        
        # Try to find JSON block if LLM adds extra text
        match_json = re.search(r"\{.*\}", extracted_json_str, re.DOTALL)
        if match_json:
            extracted_json_str = match_json.group(0)

        llm_output = json.loads(extracted_json_str)
        print(f"LLM Raw Output: {llm_output}")

        # Construct the final dictionary to return
        final_output = {}
        final_output["intent"] = llm_output.get("intent", "general_query")
        
        # Merge parameters into the top-level of the dictionary
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    message_text = request.message.strip()
    employee_id = request.employee_id
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
        collected_info = user_data_store[employee_id]["collected_info"] 
        final_bot_response = generate_initial_greeting(employee_id)
        is_flow_complete_for_response = True 
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

    if message_text.lower() in GREETING_COMMANDS and \
       not collected_info.get("intent") and \
       not collected_info.get("awaiting_booking_confirmation") and \
       not collected_info.get("cancellation_step"):
        final_bot_response = generate_initial_greeting(employee_id)
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=False)

    # --- 2. Log User Message and Extract Intent & Parameters via LLM ---
    history.append({"role": "user", "content": message_text})
    
    previous_intent_before_llm = collected_info.get("intent")
    llm_context = {k: v for k, v in collected_info.items() if k not in ['cancellable_bookings_cache', 'booking_history_cache', 'associate_api_data', 'employee_name', 'employee_email']}
    
    extracted_llm_data = await extract_info_with_llm(message_text, llm_context, today_iso_date_str)
    
    newly_extracted_intent = extracted_llm_data.get("intent")

    # --- 3. Handle Intent Change (Interrupt Flow if Necessary) ---
    if newly_extracted_intent and newly_extracted_intent != previous_intent_before_llm:
        print(f"Intent changed from '{previous_intent_before_llm}' to '{newly_extracted_intent}'. Clearing state for '{previous_intent_before_llm}'.")
        clear_user_flow_state(employee_id, previous_intent_before_llm) # Clear data of OLD intent
    
    # Update collected_info with all data extracted by LLM
    collected_info.update(extracted_llm_data)

    current_intent = collected_info.get("intent") 

    user_confirms = any(w in message_text.lower() for w in ["yes", "yep", "yeah", "confirm", "proceed", "ok", "sure", "do it"])
    user_declines = any(w in message_text.lower() for w in ["no", "nope", "cancel", "stop", "don't", "do not", "never mind"])

    # --- 4. Process Confirmation Steps (if intent hasn't changed away from them) ---
    if collected_info.get("cancellation_step") == "awaiting_confirmation" and current_intent == "cancel_seat":
        if user_confirms:
            allocation_id = collected_info.get("selected_allocation_id")
            access_token = await get_new_access_token() 
            if not access_token:
                final_bot_response = "Sorry, I couldn't get authorization. Please try again later."
            elif allocation_id:
                cancel_result = await cancel_booking_api(access_token, allocation_id)
                if cancel_result.message.startswith("Seat is cancled"):
                    final_bot_response += "Your booking has been successfully cancelled."
                print(f"Cancel result message: {cancel_result.message}")
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
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

    elif collected_info.get("awaiting_booking_confirmation") and current_intent == "book_seat":
        if user_confirms:
            dates_to_book_iso = collected_info.get("llm_parsed_dates_iso_list", [])

            # --- START: Past Date Validation Fix ---
            today = datetime.now().date()
            valid_dates_iso = []
            ignored_dates_count = 0

            for date_str in dates_to_book_iso:
                try:
                    # Compare only the date part, ignoring any time information
                    if datetime.fromisoformat(date_str.split('T')[0]).date() >= today:
                        valid_dates_iso.append(date_str)
                    else:
                        ignored_dates_count += 1
                except (ValueError, TypeError):
                    # Also count malformed date strings from the LLM as ignored
                    ignored_dates_count += 1
            
            warning_message = ""
            if ignored_dates_count > 0:
                plural_s = "s" if ignored_dates_count > 1 else ""
                warning_message = f"Please note: I cannot book seats for past dates. {ignored_dates_count} requested date{plural_s} were ignored.\n\n"

            if valid_dates_iso:
                # Proceed to book only the valid, future-or-today dates
                booking_process_result = await process_multi_date_booking(employee_id, valid_dates_iso, collected_info)
                if booking_process_result["success"]: 
                    final_bot_response = warning_message + f"Your booking request for the valid dates has been processed.\n\n{booking_process_result['confirmation_table']}"
                    history.clear() 
                    history.append({"role": "user", "content": message_text})
                else:
                    final_bot_response = warning_message + f"Your booking request for the valid dates was processed, but encountered issues:\n\n{booking_process_result['confirmation_table']}"
            else:
                # This branch is reached if the initial date list was empty OR contained only past/invalid dates.
                final_bot_response = warning_message + "There were no valid future dates to book. Please try the booking process again with a valid date (today or later)."
            
            is_flow_complete_for_response = True
            clear_user_flow_state(employee_id, "book_seat")
            # --- END: Past Date Validation Fix ---

        elif user_declines:
            final_bot_response = "Okay, I will not proceed with the booking. Is there anything else?"
            is_flow_complete_for_response = True
            clear_user_flow_state(employee_id, "book_seat")
        else: 
            final_bot_response = "I'm sorry, I didn't catch that. Do you want to proceed with the booking? (yes/no)"

        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

    # --- 5. Main Intent Processing Logic (driven by LLM output) ---
    if current_intent == "cancel_seat":
        access_token = await get_new_access_token()
        if not access_token:
            final_bot_response = "Sorry, system error getting authorization. Please try again."
            is_flow_complete_for_response = True
        else:
            seat_number = collected_info.get("cancel_seat_number")
            parsed_cancel_date_iso = collected_info.get("llm_parsed_cancel_date_iso") # From LLM
            
            if not seat_number and not parsed_cancel_date_iso:
                 final_bot_response = "To cancel a booking, please tell me the seat number and the date (e.g., 'cancel L1-001 for tomorrow')."
            elif not seat_number:
                final_bot_response = "Okay, which seat number do you want to cancel?"
            elif not parsed_cancel_date_iso:
                final_bot_response = f"And for which date do you want to cancel seat {seat_number}? Please provide a single specific date (e.g., 'July 15th' or 'tomorrow')."
                if collected_info.get("cancel_date_description") and not parsed_cancel_date_iso: 
                    final_bot_response = f"I had trouble understanding the date '{collected_info.get('cancel_date_description')}' for cancelling seat {seat_number}. Could you please provide a single specific date?"
                collected_info.pop("llm_parsed_cancel_date_iso", None)
                collected_info.pop("cancel_date_description", None)
            else: # Both seat and parsed_cancel_date_iso provided
                try:
                    target_date_obj = datetime.strptime(parsed_cancel_date_iso, "%Y-%m-%d").date()
                except ValueError:
                    final_bot_response = "The date provided for cancellation seems invalid. Please try again with a YYYY-MM-DD format or a clear description like 'tomorrow'."
                    collected_info.pop("llm_parsed_cancel_date_iso", None)
                    collected_info.pop("cancel_date_description", None)
                else:
                    all_bookings = await get_booking_history(access_token, employee_id)
                    found_booking = None
                    for booking in all_bookings:
                        try:
                            booking_date = datetime.fromisoformat(booking.fromDate).date()
                            if (seats_match(booking.seat, seat_number) and 
                                  booking_date == target_date_obj and 
                                   booking.status == 4 and booking_date >= datetime.now().date()):
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
                                            f"Please double-check the details or note that only 'Booked' status items for today or future can be cancelled.")
                        clear_user_flow_state(employee_id, "cancel_seat")


    elif current_intent == "view_booking_history":
        access_token = await get_new_access_token()
        if not access_token:
            final_bot_response = "Sorry, system error getting authorization for history. Please try again."
        else:
            booking_hist_items = await get_booking_history(access_token, employee_id)
            print(f"Booking history items fetched: {len(booking_hist_items)}")
            if not booking_hist_items:
                final_bot_response = "You have no booking history."
            else:
                # Sort by 'time' field (when booking was made/cancelled) instead of 'fromDate'
                booking_hist_items.sort(key=lambda item: datetime.fromisoformat(item.time), reverse=True)
                count_str = collected_info.get("history_count", "5") 
                try:
                    history_count_to_show = int(re.search(r'\d+', str(count_str)).group(0) if re.search(r'\d+', str(count_str)) else 5)
                    print(f"History count to show: {history_count_to_show}")
                except (ValueError, AttributeError): history_count_to_show = 5
                
                history_to_show = booking_hist_items[:min(history_count_to_show, len(booking_hist_items))]
                print(f"Filtered history to show: {len(history_to_show)} items")
                response_parts = [f"Here are your {len(history_to_show)} most recent booking(s):\n"]
                for item in history_to_show:
                    dt_obj = datetime.fromisoformat(item.fromDate)
                    booking_time = datetime.fromisoformat(item.time)
                    status_name = get_status_name(item.status)
                    response_parts.append(
                        f"- Seat: {item.seat}, Booked for: {dt_obj.strftime('%d %b %Y, %I:%M %p')}, "
                        f"Booked on: {booking_time.strftime('%d %b %Y, %I:%M %p')}, "
                        f"Building: {item.building}, Floor: {item.floor}, Status: {status_name}"
                    )
                    print("response_parts ",response_parts)  # Debug print for each item
                final_bot_response = "\n".join(response_parts)
        is_flow_complete_for_response = True
        clear_user_flow_state(employee_id, "view_booking_history")

    elif current_intent == "book_seat":
        required_booking_params = ["llm_parsed_dates_iso_list", "building_no", "floor", "seat_number"]
        all_info_collected = all(field in collected_info and collected_info[field] for field in required_booking_params)

        if all_info_collected:
            parsed_dates_iso = collected_info["llm_parsed_dates_iso_list"]
            booking_summary = f"""
Okay, I have the following details for your booking:
- Seat: {collected_info['seat_number'].upper()}
- Building: {collected_info['building_no']}
- Floor: {collected_info['floor']}
- Dates: {format_dates_for_display(parsed_dates_iso)}

Do you want to proceed with this booking? (yes/no)
"""
            final_bot_response = booking_summary
            collected_info["awaiting_booking_confirmation"] = True
        else: 
            final_bot_response = await get_llm_response_for_booking(message_text, history, collected_info)

    elif current_intent == "general_query" or not current_intent :
        final_bot_response = "I can help you book seats, cancel bookings, or view your booking history. What would you like to do today?"
        is_flow_complete_for_response = True 
        clear_user_flow_state(employee_id, "general_query") 
    
    else: 
        final_bot_response = "I'm not sure how to help with that. You can ask me to 'book a seat', 'cancel a booking', or 'view my history'."
        is_flow_complete_for_response = True 

    # --- 6. Log Bot Response and Return ---
    MAX_HISTORY_LEN = 20 
    if len(history) > MAX_HISTORY_LEN:
        history = history[-MAX_HISTORY_LEN:]
        user_data_store[employee_id]["conversation_history"] = history

    history.append({"role": "assistant", "content": final_bot_response})
    return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

@app.get("/")
async def root():
    return {"message": f"Bosch seat booking Chatbot API is running version: 3.14"}