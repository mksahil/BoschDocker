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
app = FastAPI(title="Agile Arena Bosch Chatbot API", version="3.10") # Version updated for new cancel/view logic

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
def format_dates_for_display(dates: List[datetime]) -> str:
    if not dates: return "No dates specified"
    dates = sorted(list(set(dates)))
    display_format_dates = [d.strftime('%B %d, %Y') for d in dates]
    if len(dates) == 1: return f"Single day: {display_format_dates[0]}"
    return f"Multiple days ({len(dates)}): {', '.join(display_format_dates)}"

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

async def extract_info_with_llm(message: str, current_info: Dict[str, Any]) -> Dict[str, Any]:
    info_to_extract_booking = {
        "booking_days_description": "Textual description of days for booking (e.g., 'today and next 3 days', '13th May, 14th, 15th, 16th May', 'Monday to Wednesday').",
        "building_no": "Building number (e.g., '903').",
        "floor": "Floor (e.g., 'Floor 1', '2nd floor').",
        "seat_number": "Seat number (e.g., 'L1-001')."
    }
    info_to_extract_cancel = {
        "cancel_seat_number": "The seat number of the booking to cancel (e.g., 'L1-001').",
        "cancel_date_description": "Textual description of the date of the booking to cancel (e.g., 'tomorrow', 'July 25th'). This should resolve to a single date."
    }
    info_to_extract_view_history = {
        "history_count": "Number of booking entries to show (e.g., '2', 'last 3'). If not specified, do not include this field."
    }

    system_content = f"""
You are an information extraction system. Your primary goal is to understand the user's INTENT and then extract relevant details.
User's message: "{message}"
Current collected information (excluding sensitive/large lists): {json.dumps({k: v for k, v in current_info.items() if k not in ['cancellable_bookings_cache', 'booking_history_cache']})}

Intents to identify:
- book_seat: User wants to reserve a new seat.
- cancel_seat: User wants to cancel an existing booking.
- view_booking_history: User wants to see their booking history.
- general_query: User is asking a question or making a statement not directly related to booking, cancelling, or viewing history.

If intent is 'book_seat', extract these details if mentioned in the user's *current* message (do NOT ask for booking time):
"""
    for key, desc in info_to_extract_booking.items():
        system_content += f"- {key}: {desc}\n"

    system_content += """
If intent is 'cancel_seat':
  - If the user provides a specific seat AND a single date/day for cancellation (e.g., "cancel my booking for L1-001 on July 20th", "cancel S001 for tomorrow"), extract:
"""
    for key, desc in info_to_extract_cancel.items():
        system_content += f"    - {key}: {desc}\n"
    system_content += """  - Do NOT ask for how many days to cancel. Cancellation is for a single specified day.
  - If the user says "cancel my booking" without specifics, do not extract these cancel-specific fields; the system will handle it by listing bookings.

If intent is 'view_booking_history', extract this detail if mentioned:
"""
    for key, desc in info_to_extract_view_history.items():
        system_content += f"- {key}: {desc}\n"

    system_content += """
Return ONLY a JSON object.
- The JSON object *must* include an "intent" field.
- If a booking, cancellation, or history field is already in 'current_info' and the user's message doesn't provide a new value, do NOT include it in the JSON.
- Output a minimal JSON with only newly extracted/updated values for the identified intent, and the determined intent.
- Normalize extracted values: building "building 903" -> "903", floor "1st floor" -> "1st floor", seat "seat L028" -> "L028".

Example (Booking):
User: "I'd like to book for Monday to Wednesday in building 903."
JSON: {"intent": "book_seat", "booking_days_description": "Monday to Wednesday", "building_no": "903"}

Example (Specific Cancellation):
User: "Cancel my booking for seat S007 on August 1st."
JSON: {"intent": "cancel_seat", "cancel_seat_number": "S007", "cancel_date_description": "August 1st"}

Example (General Cancellation):
User: "Please cancel my reservation."
JSON: {"intent": "cancel_seat"}

Example (View History):
User: "Show my last 2 bookings."
JSON: {"intent": "view_booking_history", "history_count": "2"}

Example (General Query):
User: "What time do bookings close?"
JSON: {"intent": "general_query"}
"""
    updated_extraction_info = current_info.copy()
    try:
        messages = [ SystemMessage(content=system_content), HumanMessage(content="Extract information based on my previous message and the rules provided.") ]
        response_llm = llm.invoke(messages)
        extracted_json_str = response_llm.content.strip()
        match = re.search(r"\{.*\}", extracted_json_str, re.DOTALL)
        if match: extracted_json_str = match.group(0)
        
        try:
            extracted_data = json.loads(extracted_json_str)
            print(f"LLM Extracted JSON: {extracted_data}")
            if "intent" in extracted_data:
                updated_extraction_info["intent"] = extracted_data["intent"]
            
            all_possible_fields = BOOKING_INFO_FIELDS + LOCATION_FIELDS + CANCEL_INFO_FIELDS + VIEW_HISTORY_FIELDS
            for field in all_possible_fields:
                if field in extracted_data and extracted_data[field] is not None:
                    # Clear related fields if intent changes to avoid stale data issues
                    if "intent" in extracted_data and updated_extraction_info.get("intent") != current_info.get("intent"):
                        if field in BOOKING_INFO_FIELDS + LOCATION_FIELDS and extracted_data["intent"] != "book_seat": continue
                        if field in CANCEL_INFO_FIELDS and extracted_data["intent"] != "cancel_seat": continue
                        if field in VIEW_HISTORY_FIELDS and extracted_data["intent"] != "view_booking_history": continue
                    updated_extraction_info[field] = str(extracted_data[field]).strip()
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in extract_info_with_llm: {e}. Raw: {extracted_json_str}")
            if not updated_extraction_info.get("intent"):
                if any(kw in message.lower() for kw in ["cancel", "remove my booking", "delete reservation"]):
                    updated_extraction_info["intent"] = "cancel_seat"
                elif any(kw in message.lower() for kw in ["show my bookings", "booking history", "my reservations"]):
                    updated_extraction_info["intent"] = "view_booking_history"
                elif any(kw in message.lower() for kw in ["book", "reserve", "get a seat"]):
                     updated_extraction_info["intent"] = "book_seat"

        # Regex fallbacks (primarily for booking, can be extended if LLM struggles with cancel/view fields)
        if updated_extraction_info.get("intent") == "book_seat" or not updated_extraction_info.get("intent"):
            if "building_no" not in updated_extraction_info or not updated_extraction_info.get("building_no"):
                b_match = re.search(r"\b(?:building\s*(?:no\.?|number)?\s*)?(\d{3,4})\b", message, re.IGNORECASE)
                if b_match: updated_extraction_info["building_no"] = b_match.group(1).strip()
            if "floor" not in updated_extraction_info or not updated_extraction_info.get("floor"):
                f_match = re.search(r"\b(\d+(?:st|nd|rd|th)?\s*floor|ground\s*floor|floor\s*\d+|G\s*floor)\b", message, re.IGNORECASE)
                if f_match: updated_extraction_info["floor"] = f_match.group(0).strip()
            if "seat_number" not in updated_extraction_info or not updated_extraction_info.get("seat_number"): # For booking
                s_match = re.search(r"\b([A-Z]\d{1,2}-?\d{2,3}|S\d{3,4}|L\d{3})\b", message) # General seat pattern
                if s_match: updated_extraction_info["seat_number"] = s_match.group(1).strip().upper()
        
        # Regex fallback for cancel_seat_number if intent is cancel_seat but LLM missed it
        if updated_extraction_info.get("intent") == "cancel_seat" and ("cancel_seat_number" not in updated_extraction_info or not updated_extraction_info.get("cancel_seat_number")):
            s_match = re.search(r"\b(?:seat\s*)?([A-Z]\d{1,2}-?\d{2,3}|S\d{3,4}|L\d{3})\b", message, re.IGNORECASE)
            if s_match: updated_extraction_info["cancel_seat_number"] = s_match.group(1).strip().upper()

        print(f"Info after LLM + fallbacks: {updated_extraction_info}")
        return updated_extraction_info
    except Exception as e:
        print(f"Error in extract_info_with_llm: {e}")
        return current_info

async def get_llm_response_for_booking(message: str, conversation_history: List[Dict[str, str]], collected_info: Dict[str, Any]):
    system_prompt_parts = ["You are a helpful assistant for Bosch seat booking."]
    missing_fields_desc = [] # For human-readable list of missing fields
    
    # Define required fields for booking and their user-friendly descriptions
    required_booking_fields_map = {
        "booking_days_description": "booking_days_description (e.g., 'today and next 2 days', 'next Monday', '15th July')",
        "building_no": "building number",
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
        "Do no show the collected information to the user.",
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

async def process_multi_date_booking(employee_id: str, dates_list: List[datetime], collected_info: Dict[str, Any]):
    multi_day_results: List[DayBookingStatus] = []
    access_token = await get_new_access_token()
    if not access_token:
        for date_obj in dates_list: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="System error: Auth token failed."))
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
            for date_obj in dates_list: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="System error: Failed to get associate info."))
            return {"success": False, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info), "message": "Failed to retrieve associate information."}
    print("building_no:", collected_info["building_no"],"floor: ", collected_info["floor"])
    unit_id = extract_unit_id(associate_api_info, collected_info["building_no"], collected_info["floor"])
    print("unit_id:", unit_id)  
    if not unit_id:
        for date_obj in dates_list: multi_day_results.append(DayBookingStatus(date=date_obj, success=False, message="Invalid Building/Floor or unit ID not found."))
        return {"success": False, "booking_results": multi_day_results, "confirmation_table": format_booking_results_table(multi_day_results, collected_info), "message": "Invalid Building/Floor or unit ID not found."}

    seat_num_to_check = collected_info["seat_number"]

    if seat_num_to_check and len(seat_num_to_check) > 2 and "-" not in seat_num_to_check:
                                formatted_seat_num = f"{seat_num_to_check[:2]}-{seat_num_to_check[2:]}"
                                print(f"Reformatted seat number from {seat_num_to_check} to {formatted_seat_num}")
                                seat_num_to_check = formatted_seat_num
                                print("seat_num_to_check:", seat_num_to_check)  
                                collected_info["seat_number"] = seat_num_to_check # Update for consistency

    print("seat_num_to_check:", seat_num_to_check)  
    print("dates_list:", dates_list)
    for date_obj in dates_list:
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
        "specific_booking_details_for_cancel" # For specific cancellation flow
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

    for field in fields_to_pop:
        user_data_store[employee_id]["collected_info"].pop(field, None)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    message_text = request.message.strip()
    employee_id = request.employee_id
    
    if employee_id not in user_data_store:
        user_data_store[employee_id] = {"conversation_history": [], "collected_info": {}, "associate_api_data": None}
        # Try to fetch associate info on first contact for name/email
        access_token = await get_new_access_token()
        if access_token:
            api_data = await get_associate_info(access_token, employee_id)
            if api_data: user_data_store[employee_id]["associate_api_data"] = api_data

    current_conversation_data = user_data_store[employee_id]
    history = current_conversation_data["conversation_history"]
    collected_info = current_conversation_data["collected_info"]

    final_bot_response = ""
    is_flow_complete_for_response = False

    # --- Initial Reset Handling ---
    if message_text.lower() in RESET_COMMANDS:
        user_data_store[employee_id]["collected_info"] = {} # Clear only collected_info, keep history for context if needed, or clear history too
        # user_data_store[employee_id]["conversation_history"] = [] # Optionally reset history
        history, collected_info = user_data_store[employee_id]["conversation_history"], user_data_store[employee_id]["collected_info"]
        final_bot_response = generate_initial_greeting(employee_id)
        is_flow_complete_for_response = True # Resets the flow
        # Append to history after reset
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

    # --- Initial Greeting Handling ---
    if message_text.lower() in GREETING_COMMANDS and not collected_info.get("intent"):
        final_bot_response = generate_initial_greeting(employee_id)
        # No flow completion here, just a greeting response
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=False)

    user_confirms = any(w in message_text.lower() for w in ["yes", "yep", "yeah", "confirm", "proceed", "ok", "sure", "do it"])
    user_declines = any(w in message_text.lower() for w in ["no", "nope", "cancel", "stop", "don't", "do not", "never mind"])

    # --- Pre-LLM State Handling for Confirmations/Selections ---
    
    # 1.A. Specific Cancellation Confirmation (for seat/date provided by user)
    if collected_info.get("cancellation_step") == "awaiting_specific_confirmation":
        print("Awaiting specific confirmation for cancellation")
        history.append({"role": "user", "content": message_text})
        allocation_id_to_cancel = collected_info.get("selected_allocation_id_for_cancel")
        booking_details = collected_info.get("specific_booking_details_for_cancel", {})
        
        if user_confirms and allocation_id_to_cancel:
            access_token = await get_new_access_token()
            if access_token:
                cancel_api_result = await cancel_booking_api(access_token, allocation_id_to_cancel)
                final_bot_response = cancel_api_result.message
            else:
                final_bot_response = "Sorry, I couldn't get authorization to cancel the booking. Please try again later."
        elif user_declines:
            final_bot_response = f"Okay, the booking for seat {booking_details.get('seat','N/A')} on {booking_details.get('date_str','N/A')} will not be cancelled. Is there anything else?"
        else: # Ambiguous
            final_bot_response = (f"I didn't quite understand. Please confirm with 'yes' or 'no' if you want to cancel the booking "
                                  f"for seat {booking_details.get('seat','N/A')} on {booking_details.get('date_str','N/A')}.")
            history.append({"role": "assistant", "content": final_bot_response})
            return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=False)

        clear_user_flow_state(employee_id, "cancel_seat")
        is_flow_complete_for_response = True
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

    # 1.B. List-based Cancellation Confirmation (user selected from a list)
    elif collected_info.get("cancellation_step") == "awaiting_confirmation": # This is for list-based selection
        print("Awaiting confirmation for cancellation from list selection")
        history.append({"role": "user", "content": message_text})
        allocation_id_to_cancel = collected_info.get("selected_allocation_id_for_cancel")
        print("allocation_id_to_cancel from step 1.B:", allocation_id_to_cancel)
        if user_confirms and allocation_id_to_cancel:
            access_token = await get_new_access_token()
            if access_token:
                cancel_api_result = await cancel_booking_api(access_token, allocation_id_to_cancel)
                final_bot_response = cancel_api_result.message
            else:
                final_bot_response = "Sorry, I couldn't get authorization to cancel the booking. Please try again later."
        elif user_declines:
            final_bot_response = "Okay, the booking will not be cancelled. Is there anything else?"
        else:
            final_bot_response = "I didn't quite understand. Please confirm with 'yes' or 'no' if you want to cancel this booking."
            history.append({"role": "assistant", "content": final_bot_response})
            return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=False)

        clear_user_flow_state(employee_id, "cancel_seat")
        is_flow_complete_for_response = True
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

    # 2. List-based Cancellation Selection (user chooses from a numbered list)
    elif collected_info.get("cancellation_step") == "awaiting_selection":
        print("Awaiting selection for cancellation from list")
        history.append({"role": "user", "content": message_text})
        cancellable_bookings_cache = collected_info.get("cancellable_bookings_cache", [])
        print("cancellable_bookings_cache:", cancellable_bookings_cache)
        selected_index = -1
        if user_declines or message_text.lower() in ["none", "skip"]:
             final_bot_response = "Okay, no bookings will be cancelled. How else can I help?"
             clear_user_flow_state(employee_id, "cancel_seat")
             is_flow_complete_for_response = True
        else:
            try:
                match = re.search(r'\d+', message_text)
                if match: selected_index = int(match.group(0)) - 1 # User sees 1-based
                
                if 0 <= selected_index < len(cancellable_bookings_cache):
                    selected_booking = cancellable_bookings_cache[selected_index]
                    collected_info["selected_allocation_id_for_cancel"] = selected_booking.allocationID
                    collected_info["cancellation_step"] = "awaiting_confirmation" # Standard confirmation for list
                    dt_obj = datetime.fromisoformat(selected_booking.fromDate)
                    final_bot_response = (f"You selected to cancel booking for {selected_booking.seat} "
                                          f"on {dt_obj.strftime('%d %b %Y')} "
                                          f"in {selected_booking.building}. Are you sure? (yes/no)")
                    is_flow_complete_for_response = False # Awaiting confirmation
                else:
                    final_bot_response = "That's not a valid selection. Please enter the number of the booking you want to cancel, or say 'none'."
                    is_flow_complete_for_response = False # Still awaiting valid selection
            except ValueError:
                final_bot_response = "Please enter a number for the booking to cancel, or 'none'."
                is_flow_complete_for_response = False

        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)

    # 3. Booking Confirmation
    elif collected_info.get("awaiting_booking_confirmation"):
        history.append({"role": "user", "content": message_text})
        if user_confirms:
            dates_to_book_iso = collected_info.get("booking_dates_iso", [])
            final_dates_to_book_dt = [datetime.fromisoformat(date_str) for date_str in dates_to_book_iso]
            current_processing_time = datetime.now()
            if collected_info.get("offered_tomorrow"): # User confirmed to add tomorrow
                tomorrow_dt = (current_processing_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                if tomorrow_dt.weekday() < 5 and not any(d.date() == tomorrow_dt.date() for d in final_dates_to_book_dt):
                    final_dates_to_book_dt.append(tomorrow_dt)
            if collected_info.get("offered_next_week"): # User confirmed to add next week
                start_of_next_week_monday_dt = (current_processing_time - timedelta(days=current_processing_time.weekday()) + timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
                for i in range(5):
                    next_week_day_dt = start_of_next_week_monday_dt + timedelta(days=i)
                    if not any(d.date() == next_week_day_dt.date() for d in final_dates_to_book_dt):
                        final_dates_to_book_dt.append(next_week_day_dt)
            final_dates_to_book_dt = sorted(list(set(final_dates_to_book_dt)))

            if final_dates_to_book_dt:
                booking_process_result = await process_multi_date_booking(employee_id, final_dates_to_book_dt, collected_info)
                final_bot_response = f"Your booking request has been processed.\n\n{booking_process_result['confirmation_table']}"
            else:
                final_bot_response = "Okay, but it seems there were no valid dates to book. Please try again with different dates."
            is_flow_complete_for_response = True
        elif user_declines:
            final_bot_response = "Okay, I will not proceed with the booking. Is there anything else I can help you with?"
            is_flow_complete_for_response = True
        else: # Ambiguous
            final_bot_response = "I'm sorry, I didn't catch that. Do you want to proceed with the booking summary shown? (yes/no)"
            history.append({"role": "assistant", "content": final_bot_response})
            return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=False)

        clear_user_flow_state(employee_id, "book_seat")
        history.append({"role": "assistant", "content": final_bot_response})
        return ChatResponse(employee_id=employee_id, response=final_bot_response, is_complete=is_flow_complete_for_response)


    # --- LLM Intent Extraction and Main Flow Dispatch ---
    history.append({"role": "user", "content": message_text})
    info_for_llm = {k: v for k, v in collected_info.items() if k not in ['cancellable_bookings_cache', 'booking_history_cache', 'associate_api_data']}
    extracted_llm_info = await extract_info_with_llm(message_text, info_for_llm)
    
    # If intent changed, clear old intent-specific fields before updating
    if "intent" in extracted_llm_info and extracted_llm_info["intent"] != collected_info.get("intent"):
        clear_user_flow_state(employee_id, collected_info.get("intent")) 
    collected_info.update(extracted_llm_info)
    user_intent = collected_info.get("intent")

    # A. Cancellation Intent
    if user_intent == "cancel_seat":
        print("Processing cancellation intent with collected info:", collected_info)
        seat_to_cancel_str = collected_info.get("cancel_seat_number")
        date_desc_to_cancel = collected_info.get("cancel_date_description")
        access_token = await get_new_access_token()
        print("seat_to_cancel_str:", seat_to_cancel_str)
        print("date_desc_to_cancel:", date_desc_to_cancel)  
        if not access_token:
            final_bot_response = "Sorry, I couldn't get authorization to manage bookings. Please try again later."
            is_flow_complete_for_response = True
        elif seat_to_cancel_str and date_desc_to_cancel: # Specific cancellation request
            parsed_cancel_dates = parse_booking_days(date_desc_to_cancel)
            if not parsed_cancel_dates or len(parsed_cancel_dates) != 1: # Enforce single date for cancellation
                final_bot_response = "The date you provided for cancellation is unclear or specifies multiple dates. Please provide a single specific date (e.g., 'July 20th', 'tomorrow')."
                collected_info.pop("cancel_date_description", None) # Clear to ask again
                is_flow_complete_for_response = False
            else:
                cancel_date_obj = parsed_cancel_dates[0]
                all_bookings = await get_booking_history(access_token, employee_id)
                found_booking_to_cancel = None
                for booking in all_bookings:
                    try:
                        booking_date_dt = datetime.fromisoformat(booking.fromDate)
                        if booking.seat.lower() == seat_to_cancel_str.lower() and \
                           booking_date_dt.date() == cancel_date_obj.date():
                            found_booking_to_cancel = booking
                            break
                    except ValueError: continue # Error parsing date from history

                if found_booking_to_cancel:
                    if found_booking_to_cancel.status == 4: # Booked, can be cancelled
                        if datetime.fromisoformat(found_booking_to_cancel.fromDate).date() >= datetime.now().date():
                            collected_info["selected_allocation_id_for_cancel"] = found_booking_to_cancel.allocationID
                            collected_info["cancellation_step"] = "awaiting_specific_confirmation"
                            collected_info["specific_booking_details_for_cancel"] = {
                                "seat": found_booking_to_cancel.seat,
                                "date_str": cancel_date_obj.strftime('%B %d, %Y')
                            }
                            final_bot_response = (f"Found booking for seat {found_booking_to_cancel.seat} on "
                                                  f"{cancel_date_obj.strftime('%B %d, %Y')}. Are you sure you want to cancel it? (yes/no)")
                            is_flow_complete_for_response = False
                        else:
                            final_bot_response = f"The booking for seat {seat_to_cancel_str} on {cancel_date_obj.strftime('%B %d, %Y')} is in the past and cannot be cancelled."
                            clear_user_flow_state(employee_id, "cancel_seat")
                            is_flow_complete_for_response = True
                    elif found_booking_to_cancel.status == 1: # Already Cancelled
                        final_bot_response = f"The booking for seat {seat_to_cancel_str} on {cancel_date_obj.strftime('%B %d, %Y')} is already cancelled."
                        clear_user_flow_state(employee_id, "cancel_seat")
                        is_flow_complete_for_response = True
                    else: # Other statuses like Checked-In (2), Auto-Cancelled (3)
                        status_name = get_status_name(found_booking_to_cancel.status)
                        final_bot_response = f"The booking for seat {seat_to_cancel_str} on {cancel_date_obj.strftime('%B %d, %Y')} has a status of '{status_name}' and cannot be cancelled by you at this time."
                        clear_user_flow_state(employee_id, "cancel_seat")
                        is_flow_complete_for_response = True
                else:
                    final_bot_response = f"I couldn't find an active booking for seat {seat_to_cancel_str} on {cancel_date_obj.strftime('%B %d, %Y')}. Would you like to see a list of your cancellable bookings, or try a different seat/date?"
                    collected_info.pop("cancel_seat_number", None)
                    collected_info.pop("cancel_date_description", None)
                    is_flow_complete_for_response = False 
        
        elif seat_to_cancel_str and not date_desc_to_cancel:
            final_bot_response = f"For which date would you like to cancel the booking for seat {seat_to_cancel_str}? Please provide a single specific date."
            is_flow_complete_for_response = False
        elif not seat_to_cancel_str and date_desc_to_cancel:
            final_bot_response = f"Which seat booking would you like to cancel for {date_desc_to_cancel}?"
            is_flow_complete_for_response = False
        else: # General "cancel" request, or if specific details were not fully provided
            booking_hist_items = await get_booking_history(access_token, employee_id)
            cancellable_bookings = filter_and_sort_cancellable_bookings(booking_hist_items) 
            
            if not cancellable_bookings:
                final_bot_response = "You have no active bookings that can be cancelled at this time."
                clear_user_flow_state(employee_id, "cancel_seat")
                is_flow_complete_for_response = True
            else:
                response_parts = ["Here are your bookings that can be cancelled (Status 4 = Booked):\n"]
                for i, booking in enumerate(cancellable_bookings):
                    dt_obj = datetime.fromisoformat(booking.fromDate)
                    response_parts.append(f"{i+1}. {booking.seat} on {dt_obj.strftime('%d %b %Y')} in {booking.building} ({booking.location})")
                response_parts.append("\nPlease enter the number of the booking you wish to cancel, or say 'none'.")
                final_bot_response = "\n".join(response_parts)
                collected_info["cancellable_bookings_cache"] = cancellable_bookings
                collected_info["cancellation_step"] = "awaiting_selection" 
                is_flow_complete_for_response = False

    # B. View Booking History Intent
    elif user_intent == "view_booking_history":
        access_token = await get_new_access_token()
        if not access_token:
            final_bot_response = "Sorry, I couldn't get authorization to check your booking history. Please try again later."
        else:
            booking_hist_items = await get_booking_history(access_token, employee_id)
            if not booking_hist_items:
                final_bot_response = "You have no booking history."
            else:
                booking_hist_items.sort(key=lambda item: datetime.fromisoformat(item.fromDate), reverse=True)
                
                try:
                    count_str = collected_info.get("history_count", "5") 
                    history_count_to_show = int(re.search(r'\d+', count_str).group(0) if re.search(r'\d+', count_str) else 5)
                except (ValueError, AttributeError):
                    history_count_to_show = 5
                
                history_to_show = booking_hist_items[:history_count_to_show]
                
                response_parts = [f"Here are your latest {len(history_to_show)} booking(s):\n"]
                for item in history_to_show:
                    dt_obj = datetime.fromisoformat(item.fromDate)
                    status_name = get_status_name(item.status)
                    response_parts.append(
                        f"- Seat: {item.seat}, Date: {dt_obj.strftime('%d %b %Y, %I:%M %p')}, "
                        f"Building: {item.building}, Floor: {item.floor}, Status: {status_name}"
                    )
                final_bot_response = "\n".join(response_parts)
        clear_user_flow_state(employee_id, "view_booking_history")
        is_flow_complete_for_response = True

    # C. Booking Intent
    elif user_intent == "book_seat":
        # Required fields for booking are: booking_days_description, building_no, floor, seat_number
        all_required_booking_fields_collected = (
            collected_info.get("booking_days_description") and
            collected_info.get("building_no") and
            collected_info.get("floor") and
            collected_info.get("seat_number")
        )

        if all_required_booking_fields_collected:
            llm_ack_response = await get_llm_response_for_booking(message_text, history, collected_info)
            history.append({"role": "assistant", "content": llm_ack_response}) 

            current_time_for_rules = datetime.now()
            parsed_dates = parse_booking_days(collected_info["booking_days_description"])
            valid_dates_for_preview, user_notifications = [], []
            offer_tomorrow_instead, offer_next_week_instead = False, False

            for date_obj in parsed_dates:
                is_today, day_name = (date_obj.date() == current_time_for_rules.date()), date_obj.strftime('%A, %d %b')
                if date_obj.weekday() >= 5: user_notifications.append(f"Bookings are not available on weekends. {day_name} skipped.")
                elif is_today and date_obj.weekday() <= 3 and current_time_for_rules.hour >= 17:
                    user_notifications.append(f"Booking for today ({day_name}) is closed (it's past 5 PM).")
                    offer_tomorrow_instead = True
                elif is_today and date_obj.weekday() == 4 and current_time_for_rules.hour >= 17:
                    user_notifications.append(f"Booking for today ({day_name}) is closed (it's past 5 PM on Friday).")
                    offer_next_week_instead = True
                elif date_obj < current_time_for_rules.replace(hour=0, minute=0, second=0, microsecond=0):
                    user_notifications.append(f"Cannot book for a past date: {day_name}. Skipped.")
                else: valid_dates_for_preview.append(date_obj)
            
            valid_dates_for_preview = sorted(list(set(valid_dates_for_preview)))
            collected_info["booking_dates_iso"] = [d.isoformat() for d in valid_dates_for_preview]
            collected_info["awaiting_booking_confirmation"] = True
            collected_info["offered_tomorrow"] = offer_tomorrow_instead
            collected_info["offered_next_week"] = offer_next_week_instead

            response_parts = [""]
            if user_notifications: response_parts.append("\n" + "\n".join(user_notifications))
            if valid_dates_for_preview:
                preview_table = "| Date | Building | Floor | Seat Number |\n|------|----------|-------|-------------|\n"
                for date_obj in valid_dates_for_preview:
                    preview_table += f"| {date_obj.strftime('%d.%b.%Y')} | {collected_info['building_no']} | {collected_info['floor']} | {collected_info['seat_number']} |\n"
                response_parts.append(f"\nHere's a summary for the valid dates based on your request:\n\n{preview_table}")
            
            alt_prompts = []
            if offer_tomorrow_instead and not any(d.date() == (current_time_for_rules + timedelta(days=1)).date() for d in valid_dates_for_preview):
                 alt_prompts.append("Would you like to book for tomorrow as well?")
            if offer_next_week_instead: alt_prompts.append("Would you like to book for next week (Mon-Fri) instead/as well?")
            if alt_prompts: response_parts.append("\n" + " ".join(alt_prompts))

            if valid_dates_for_preview or alt_prompts: 
                response_parts.append("\n\n**Do you want to proceed with this booking?** (yes/no)")
            else: 
                response_parts.append("\nIt seems there are no valid dates to book based on your request and current booking rules. Please try different dates or criteria.")
                clear_user_flow_state(employee_id, "book_seat") 
                is_flow_complete_for_response = True
            final_bot_response = "\n".join(filter(None,response_parts)) # Filter out empty strings before joining
            is_flow_complete_for_response = is_flow_complete_for_response or False # Keep false if not set true by no valid dates
        else:
            final_bot_response = await get_llm_response_for_booking(message_text, history, collected_info)
            is_flow_complete_for_response = False
    
    # D. General Query or Unclear Intent
    else:
        final_bot_response = "I can help you book a seat (I'll need the days, building, floor, and seat number), cancel an existing booking (I'll need the seat number and specific date), or view your booking history. Please specify what you'd like to do."
        if history and history[-1]["role"] == "assistant" and "book a seat" in history[-1]["content"].lower(): # Avoid repeating generic help
             final_bot_response = "Sorry, I'm not sure how to help with that. You can ask me to 'book a seat', 'cancel a booking', or 'view booking history'."
        clear_user_flow_state(employee_id, None) # Clear any stale intent
        is_flow_complete_for_response = False # Open-ended

    history.append({"role": "assistant", "content": final_bot_response})
    
    return ChatResponse(
        employee_id=employee_id,
        response=final_bot_response,
        is_complete=is_flow_complete_for_response
    )

@app.get("/")
async def root():
    return {"message": "Bosch seat booking Chatbot API is running version:3.10"}