# functions/main.py

from firebase_functions import https_fn
import firebase_admin
from firebase_admin import firestore

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
# from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler, Filters, CallbackContext # Not directly used here

import requests
import json
import os
import base64
import re
from datetime import datetime, timedelta

# --- Global Variables (will be initialized in the webhook handler's call to initialize_services) ---
db = None
bot = None
ADMIN_USER_ID = None


# --- Firebase Admin SDK and Bot Initialization (Idempotent) ---
def initialize_services():
    global db, bot, ADMIN_USER_ID
    # Initialize Firebase Admin SDK
    if not firebase_admin._apps:
        try:
            firebase_admin.initialize_app()  # Default initialization uses ADC or service account in cloud
            print("INIT: Firebase Admin SDK initialized.")
        except ValueError as e:  # Catches "already initialized"
            print(f"INIT: Firebase Admin SDK already initialized or error: {e}")
    db = firestore.client()

    # Initialize Telegram Bot
    telegram_token_env = os.environ.get("TELEGRAM_TOKEN")
    if telegram_token_env:
        # Only re-initialize if token changed or bot is None
        if bot is None or bot.token != telegram_token_env:
            bot = telegram.Bot(token=telegram_token_env)
            print("INIT: Telegram Bot initialized/re-initialized.")
    elif bot is None:  # Only print critical if bot couldn't be initialized at all
        print("INIT CRITICAL: TELEGRAM_TOKEN environment variable not set. Bot cannot be initialized.")

    # Load Admin User ID
    admin_user_id_env = os.environ.get("ADMIN_USER_ID")
    if admin_user_id_env:
        ADMIN_USER_ID = admin_user_id_env
        print(f"INIT: Admin User ID loaded: {ADMIN_USER_ID}")
    elif ADMIN_USER_ID is None:  # Only print critical if not loaded at all
        print("INIT WARNING: ADMIN_USER_ID environment variable not set. Admin features will not work correctly.")

    print("INIT: initialize_services finished.")


# --- LLM Interaction and JSON Cleaning ---
def clean_llm_json_output(json_string_from_llm: str) -> str:
    """Cleans the JSON string output from LLM."""
    # Remove markdown fences (```json ... ``` or just ``` ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_string_from_llm, re.IGNORECASE)
    if match:
        cleaned_json_string = match.group(1)
    else:
        cleaned_json_string = json_string_from_llm
    return cleaned_json_string.strip()


def call_llm_for_receipt(image_bytes: bytes, image_mime_type: str = "image/jpeg") -> dict:
    """Sends image to Gemini, gets JSON receipt data, cleans, and parses it."""
    gemini_api_key_env = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key_env:
        print("LLM_CALL ERROR: GEMINI_API_KEY not set.")
        raise ValueError("Gemini API Key is not configured for the system.")

    gemini_model_name = "gemini-2.0-flash"  # Or "gemini-pro-vision", "gemini-1.5-pro-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:generateContent?key={gemini_api_key_env}"
    base64_image_data = base64.b64encode(image_bytes).decode('utf-8')

    prompt_text = """
    You are an expert AI assistant specializing in processing grocery receipts.
    Your task is to analyze the provided image of a grocery receipt, which is primarily in Hebrew.
    You must extract specific information, translate relevant Hebrew text to English, categorize items, and return ALL output as a single, well-formed JSON object matching the specified schema.

    Input: An image file containing a single grocery receipt written primarily in Hebrew.
    Processing Steps & Extraction Rules:
    1.  Overall Receipt Information (Translate to English where specified):
        *   `store_name` (String, English): Extract the store name and translate to English.
        *   `date` (String, `YYYY-MM-DD`): Extract transaction date. Convert to `YYYY-MM-DD`.
        *   `total_price` (Number): Extract final total amount paid. If not found, use `null`.
        *   `currency_code` (String, optional): If identifiable (e.g., ₪), output "ILS". Else, `null` or omit.
    2.  Line Items Processing (Translate item names and units to English):
        *   For each item, create an object:
            *   `item_name` (String, English): Extract item description (Hebrew) and translate to English.
            *   `item_price` (Number): Extract total price for this line item. If not found, use `null`.
            *   `grocery_category` (String, English): From translated name, assign ONE category from: "Produce", "Dairy & Eggs", "Meat & Seafood", "Bakery", "Pantry Staples", "Frozen Foods", "Beverages (Non-alcoholic)", "Alcohol", "Snacks & Sweets", "Household Supplies", "Personal Care", "Baby Items", "Pet Supplies", "Other". If unsure, use "Other".
            *   `quantity` (Number, optional): Extract item quantity. If not stated or clearly '1', assume 1. If unknown/NA, use `null`.
            *   `price_per_unit` (Number, optional): If listed, extract. If calculable (`item_price` / `quantity`), calculate. If single unit (`quantity`=1), can be `item_price`. If indeterminable/NA, use `null`.
            *   `unit_of_measurement` (String, English, optional): Extract unit. Translate: "ק\"ג"->"kg", "גרם"->"g", "ליטר"->"L", "מ\"ל"->"ml", "יח'", "יחידה"->"unit", "אריזה", "חבילה"->"pack". If unknown/NA, use "unit" or `null`.
    Output Format (Strict JSON):
    Return entire response as a single, valid JSON object. No explanatory text. Schema:
    ```json
    {
      "store_name": "Example Supermarket", "date": "2024-05-15", "total_price": 138.50, "currency_code": "ILS",
      "items": [
        { "item_name": "Milk 3%", "item_price": 6.20, "grocery_category": "Dairy & Eggs", "quantity": 1, "price_per_unit": 6.20, "unit_of_measurement": "L" }
      ]
    }
    ```
    Important Notes for AI: Prioritize accuracy. Adherence to JSON schema and data types (String, Number, `null`) is mandatory. Use `null` for genuinely unavailable information.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt_text},
                                {"inline_data": {"mime_type": image_mime_type, "data": base64_image_data}}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1, "maxOutputTokens": 16384}
    }
    headers = {'Content-Type': 'application/json'}

    print(f"LLM_CALL: Sending request to Gemini API for receipt processing. Model: {gemini_model_name}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)  # Generous timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)

        response_json_data_from_api = response.json()

        if response_json_data_from_api.get("promptFeedback"):
            prompt_feedback = response_json_data_from_api["promptFeedback"]
            if prompt_feedback.get("blockReason"):
                block_reason_detail = prompt_feedback.get("blockReasonMessage", "No additional details.")
                print(
                    f"LLM_CALL ERROR: Prompt was blocked. Reason: {prompt_feedback['blockReason']}. Details: {block_reason_detail}")
                raise ValueError(f"LLM prompt blocked: {prompt_feedback['blockReason']}. {block_reason_detail}")
            if prompt_feedback.get("safetyRatings"):
                for rating in prompt_feedback.get("safetyRatings", []):
                    if rating.get("probability") not in ["NEGLIGIBLE", "LOW"]:  # Check for harmful content in prompt
                        print(
                            f"LLM_CALL WARNING: Prompt Safety Rating - Category: {rating.get('category')}, Probability: {rating.get('probability')}")

        if not response_json_data_from_api.get('candidates'):
            print(f"LLM_CALL ERROR: No candidates in Gemini response. Full Response: {response_json_data_from_api}")
            raise ValueError("LLM response error: No candidates found in API response.")

        candidate = response_json_data_from_api['candidates'][0]
        finish_reason = candidate.get('finishReason')

        print(f"LLM_CALL INFO: Generation finishReason: {finish_reason}")

        # Handle different finish reasons
        if finish_reason not in [None, "STOP", "MAX_TOKENS", "MODEL_LENGTH"]:
            # "MODEL_LENGTH" is similar to "MAX_TOKENS" for some models/contexts.
            # Other reasons: "SAFETY", "RECITATION", "OTHER"
            safety_ratings_text = ""
            if candidate.get("safetyRatings"):
                safety_ratings_text = " SafetyRatings: " + json.dumps(candidate.get("safetyRatings"))

            print(
                f"LLM_CALL WARNING: Generation finished due to an unusual reason: {finish_reason}.{safety_ratings_text}")
            if not candidate.get('content') or not candidate['content']['parts']:  # If no content due to these reasons
                raise ValueError(
                    f"LLM generation stopped due to '{finish_reason}' and returned no usable content.{safety_ratings_text}")
            # If there's content despite the finish reason, proceed but be wary.
            # You might want to inform the user or flag for review.

        elif finish_reason in ["MAX_TOKENS", "MODEL_LENGTH"]:
            print(
                f"LLM_CALL WARNING: Output may be truncated as it reached the token limit (finishReason: {finish_reason}).")
            # Proceed with parsing, but the JSON might be incomplete.

        if not candidate.get('content') or not candidate['content']['parts'] or not candidate['content']['parts'][
            0].get('text'):
            # This can happen even with a "STOP" reason if the model generates an empty response, or if responseMimeType: application/json failed.
            print(
                f"LLM_CALL ERROR: No text content found in candidate part, even if finishReason was '{finish_reason}'. Candidate: {candidate}")
            raise ValueError("LLM response error: No text content found in the candidate.")

        # The actual JSON string is usually in the first part of the first candidate's content
        generated_json_str_from_llm = candidate['content']['parts'][0]['text']

        print(f"LLM_CALL: Raw string from Gemini (first 500 chars): {generated_json_str_from_llm[:500]}...")

        cleaned_json_str = clean_llm_json_output(generated_json_str_from_llm)
        print(f"LLM_CALL: Cleaned JSON string (first 500 chars): {cleaned_json_str[:500]}...")

        parsed_json_dict = json.loads(cleaned_json_str)
        print("LLM_CALL: Successfully parsed JSON from LLM.")
        return parsed_json_dict  # Return the parsed Python dictionary

    except requests.exceptions.Timeout:
        print(f"LLM_CALL ERROR: Request to Gemini API timed out after 180 seconds.")
        raise ValueError("LLM API request timed out. Please try again later.")
    except requests.exceptions.RequestException as e:
        print(f"LLM_CALL ERROR: Network or API error calling Gemini: {e}")
        # Log response details if available
        if hasattr(e, 'response') and e.response is not None:
            print(f"LLM_CALL ERROR: Gemini API Response Status: {e.response.status_code}")
            try:
                print(f"LLM_CALL ERROR: Gemini API Response Body: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"LLM_CALL ERROR: Gemini API Response Body (not JSON): {e.response.text}")
        raise ValueError(f"Network or API error communicating with LLM: {str(e)}")
    except json.JSONDecodeError as je:
        print(f"LLM_CALL CRITICAL ERROR: Failed to decode JSON from LLM after cleaning.")
        print(
            f"LLM_CALL INFO: Original string from LLM was: {generated_json_str_from_llm if 'generated_json_str_from_llm' in locals() else 'N/A'}")
        print(
            f"LLM_CALL INFO: Cleaned string attempted to parse was: {cleaned_json_str if 'cleaned_json_str' in locals() else 'N/A'}")
        print(f"LLM_CALL INFO: JSONDecodeError: {je}")
        raise ValueError(f"LLM returned invalid JSON format: {je.msg}. Please check the raw output for issues.")
    except (KeyError, IndexError) as e:
        print(f"LLM_CALL ERROR: Error parsing expected structure from Gemini response: {e}")
        print(
            f"LLM_CALL INFO: Full Gemini Response: {response_json_data_from_api if 'response_json_data_from_api' in locals() else 'N/A'}")
        raise ValueError(f"LLM response structure parsing error: {e}")


# --- Firestore Group and User Profile Helper Functions ---
def get_user_profile(user_id_str: str) -> dict or None:
    """Fetches user profile from Firestore, returns data dict or None if not found."""
    if not db:
        print("DB_HELPER ERROR: Firestore db client not initialized in get_user_profile.")
        return None
    profile_ref = db.collection('user_profiles').document(user_id_str)
    profile_doc = profile_ref.get()
    return profile_doc.to_dict() if profile_doc.exists else None


def ensure_user_profile_exists(user_id_str: str, chat_id_for_message: int = None) -> dict:
    """Ensures a user profile exists, creates a basic one if not. Returns profile data."""
    if not db:
        print("DB_HELPER ERROR: Firestore db client not initialized in ensure_user_profile_exists.")
        # Attempt to initialize again, or raise error
        initialize_services()
        if not db:
            raise ConnectionError("Database client could not be initialized.")

    profile = get_user_profile(user_id_str)
    if not profile:
        is_admin_creating_own_profile = (user_id_str == ADMIN_USER_ID and ADMIN_USER_ID is not None)
        default_status = 'approved' if is_admin_creating_own_profile else 'pending_approval'

        new_profile_data = {
            'telegramUserId': user_id_str,
            'groupId': None,
            'status': default_status,
            'requestedAt': firestore.SERVER_TIMESTAMP,
            'createdAt': firestore.SERVER_TIMESTAMP  # Use createdAt for the very first record
        }
        db.collection('user_profiles').document(user_id_str).set(new_profile_data)
        print(f"PROFILE_ENSURE: Created new user profile for {user_id_str} with status: {default_status}")

        if default_status == 'pending_approval' and chat_id_for_message and bot:
            try:
                welcome_text = (f"Welcome! To use this service, your account needs approval.\n"
                                f"Your User ID is: `{user_id_str}`\n"
                                f"Please send this ID to the administrator.")
                bot.send_message(chat_id=chat_id_for_message, text=welcome_text, parse_mode='Markdown')
            except telegram.error.TelegramError as te:
                print(
                    f"PROFILE_ENSURE: Telegram API Error sending pending approval message to {chat_id_for_message}: {te}")
            except Exception as e:  # Catch any other exception during send_message
                print(f"PROFILE_ENSURE: Generic error sending pending approval message to {chat_id_for_message}: {e}")
        return new_profile_data

    # If profile exists but status is missing (e.g. old data), update it
    elif 'status' not in profile:
        db.collection('user_profiles').document(user_id_str).update({
            'status': 'pending_approval',
            'updatedAt': firestore.SERVER_TIMESTAMP
        })
        profile['status'] = 'pending_approval'  # Update in-memory dict too
        print(f"PROFILE_ENSURE: Updated profile for {user_id_str}, added missing status: pending_approval")
    return profile


def get_user_group_id(user_id_str: str) -> str or None:
    """Gets the groupId for a user from their profile, or None."""
    profile = get_user_profile(user_id_str)
    return profile.get('groupId') if profile else None


def get_group_name(group_id: str) -> str or None:
    """Fetches group name from Firestore."""
    if not group_id or not db: return None
    group_doc = db.collection('groups').document(group_id).get()
    return group_doc.to_dict().get('groupName') if group_doc.exists else None


def is_user_approved(user_id_str: str) -> bool:
    """Checks if user is approved. Admin is always considered approved."""
    if user_id_str == ADMIN_USER_ID and ADMIN_USER_ID is not None: return True
    profile = get_user_profile(user_id_str)
    return profile and profile.get('status') == 'approved'


def is_user_admin(user_id_str: str) -> bool:
    """Checks if the user is THE admin."""
    return user_id_str == ADMIN_USER_ID and ADMIN_USER_ID is not None


def create_group_in_firestore(acting_user_id_str: str, group_name_str: str, initial_member_ids: list = None) -> (
str or None, str):
    """Admin creates a group. Regular approved users can create if not in a group."""
    if not group_name_str.strip(): return None, "Group name cannot be empty."

    is_admin_actor = is_user_admin(acting_user_id_str)

    if not is_admin_actor:  # Regular user trying to create
        if not is_user_approved(acting_user_id_str):
            return None, "Your account is not approved to create groups."
        if get_user_group_id(acting_user_id_str):
            return None, "You are already in a group. Leave your current group first with /leavegroup."

    # If admin is creating, initial_member_ids can be specified.
    # If regular user is creating, they are the only initial member.
    members_to_add = [acting_user_id_str]  # Creator is always a member
    if is_admin_actor and initial_member_ids:
        for member_id in initial_member_ids:
            if member_id not in members_to_add:
                profile_to_add = get_user_profile(member_id)
                if profile_to_add and profile_to_add.get('status') == 'approved':
                    members_to_add.append(member_id)
                else:
                    print(
                        f"GROUP_CREATE: Admin trying to add non-existent or non-approved user {member_id} to new group. Skipping.")

    new_group_ref = db.collection('groups').document()  # Auto-generate ID
    group_data = {
        'groupName': group_name_str.strip(),
        'ownerId': acting_user_id_str,  # The user who initiated creation
        'createdAt': firestore.SERVER_TIMESTAMP,
        'memberUserIds': list(set(members_to_add))  # Ensure unique members
    }
    new_group_ref.set(group_data)
    group_id = new_group_ref.id

    # Update profiles for all initial members
    # This will overwrite existing groupId if admin adds someone already in another group.
    for member_id in members_to_add:
        member_profile_ref = db.collection('user_profiles').document(member_id)
        member_profile_ref.set({'groupId': group_id, 'groupJoinedAt': firestore.SERVER_TIMESTAMP}, merge=True)

    return group_id, f"Group '{group_name_str}' created with ID: `{group_id}`."


def create_group_in_firestore(acting_user_id_str: str, group_name_str: str, initial_member_ids: list = None) -> (str or None, str):
    """
    Handles group creation.
    - Admin can create groups for specified initial members. Admin's own profile groupId is NOT changed.
    - Regular approved users can create a group for themselves if not already in one.
    Returns (group_id, message) or (None, error_message).
    """
    if not db: return None, "Database not initialized."
    if not group_name_str.strip(): return None, "Group name cannot be empty."

    is_admin_actor = is_user_admin(acting_user_id_str)
    
    # --- Authorization and Pre-checks ---
    if not is_admin_actor: # Regular user trying to create
        if not is_user_approved(acting_user_id_str):
            return None, "Your account is not approved to create groups."
        if get_user_group_id(acting_user_id_str):
            return None, "You are already in a group. To create a new one, please leave your current group first using the 'Group Options' menu."
            
    # --- Determine Initial Members ---
    members_to_add_to_group = [] # List of user IDs who will have their profile updated
    
    if is_admin_actor:
        print(f"GROUP_CREATE: Admin {acting_user_id_str} creating group '{group_name_str}'.")
        if initial_member_ids: # Admin specified members
            print(f"GROUP_CREATE: Admin provided initial members: {initial_member_ids}")
            for member_id in initial_member_ids:
                member_id = str(member_id).strip() # Ensure string and clean
                if not member_id: continue

                profile_to_add = get_user_profile(member_id)
                if profile_to_add and profile_to_add.get('status') == 'approved':
                    # Check if this member is already in another group (optional but good)
                    existing_group = get_user_group_id(member_id)
                    if existing_group:
                         print(f"GROUP_CREATE: Warning - User {member_id} (being added by admin) is already in group {existing_group}. They will be moved.")
                         # Consider adding logic here if you DON'T want to move them automatically.
                    members_to_add_to_group.append(member_id)
                else:
                    print(f"GROUP_CREATE: Admin trying to add non-existent or non-approved user {member_id}. Skipping.")
        else:
            # Admin created a group but specified no members. Group starts empty.
             print(f"GROUP_CREATE: Admin created group '{group_name_str}' with no initial members.")
             pass
             
    else: # Regular user creates group - they are the only initial member
        print(f"GROUP_CREATE: User {acting_user_id_str} creating group '{group_name_str}'.")
        members_to_add_to_group = [acting_user_id_str]

    # --- Create Group Document ---
    new_group_ref = db.collection('groups').document() # Auto-generate ID
    group_id = new_group_ref.id
    group_data = {
        'groupName': group_name_str.strip(),
        'ownerId': acting_user_id_str, # The creator is always the owner
        'createdAt': firestore.SERVER_TIMESTAMP,
        'memberUserIds': list(set(members_to_add_to_group)) # Store the unique list of initial members
    }
    try:
        new_group_ref.set(group_data)
        print(f"GROUP_CREATE: Successfully created group document {group_id} in Firestore.")
    except Exception as e_set:
        print(f"GROUP_CREATE ERROR: Failed to set group document {group_id}: {e_set}")
        return None, "An error occurred while creating the group document."

    # --- Update Profiles for Initial Members ---
    update_errors = []
    for member_id in members_to_add_to_group:
        try:
            member_profile_ref = db.collection('user_profiles').document(member_id)
            # Set the groupID for this member. merge=True ensures other profile fields aren't lost.
            member_profile_ref.set({'groupId': group_id, 'groupJoinedAt': firestore.SERVER_TIMESTAMP}, merge=True)
            print(f"GROUP_CREATE: Updated profile for member {member_id}, set groupId to {group_id}.")
        except Exception as e_update:
            print(f"GROUP_CREATE ERROR: Failed to update profile for member {member_id}: {e_update}")
            update_errors.append(member_id)

    # --- Prepare Response Message ---
    success_message = f"Group '{group_name_str}' created with ID: `{group_id}`."
    if is_admin_actor:
        if members_to_add_to_group:
            success_message += f"\nAdded members: {', '.join([f'`{mid}`' for mid in members_to_add_to_group])}."
        else:
             success_message += "\nNo initial members were added."
    else: # Regular user created
         success_message += "\nYou have been added as the first member."

    if update_errors:
        success_message += f"\nWarning: Failed to update profile for users: {', '.join([f'`{eid}`' for eid in update_errors])}."

    return group_id, success_message


def user_leave_group(user_id_str: str) -> str:
    """Allows an approved user (not admin) to leave their current group."""
    if not is_user_approved(user_id_str) or is_user_admin(user_id_str):  # Admin uses admin commands
        return "This action is for regular approved users. Admins should use admin commands for group management."
    if not db: return "Database not initialized."

    current_group_id = get_user_group_id(user_id_str)
    if not current_group_id: return "You are not currently in any group."

    group_ref = db.collection('groups').document(current_group_id)
    group_doc = group_ref.get()
    group_name = get_group_name(current_group_id) or "your current group"

    db.collection('user_profiles').document(user_id_str).update({
        'groupId': firestore.DELETE_FIELD,  # Or None
        'groupLeftAt': firestore.SERVER_TIMESTAMP
    })
    if group_doc.exists:
        group_ref.update({'memberUserIds': firestore.ArrayRemove([user_id_str])})
        # Consider deleting group if owner leaves and it's empty, or if members array becomes empty
        group_data_after_leave = group_ref.get().to_dict()  # Get fresh data
        if group_data_after_leave and not group_data_after_leave.get('memberUserIds'):
            print(f"GROUP_LEAVE: Group {current_group_id} ('{group_name}') is now empty. Deleting.")
            group_ref.delete()
            return f"You have left '{group_name}'. The group is now empty and has been deleted."

    return f"You have left '{group_name}'."


def get_my_group_info(user_id_str: str) -> str:
    """Gets information about the user's current group."""
    if not db: return "Database not initialized."
    current_group_id = get_user_group_id(user_id_str)
    if not current_group_id: return "You are not in a group. Admin can add you or you can ask admin to create one for you."

    group_ref = db.collection('groups').document(current_group_id)
    group_doc = group_ref.get()
    if not group_doc.exists:
        db.collection('user_profiles').document(user_id_str).update({'groupId': firestore.DELETE_FIELD})
        return "Error: Your group data was inconsistent. You've been removed from the non-existent group."

    data = group_doc.to_dict()
    info = f"Group: '{data.get('groupName', 'Unnamed')}' (ID: `{current_group_id}`)\n"
    info += f"Owner ID: `{data.get('ownerId')}` {'(You)' if data.get('ownerId') == user_id_str else ''}\n"
    member_ids = data.get('memberUserIds', [])
    info += f"Members ({len(member_ids)}):\n"
    for member_id in member_ids:
        info += f"  - `{member_id}` {'(You)' if member_id == user_id_str else ''}\n"
    return info


# --- Admin Specific Functions ---
def admin_list_users(status_filter: str = "all") -> str:
    if not db: return "Database not initialized."
    users_ref = db.collection('user_profiles')
    query = users_ref
    if status_filter in ["pending_approval", "approved", "banned"]:
        query = users_ref.where('status', '==', status_filter)

    docs = query.order_by('createdAt', direction=firestore.Query.DESCENDING).limit(50).stream()
    user_list_items = []
    for doc in docs:
        data = doc.to_dict()
        user_list_items.append(
            f"- ID: `{data.get('telegramUserId')}` (Status: {data.get('status', 'N/A')}, Group: `{data.get('groupId', 'None')}`)")

    if not user_list_items: return f"No users found with status '{status_filter}'."
    return f"Users (Status: {status_filter}):\n" + "\n".join(user_list_items)


def admin_set_user_status(user_id_to_modify: str, new_status: str) -> str:
    if new_status not in ["approved", "banned", "pending_approval"]:
        return "Invalid status. Use 'approved', 'banned', or 'pending_approval'."
    if not db: return "Database not initialized."

    user_profile_ref = db.collection('user_profiles').document(user_id_to_modify)
    user_doc = user_profile_ref.get()
    if not user_doc.exists: return f"User ID `{user_id_to_modify}` not found."

    user_profile_ref.update({'status': new_status, 'statusUpdatedAt': firestore.SERVER_TIMESTAMP})

    if bot:
        try:
            if new_status == "approved":
                bot.send_message(chat_id=user_id_to_modify,
                                 text="Your account has been approved! You can now use the bot. Send /menu or an image.")
            elif new_status == "banned":
                bot.send_message(chat_id=user_id_to_modify,
                                 text="Your account access has been restricted by the administrator.")
        except telegram.error.TelegramError as te:
            print(f"ADMIN_SET_STATUS: Telegram error sending notification to user {user_id_to_modify}: {te}")
        except Exception as e:
            print(f"ADMIN_SET_STATUS: Generic error sending notification to user {user_id_to_modify}: {e}")
    return f"User `{user_id_to_modify}` status set to '{new_status}'."


def admin_remove_user_from_group(user_id_to_remove: str, group_id_from_which_to_remove: str) -> str:
    if not db: return "Database not initialized."
    user_profile_ref = db.collection('user_profiles').document(user_id_to_remove)
    user_profile_doc = user_profile_ref.get()
    if not user_profile_doc.exists: return f"User ID `{user_id_to_remove}` not found."

    group_ref = db.collection('groups').document(group_id_from_which_to_remove)
    group_doc = group_ref.get()
    if not group_doc.exists: return f"Group ID `{group_id_from_which_to_remove}` not found."
    group_name = group_doc.to_dict().get('groupName', group_id_from_which_to_remove)

    group_ref.update({'memberUserIds': firestore.ArrayRemove([user_id_to_remove])})

    # Only clear groupId in user's profile if it matches the group they are being removed from
    user_current_group = user_profile_doc.to_dict().get('groupId')
    if user_current_group == group_id_from_which_to_remove:
        user_profile_ref.update({'groupId': firestore.DELETE_FIELD, 'groupLeftAt': firestore.SERVER_TIMESTAMP})

    # Check if group became empty after removal
    updated_group_doc = group_ref.get()
    if updated_group_doc.exists and not updated_group_doc.to_dict().get('memberUserIds'):
        print(
            f"ADMIN_REMOVE_USER: Group {group_id_from_which_to_remove} ('{group_name}') is now empty after removing {user_id_to_remove}. Deleting group.")
        group_ref.delete()
        return f"User `{user_id_to_remove}` removed from group '{group_name}'. The group was empty and has been deleted."
    return f"User `{user_id_to_remove}` removed from group '{group_name}'."


def admin_delete_group(group_id_to_delete: str) -> str:
    if not db: return "Database not initialized."
    group_ref = db.collection('groups').document(group_id_to_delete)
    group_doc = group_ref.get()
    if not group_doc.exists: return f"Group ID `{group_id_to_delete}` not found."

    group_data = group_doc.to_dict()
    group_name = group_data.get('groupName', group_id_to_delete)
    member_ids = group_data.get('memberUserIds', [])

    # Clear groupId for all members who were in this group
    # Using a loop for clarity; batch could be used for >500 members but unlikely here.
    for member_id in member_ids:
        member_profile_ref = db.collection('user_profiles').document(member_id)
        member_profile = member_profile_ref.get()
        if member_profile.exists and member_profile.to_dict().get('groupId') == group_id_to_delete:
            member_profile_ref.update({'groupId': firestore.DELETE_FIELD, 'groupLeftAt': firestore.SERVER_TIMESTAMP})

    group_ref.delete()
    return f"Group '{group_name}' (ID: `{group_id_to_delete}`) and its members' associations have been deleted."


# --- Firestore Aggregation Functions ---
def get_query_target(user_id_str: str) -> (str, str, str):
    """Determines query field, value, and label based on user's group status."""
    if not db: initialize_services()  # Ensure db is available
    group_id = get_user_group_id(user_id_str)
    if group_id:
        group_name_str = get_group_name(group_id) or "Unnamed Group"
        return 'groupId', group_id, f"for group '{group_name_str}'"
    return 'telegramUserId', user_id_str, "(personal)"


def _aggregate_spending(user_id_str: str, start_date_str: str, end_date_str: str, mode: str = "total_by_date") -> str:
    """Generic helper for date-bound spending aggregations."""
    query_field, query_value, context_label = get_query_target(user_id_str)
    receipts_ref = db.collection('receipts')

    # Build the base query
    query = receipts_ref.where(query_field, '==', query_value)

    # Add date filters if provided (Firestore requires date strings to be comparable, e.g., YYYY-MM-DD)
    if start_date_str:
        query = query.where('date', '>=', start_date_str)
    if end_date_str:
        query = query.where('date', '<=', end_date_str)

    total_spent = 0.0
    receipt_count = 0
    category_spending = {}
    store_spending = {}
    currency_code = ""  # Attempt to get from first receipt

    docs_stream = query.stream()
    for doc_snapshot in docs_stream:
        data = doc_snapshot.to_dict()
        current_total = data.get('total_price')
        if not isinstance(current_total, (int, float)):  # Ensure it's a number
            print(f"AGG_WARN: Receipt {doc_snapshot.id} has invalid total_price: {current_total}. Skipping.")
            continue

        total_spent += current_total
        receipt_count += 1
        if not currency_code: currency_code = data.get('currency_code', '')

        if mode == "by_category":
            for item in data.get('items', []):
                category = item.get('grocery_category', 'Uncategorized')
                item_price = item.get('item_price')
                if isinstance(item_price, (int, float)):
                    category_spending[category] = category_spending.get(category, 0.0) + item_price
        elif mode == "by_store":
            store_name = data.get('store_name', 'Unknown Store')
            store_spending[store_name] = store_spending.get(store_name, 0.0) + current_total

    if receipt_count == 0:
        date_range_msg = f"between {start_date_str} and {end_date_str}" if start_date_str and end_date_str else "for the period"
        return f"No receipts found {context_label} {date_range_msg}."

    currency_display = currency_code if currency_code else ""

    if mode == "total_by_date":
        return f"Total spent {context_label} ({receipt_count} receipts): {total_spent:.2f} {currency_display}"
    elif mode == "by_category":
        response_text = f"Spending by category {context_label} (Total: {total_spent:.2f} {currency_display}):\n"
        if not category_spending:
            response_text += "No categorized items found."
        else:
            for cat, amount in sorted(category_spending.items(), key=lambda item_val: item_val[1], reverse=True):
                response_text += f"- {cat}: {amount:.2f}\n"
        return response_text
    elif mode == "by_store":
        response_text = f"Spending by store {context_label} (Total: {total_spent:.2f} {currency_display}):\n"
        if not store_spending:
            response_text += "No store data found."
        else:
            for store, amount in sorted(store_spending.items(), key=lambda item_val: item_val[1], reverse=True):
                response_text += f"- {store}: {amount:.2f}\n"
        return response_text
    elif mode == "avg_receipt":
        avg = total_spent / receipt_count if receipt_count > 0 else 0
        return f"Average receipt value {context_label} ({receipt_count} receipts): {avg:.2f} {currency_display}"
    return "Invalid aggregation mode specified."


def get_current_month_start_end() -> (str, str, str):
    """Returns start_date, end_date, and YYYY-MM string for the current month."""
    now = datetime.now()
    current_month_year_str = now.strftime("%Y-%m")
    start_of_month = now.replace(day=1).strftime("%Y-%m-%d")
    if now.month == 12:
        end_of_month = now.replace(day=31).strftime("%Y-%m-%d")
    else:
        end_of_month = (now.replace(month=now.month + 1, day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
    return start_of_month, end_of_month, current_month_year_str


def get_spending_by_date_range_for_user(user_id_str: str, start_date_str: str, end_date_str: str) -> str:
    try:
        datetime.strptime(start_date_str, "%Y-%m-%d")  # Validate format
        datetime.strptime(end_date_str, "%Y-%m-%d")  # Validate format
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD for both start and end dates."
    return _aggregate_spending(user_id_str, start_date_str, end_date_str, mode="total_by_date")


def get_spending_by_category_for_user(user_id_str: str, month_year_str: str) -> str:
    try:
        year, month = map(int, month_year_str.split('-'))
        start_date = f"{year:04d}-{month:02d}-01"
        if month == 12:
            end_date = f"{year:04d}-12-31"
        else:
            end_date = (datetime(year, month + 1, 1) - timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return "Error: Invalid month format. Use YYYY-MM (e.g., 2023-10)."
    return _aggregate_spending(user_id_str, start_date, end_date, mode="by_category")


# ... (Similar wrappers for get_spending_by_store_for_user and get_average_receipt_value_for_user, calling _aggregate_spending)
def get_spending_by_store_for_user(user_id_str: str, month_year_str: str) -> str:
    try:
        year, month = map(int, month_year_str.split('-'))
        start_date = f"{year:04d}-{month:02d}-01"
        if month == 12:
            end_date = f"{year:04d}-12-31"
        else:
            end_date = (datetime(year, month + 1, 1) - timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return "Error: Invalid month format. Use YYYY-MM."
    return _aggregate_spending(user_id_str, start_date, end_date, mode="by_store")


def get_average_receipt_value_for_user(user_id_str: str, month_year_str: str) -> str:
    try:
        year, month = map(int, month_year_str.split('-'))
        start_date = f"{year:04d}-{month:02d}-01"
        if month == 12:
            end_date = f"{year:04d}-12-31"
        else:
            end_date = (datetime(year, month + 1, 1) - timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return "Error: Invalid month format. Use YYYY-MM."
    return _aggregate_spending(user_id_str, start_date, end_date, mode="avg_receipt")


# --- Telegram Bot UI ---
def build_main_menu_keyboard(is_admin: bool = False):
    keyboard_buttons = [
        [InlineKeyboardButton("📊 Current Month Summary", callback_data='summary_current_month')],
        [InlineKeyboardButton("📅 Custom Date Range Sum", callback_data='summary_date_range_prompt')],
        [InlineKeyboardButton("🏷️ Categories (This Month)", callback_data='summary_category_current_month')],
        [InlineKeyboardButton("🏪 Stores (This Month)", callback_data='summary_store_current_month')],
        [InlineKeyboardButton("🧾 Avg. Receipt (This Month)", callback_data='summary_avg_receipt_current_month')],
        [InlineKeyboardButton("👥 Group Options", callback_data='group_menu_user')],
        [InlineKeyboardButton("📄 List Recent Receipts", callback_data='list_receipts_action')],
        [InlineKeyboardButton("🗑️ Delete Recent Receipt", callback_data='delete_receipts_action')], # Adding Delete button too
    ]
    if is_admin:
        keyboard_buttons.insert(0, [InlineKeyboardButton("👑 Admin Panel",
                                                         callback_data='admin_menu')])  # Add Admin Panel button at the top for admin
    return InlineKeyboardMarkup(keyboard_buttons)


def build_user_group_menu_keyboard(user_id: str):
    buttons = [[InlineKeyboardButton("ℹ️ My Group Info", callback_data='mygroup_info_action')]]
    if get_user_group_id(user_id) and not is_user_admin(user_id):  # Only non-admins can leave via this button
        buttons.append([InlineKeyboardButton("🚪 Leave Current Group", callback_data='leavegroup_action_user')])
    buttons.append([InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='main_menu')])
    return InlineKeyboardMarkup(buttons)


def build_admin_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("List Pending Users", callback_data='admin_list_pending')],
        [InlineKeyboardButton("List Approved Users", callback_data='admin_list_approved')],
        [InlineKeyboardButton("List All Users", callback_data='admin_list_all')],
        [InlineKeyboardButton("Approve User (Type: /approveuser <ID>)", callback_data='admin_approve_prompt')],
        [InlineKeyboardButton("Ban User (Type: /banuser <ID>)", callback_data='admin_ban_prompt')],
        [InlineKeyboardButton("Create Group (Type: /admincreategroup <Name> [IDs...])",
                              callback_data='admin_creategroup_prompt')],
        [InlineKeyboardButton("Add to Group (Type: /addusertogroup <User_ID> <Group_ID>)",
                              callback_data='admin_addtogroup_prompt')],
        [InlineKeyboardButton("Remove from Group (Type: /removeuserfromgroup <User_ID> <Group_ID>)",
                              callback_data='admin_removefromgroup_prompt')],
        [InlineKeyboardButton("Delete Group (Type: /deletegroup <Group_ID>)",
                              callback_data='admin_deletegroup_prompt')],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='main_menu')],
    ]
    return InlineKeyboardMarkup(keyboard)

def format_receipt_for_display(receipt_data: dict, doc_id: str) -> str:
    """Formats receipt data into a human-readable string for Telegram."""
    if not receipt_data:
        return "Error: No receipt data to format."

    display_text = f"🧾 **Receipt Processed** (Ref: `{doc_id}`)\n"
    display_text += "------------------------------------\n"
    display_text += f"Store: {receipt_data.get('store_name', 'N/A')}\n"
    # Today's date will be used, so we display that.
    # The date field in receipt_data will be today's date.
    display_text += f"Date: {receipt_data.get('date', 'N/A')}\n"
    total_price = receipt_data.get('total_price', 0.0)
    currency = receipt_data.get('currency_code', '')
    display_text += f"Total: {total_price:.2f} {currency}\n"
    display_text += "------------------------------------\n"
    display_text += "**Items:**\n"

    items = receipt_data.get('items', [])
    if not items:
        display_text += "- No items found.\n"
    else:
        for i, item in enumerate(items):
            item_name = item.get('item_name', f'Item {i+1}')
            item_price = item.get('item_price', 0.0)
            item_qty = item.get('quantity', 1)
            item_unit = item.get('unit_of_measurement', 'unit')
            item_cat = item.get('grocery_category', 'Uncategorized')
            
            display_text += (f"- {item_name} ({item_cat})\n"
                             f"  Qty: {item_qty} {item_unit} | Price: {item_price:.2f} {currency}\n")
            if item.get('price_per_unit') is not None:
                ppu = item.get('price_per_unit', 0.0)
                display_text += f"  (PPU: {ppu:.2f} {currency}/{item_unit})\n"
            display_text += "\n" # Extra newline for readability between items

    display_text += "------------------------------------\n"
    display_text += "To correct any details, copy the JSON below, edit it, then use:\n"
    display_text += f"`/edit {doc_id}`\n"
    display_text += "followed by your corrected JSON data on a new line.\n\n"
    
    # Add the JSON data itself for easy copying
    # We should pretty-print it
    try:
        json_to_copy = json.dumps(receipt_data, indent=2, ensure_ascii=False)
        display_text += f"```json\n{json_to_copy}\n```"
    except Exception as e:
        print(f"FORMAT_RECEIPT: Error dumping JSON for display: {e}")
        display_text += "(Error preparing JSON for copying)"
        
    return display_text

def handle_edit_receipt_command(user_id_str: str, full_arg_string: str) -> str:
    """Handles the /edit <doc_id> {JSON_DATA} command."""
    if not db: return "Database not initialized."
    if not is_user_approved(user_id_str): # Also covers admin
        return "Your account is not approved for this action."

    parts = full_arg_string.split(maxsplit=1)
    if len(parts) < 2:
        return ("Usage: `/edit <RECEIPT_REF_ID>`\n"
                "Then, on a **new line** (or after a space), paste the full corrected JSON data.")

    doc_id_to_edit = parts[0].strip()
    json_string_from_user = parts[1].strip()

    if not doc_id_to_edit:
        return "Receipt Reference ID is missing."
    if not json_string_from_user:
        return "Corrected JSON data is missing."

    # Clean the JSON string (in case user pasted with ```json ... ```)
    cleaned_json_str = clean_llm_json_output(json_string_from_user)

    try:
        corrected_data_dict = json.loads(cleaned_json_str)
    except json.JSONDecodeError as je:
        return f"Error: The JSON data you provided is invalid. Please check the syntax.\nDetails: {je.msg}"

    receipt_ref = db.collection('receipts').document(doc_id_to_edit)
    receipt_doc = receipt_ref.get()

    if not receipt_doc.exists:
        return f"Error: Receipt with Reference ID `{doc_id_to_edit}` not found."

    original_receipt_data = receipt_doc.to_dict()

    # --- Authorization Check for Edit ---
    # User must be the original uploader OR in the same group OR an admin
    is_admin_editing = is_user_admin(user_id_str)
    user_is_uploader = (original_receipt_data.get('telegramUserId') == user_id_str)
    
    user_group_id = get_user_group_id(user_id_str)
    receipt_group_id = original_receipt_data.get('groupId')
    user_in_receipt_group = (receipt_group_id is not None and user_group_id == receipt_group_id)

    if not (is_admin_editing or user_is_uploader or user_in_receipt_group):
        return f"Error: You are not authorized to edit receipt `{doc_id_to_edit}`."
    # --- End Authorization Check ---

    # --- Basic Validation of corrected_data_dict (add more as needed) ---
    if not isinstance(corrected_data_dict, dict):
        return "Error: The corrected data must be a valid JSON object."
    if 'items' not in corrected_data_dict or not isinstance(corrected_data_dict['items'], list):
        corrected_data_dict['items'] = [] # Default to empty list if missing/wrong type
        print(f"EDIT_VALIDATE: 'items' field was missing or not a list in corrected data for {doc_id_to_edit}. Defaulted to [].")
    # Ensure date remains (it should be today's date, user might not change it)
    if 'date' not in corrected_data_dict:
         corrected_data_dict['date'] = original_receipt_data.get('date', datetime.now().strftime("%Y-%m-%d"))
    # You might want to re-validate numeric fields, categories, etc.
    # For simplicity, we're trusting the user's structure for now if it's valid JSON.
    # --- End Basic Validation ---

    # Preserve original upload timestamp and uploader, update the rest
    update_payload = corrected_data_dict.copy() # Start with user's full data
    update_payload['telegramUserId'] = original_receipt_data.get('telegramUserId') # Keep original uploader
    update_payload['uploadTimestamp'] = original_receipt_data.get('uploadTimestamp') # Keep original upload time
    if 'groupId' in original_receipt_data and original_receipt_data['groupId'] is not None: # Preserve original group
        update_payload['groupId'] = original_receipt_data['groupId']
    else:
        update_payload.pop('groupId', None) # Ensure no group if original had none

    update_payload['lastUpdatedAt'] = firestore.SERVER_TIMESTAMP
    update_payload['isVerifiedByUser'] = True # Mark as edited/verified
    update_payload['editedBy'] = user_id_str # Track who edited

    try:
        receipt_ref.set(update_payload) # Overwrite with the new validated data
        return f"✅ Receipt `{doc_id_to_edit}` updated successfully!"
    except Exception as e_update:
        print(f"EDIT_RECEIPT: Firestore update error for {doc_id_to_edit}: {e_update}")
        return f"Error: Could not update receipt `{doc_id_to_edit}` in the database."

def format_single_receipt_for_view(receipt_data: dict) -> str:
    """Formats a single receipt for detailed viewing."""
    if not receipt_data: return "Error: Receipt data not found."
    
    doc_id = receipt_data.get('id', 'N/A')
    display_text = f"🧾 **Receipt Details** (Ref: `{doc_id}`)\n"
    display_text += "------------------------------------\n"
    display_text += f"Store: {receipt_data.get('store_name', 'N/A')}\n"
    display_text += f"Date: {receipt_data.get('date', 'N/A')}\n"
    total_price = receipt_data.get('total_price', 0.0)
    currency = receipt_data.get('currency_code', '')
    display_text += f"Total: {total_price:.2f} {currency}\n"
    display_text += f"Uploaded By: `{receipt_data.get('telegramUserId', 'N/A')}`\n"
    if receipt_data.get('groupId'):
         display_text += f"Group ID: `{receipt_data.get('groupId')}`\n"
    display_text += f"Verified: {'Yes' if receipt_data.get('isVerifiedByUser') else 'No'}"
    if receipt_data.get('editedBy'):
        display_text += f" (Edited by: `{receipt_data.get('editedBy')}`)\n"
    else:
        display_text += "\n"
    display_text += "------------------------------------\n"
    display_text += "**Items:**\n"

    currency = receipt_data.get('currency_code', '') 

    display_text += "**Items:**\n"
    items = receipt_data.get('items', [])
    if not items:
        display_text += "- No items found.\n"
    else:
        for i, item in enumerate(items):
            item_name = item.get('item_name', f'Item {i+1}')
            item_price = item.get('item_price') # Get value, could be None
            item_qty = item.get('quantity', 1) 
            item_unit = item.get('unit_of_measurement', 'unit')
            item_cat = item.get('grocery_category', 'Uncategorized')
            
            # --- FIX: Check if item_price is None before formatting ---
            item_price_display = f"{item_price:.2f}" if item_price is not None else "N/A"
            # --- END FIX ---
            
            # Construct the item lines using the display-ready price
            display_text += (f"- {escape_markdown_v2(item_name)} ({escape_markdown_v2(item_cat)})\n" # Escape dynamic content
                             f"  Qty: {item_qty} {escape_markdown_v2(item_unit)} | Price: {item_price_display} {currency}\n") 
            
            # Check price_per_unit exists AND is not None before formatting
            price_per_unit_val = item.get('price_per_unit') 
            if price_per_unit_val is not None:
                # --- FIX: Format ppu safely ---
                ppu_display = f"{price_per_unit_val:.2f}" 
                # --- END FIX ---
                display_text += f"  (PPU: {ppu_display} {currency}/{escape_markdown_v2(item_unit)})\n" 
            display_text += "\n" # Extra newline for readability
            
    display_text += "------------------------------------\n"
    
    return display_text

# Define your allowed categories for validation during edit
ALLOWED_GROCERY_CATEGORIES = [
    "Produce", "Dairy & Eggs", "Meat & Seafood", "Bakery", "Pantry Staples", 
    "Frozen Foods", "Beverages (Non-alcoholic)", "Alcohol", "Snacks & Sweets", 
    "Household Supplies", "Personal Care", "Baby Items", "Pet Supplies", "Other"
]

def parse_edited_receipt_text(text_data: str) -> (str, dict, list, list):
    """
    Parses the multi-line text input for editing a receipt.
    Returns (doc_id, header_data, item_lines, errors_list)
    """
    lines = text_data.strip().split('\n')
    doc_id = None
    header_data = {}
    item_lines_raw = []
    parse_errors = []

    if not lines:
        parse_errors.append("No data provided for editing.")
        return None, {}, [], parse_errors

    # Parse headers first
    header_keys_map = {
        "ref": "doc_id",
        "store": "store_name",
        "date": "date",
        "total": "total_price",
        "currency": "currency_code" # Optional currency
    }
    
    header_lines_count = 0
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if not line_lower: continue # Skip empty lines

        is_item_line = ';' in line # Heuristic: item lines contain semicolons

        if not is_item_line and ':' in line:
            key_raw, value_raw = line.split(':', 1)
            key_clean = key_raw.strip().lower()
            value_clean = value_raw.strip()

            if key_clean in header_keys_map:
                field_name = header_keys_map[key_clean]
                if field_name == "doc_id":
                    doc_id = value_clean
                elif field_name == "total_price":
                    try:
                        header_data[field_name] = float(value_clean)
                    except ValueError:
                        parse_errors.append(f"Invalid format for Total Price: '{value_clean}'. Must be a number.")
                elif field_name == "date":
                    try:
                        datetime.strptime(value_clean, "%Y-%m-%d") # Validate format
                        header_data[field_name] = value_clean
                    except ValueError:
                        parse_errors.append(f"Invalid format for Date: '{value_clean}'. Must be YYYY-MM-DD.")
                else: # store_name, currency_code
                    header_data[field_name] = value_clean
                header_lines_count = i + 1 # Mark how many lines were headers
            elif i < 5: # Assume first few lines without ';' could be headers even if key not recognized
                 parse_errors.append(f"Unrecognized header line (or misplaced item): '{line}'")

        elif is_item_line:
            # All subsequent lines should be item lines
            item_lines_raw = lines[i:]
            break # Stop header parsing, rest are items
        elif i >= header_lines_count and line.strip(): # Non-empty line after headers that's not an item line
            parse_errors.append(f"Unexpected line after headers (expected item or empty line): '{line}'")


    if not doc_id:
        parse_errors.append("`Ref: <ID>` line is missing or improperly formatted.")
        
    return doc_id, header_data, item_lines_raw, parse_errors

MAX_RECEIPTS_PER_LIST = 20 # Define how many receipts to show per "page"

def get_recent_receipts_for_user(user_id_str: str, limit: int = MAX_RECEIPTS_PER_LIST) -> (list, str):
    """Fetches recent receipts for a user or their group."""
    if not db: return [], "Database not initialized."
    
    query_field, query_value, context_label = get_query_target(user_id_str) # Use existing helper

    receipts_ref = db.collection('receipts')
    
    # Order by date descending, then by upload timestamp descending as a secondary sort
    # Note: This complex ordering might require a composite index!
    # Firestore may complain and give you a link to create:
    # Index on: <query_field> (Asc), date (Desc), uploadTimestamp (Desc)
    query = receipts_ref.where(query_field, '==', query_value)\
                        .order_by('date', direction=firestore.Query.DESCENDING)\
                        .order_by('uploadTimestamp', direction=firestore.Query.DESCENDING)\
                        .limit(limit)
                        
    receipt_list = []
    try:
        docs = query.stream()
        for doc in docs:
            receipt_data = doc.to_dict()
            receipt_data['id'] = doc.id # Add the document ID to the data dict
            receipt_list.append(receipt_data)
        return receipt_list, context_label
    except Exception as e:
        print(f"LIST_RECEIPTS: Error querying Firestore: {e}")
        # Check if it's a missing index error
        if "index" in str(e).lower():
             return [], "Error: A database index needed for sorting recent receipts is missing. The administrator may need to create it via a link in the error logs."
        return [], f"An error occurred while fetching receipts: {type(e).__name__}"

def format_receipt_list_for_display(receipt_list: list, context_label: str) -> (str, InlineKeyboardMarkup):
    """Formats the list of receipts into a message with inline buttons."""
    if not receipt_list:
        return f"No recent receipts found {context_label}.", None

    message_text = f"📄 **Recent Receipts** {context_label} (max {MAX_RECEIPTS_PER_LIST}):\n"
    message_text += "------------------------------------\n"
    
    keyboard_buttons = []
    for receipt in receipt_list:
        doc_id = receipt.get('id')
        date_str = receipt.get('date', 'N/A')
        store_name = receipt.get('store_name', 'Unknown Store')
        total_price = receipt.get('total_price', 0.0)
        currency = receipt.get('currency_code', '')
        
        # Format button text concisely
        button_text = f"{date_str} | {store_name[:20]} | {total_price:.2f} {currency}"
        # Shorten text if too long for button constraints
        if len(button_text) > 60: # Telegram button data limits text somewhat
             button_text = button_text[:57] + "..."

        # Callback data includes action prefix and the doc ID
        callback_data = f"view_receipt_{doc_id}" 
        
        keyboard_buttons.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    if not keyboard_buttons: # Should not happen if receipt_list is not empty, but safety check
        return f"Found receipts {context_label}, but couldn't create buttons.", None
        
    message_text += "Click on a receipt to view details."
    reply_markup = InlineKeyboardMarkup(keyboard_buttons)
    
    return message_text, reply_markup

def get_receipt_details_for_view(doc_id: str) -> dict or None:
    """Fetches a single receipt document by ID."""
    if not db or not doc_id: return None
    doc_ref = db.collection('receipts').document(doc_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        data['id'] = doc.id # Ensure ID is included
        return data
    return None

def handle_simple_edit_receipt_command(user_id_str: str, full_edit_text: str) -> str:
    """Handles the /edit command with simpler text format."""
    if not db: return "Database not initialized."
    if not is_user_approved(user_id_str):
        return "Your account is not approved for this action."

    # The first line might be "/edit", subsequent lines are the data.
    # We expect full_edit_text to be the data *after* the "/edit" command word.
    
    doc_id_to_edit, parsed_header_data, item_lines_raw, parse_errors = parse_edited_receipt_text(full_edit_text)

    if not doc_id_to_edit: # Critical error from parsing
        parse_errors.insert(0, "Could not identify the Receipt Reference ID.")
        return "Error parsing your edit request:\n" + "\n".join(parse_errors) + "\nPlease use the specified format."
    
    if parse_errors and not item_lines_raw: # If only header errors and no items, probably bad format
        return "Error parsing receipt headers:\n" + "\n".join(parse_errors) + "\nPlease check the format for Store, Date, Total, and Ref ID."


    receipt_ref = db.collection('receipts').document(doc_id_to_edit)
    receipt_doc = receipt_ref.get()

    if not receipt_doc.exists:
        return f"Error: Receipt with Reference ID `{doc_id_to_edit}` not found."

    original_receipt_data = receipt_doc.to_dict()

    # --- Authorization Check (same as before) ---
    is_admin_editing = is_user_admin(user_id_str)
    user_is_uploader = (original_receipt_data.get('telegramUserId') == user_id_str)
    user_group_id = get_user_group_id(user_id_str)
    receipt_group_id = original_receipt_data.get('groupId')
    user_in_receipt_group = (receipt_group_id is not None and user_group_id == receipt_group_id)

    if not (is_admin_editing or user_is_uploader or user_in_receipt_group):
        return f"Error: You are not authorized to edit receipt `{doc_id_to_edit}`."
    
    # --- Construct New Receipt Data from Parsed Input ---
    corrected_data = {}
    # Start with original, then update with parsed header data
    corrected_data['store_name'] = parsed_header_data.get('store_name', original_receipt_data.get('store_name'))
    # Date: Use parsed if valid, else original, else today (but original should always have one)
    corrected_data['date'] = parsed_header_data.get('date', original_receipt_data.get('date', datetime.now().strftime("%Y-%m-%d")))
    corrected_data['total_price'] = parsed_header_data.get('total_price', original_receipt_data.get('total_price'))
    corrected_data['currency_code'] = parsed_header_data.get('currency_code', original_receipt_data.get('currency_code'))

    # Process items
    parsed_items = []
    item_parse_errors = []
    for idx, item_line in enumerate(item_lines_raw):
        item_line = item_line.strip()
        if not item_line: continue # Skip empty lines

        parts = [p.strip() for p in item_line.split(';')]
        if len(parts) < 3: # Name, Price, Category are mandatory
            item_parse_errors.append(f"Item line {idx+1} ('{item_line[:30]}...') has too few parts. Expected: Name; Price; Category; [Quantity; Unit]")
            continue
        
        item_name = parts[0]
        item_price_str = parts[1]
        item_category = parts[2]
        item_quantity_str = parts[3] if len(parts) > 3 else "1" # Default quantity 1
        item_unit = parts[4] if len(parts) > 4 else "unit"    # Default unit "unit"

        item_data = {'item_name': item_name}
        try:
            item_data['item_price'] = float(item_price_str)
        except ValueError:
            item_parse_errors.append(f"Item '{item_name}': Invalid price '{item_price_str}'. Must be a number.")
            continue # Skip this item if price is invalid
        
        if item_category not in ALLOWED_GROCERY_CATEGORIES and item_category != "Other":
             # Optionally, be strict or just log a warning and accept it
            print(f"EDIT_ITEM_WARN: Category '{item_category}' for item '{item_name}' not in predefined list. Accepting anyway.")
            # item_parse_errors.append(f"Item '{item_name}': Category '{item_category}' is not a recognized category. Use one of: {', '.join(ALLOWED_GROCERY_CATEGORIES)}, or 'Other'.")
        item_data['grocery_category'] = item_category

        try:
            item_data['quantity'] = float(item_quantity_str) if '.' in item_quantity_str else int(item_quantity_str)
        except ValueError:
            item_parse_errors.append(f"Item '{item_name}': Invalid quantity '{item_quantity_str}'. Defaulting to 1.")
            item_data['quantity'] = 1
        
        item_data['unit_of_measurement'] = item_unit
        
        # Calculate price_per_unit if possible
        if item_data['quantity'] > 0 and item_data['item_price'] is not None:
            item_data['price_per_unit'] = round(item_data['item_price'] / item_data['quantity'], 2)
        else:
            item_data['price_per_unit'] = item_data['item_price'] # Or null if qty is 0/invalid

        parsed_items.append(item_data)

    # Combine all parsing errors
    all_errors = parse_errors + item_parse_errors
    if all_errors:
        error_summary = "Found errors in your edit input:\n" + "\n".join(all_errors)
        error_summary += "\n\nPlease correct these and try `/edit` again."
        return error_summary

    corrected_data['items'] = parsed_items

    # --- Preserve original metadata and update ---
    update_payload = corrected_data.copy()
    update_payload['telegramUserId'] = original_receipt_data.get('telegramUserId')
    update_payload['uploadTimestamp'] = original_receipt_data.get('uploadTimestamp')
    if 'groupId' in original_receipt_data and original_receipt_data['groupId'] is not None:
        update_payload['groupId'] = original_receipt_data['groupId']
    else:
        update_payload.pop('groupId', None)
    
    update_payload['lastUpdatedAt'] = firestore.SERVER_TIMESTAMP
    update_payload['isVerifiedByUser'] = True
    update_payload['editedBy'] = user_id_str

    try:
        receipt_ref.set(update_payload) # Overwrite with the new validated data
        return f"✅ Receipt `{doc_id_to_edit}` updated successfully with your changes!"
    except Exception as e_update:
        print(f"EDIT_RECEIPT_SIMPLE: Firestore update error for {doc_id_to_edit}: {e_update}")
        return f"Error: Could not update receipt `{doc_id_to_edit}` in the database."

def format_receipt_list_for_delete(receipt_list: list, context_label: str) -> (str, InlineKeyboardMarkup):
    """Formats the list of receipts with 'Request Delete' buttons."""
    if not receipt_list:
        return f"No recent receipts found {context_label} to delete.", None

    message_text = f"🗑️ **Select Receipt to Delete** {context_label} (max {MAX_RECEIPTS_PER_LIST}):\n"
    message_text += "------------------------------------\n"
    
    keyboard_buttons = []
    for receipt in receipt_list:
        doc_id = receipt.get('id')
        if not doc_id: continue # Skip if somehow ID is missing

        date_str = receipt.get('date', 'N/A')
        store_name = receipt.get('store_name', 'Unknown Store')
        total_price = receipt.get('total_price', 0.0)
        currency = receipt.get('currency_code', '')
        
        button_text = f"{date_str} | {store_name[:20]} | {total_price:.2f} {currency}"
        if len(button_text) > 60: button_text = button_text[:57] + "..."

        # Callback data for initiating delete confirmation
        callback_data = f"del_confirm_req_{doc_id}" 
        
        keyboard_buttons.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    if not keyboard_buttons:
        return f"Found receipts {context_label}, but couldn't create delete buttons.", None
        
    message_text += "Click on a receipt to request its deletion."
    reply_markup = InlineKeyboardMarkup(keyboard_buttons)
    
    return message_text, reply_markup

def build_delete_confirmation_keyboard(doc_id: str) -> InlineKeyboardMarkup:
    """Builds the Yes/Cancel keyboard for delete confirmation."""
    keyboard = [[
        InlineKeyboardButton("✅ Yes, Delete", callback_data=f"del_do_{doc_id}"),
        InlineKeyboardButton("❌ Cancel", callback_data=f"del_cancel_{doc_id}") # Or just 'main_menu' maybe
    ]]
    return InlineKeyboardMarkup(keyboard)

def delete_receipt_from_firestore(acting_user_id_str: str, doc_id_to_delete: str) -> str:
    """Performs authorization check and deletes the receipt."""
    if not db: return "Database not initialized."

    receipt_ref = db.collection('receipts').document(doc_id_to_delete)
    receipt_doc = receipt_ref.get()

    if not receipt_doc.exists:
        return f"Error: Receipt Ref: `{doc_id_to_delete}` no longer exists."

    original_receipt_data = receipt_doc.to_dict()

    # --- Authorization Check for Deletion ---
    is_admin_deleting = is_user_admin(acting_user_id_str)
    user_is_uploader = (original_receipt_data.get('telegramUserId') == acting_user_id_str)

    if not (is_admin_deleting or user_is_uploader):
        return f"Error: You are not authorized to delete receipt Ref: `{doc_id_to_delete}`. Only the uploader or admin can delete."
    # --- End Authorization Check ---

    try:
        receipt_ref.delete()
        print(f"RECEIPT_DELETE: User {acting_user_id_str} deleted receipt {doc_id_to_delete}.")
        # Optionally log who deleted it to an audit log collection
        return f"✅ Receipt Ref: `{doc_id_to_delete}` has been deleted."
    except Exception as e_delete:
        print(f"RECEIPT_DELETE: Firestore delete error for {doc_id_to_delete}: {e_delete}")
        return f"Error: Could not delete receipt `{doc_id_to_delete}` from the database."

def escape_markdown_v2(text: str) -> str:
    """
    Escapes characters reserved in Telegram MarkdownV2 parse mode.

    See: https://core.telegram.org/bots/api#markdownv2-style
    
    Args:
        text: The input string potentially containing reserved characters.

    Returns:
        The escaped string, safe for use in MarkdownV2 messages.
    """
    if not isinstance(text, str): # Handle non-string input gracefully
        return ""
        
    # Characters to escape as defined by Telegram Bot API documentation
    # Note: ` is handled by code blocks, ~ by strikethrough, etc.
    # We only need to escape characters that should appear literally in the text.
    escape_chars = r'_*[]()~`>#+-=|{}.!' 
    
    # Use re.sub to find any character in the escape_chars set and prepend it with a backslash '\'
    # We need re.escape() around escape_chars itself to handle potential regex metacharacters within that string (like -, [, ], etc.)
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- Main Webhook Handler using @https_fn ---
@https_fn.on_request(max_instances=10)
def telegram_webhook(req: https_fn.Request) -> https_fn.Response:
    initialize_services()  # Ensure db, bot, ADMIN_USER_ID are ready

    # Critical check after initialization attempt
    if bot is None:
        print("WEBHOOK CRITICAL: Bot not initialized. Exiting.")
        return https_fn.Response("Internal Server Error: Bot configuration issue", status=500)
    # ADMIN_USER_ID being None is a warning but function can proceed for user ID requests

    # Get Telegram update JSON
    update_json = None
    if req.method == "POST":
        try:
            update_json = req.json
            if not isinstance(update_json, dict):  # req.json might return non-dict if body is malformed
                print(f"WEBHOOK: Invalid JSON payload type: {type(update_json)}. Body: {req.data[:200]}")
                return https_fn.Response("Bad Request: Invalid JSON payload", status=400)
        except Exception as e:  # Broader catch for any parsing issue with req.json
            print(f"WEBHOOK: Error parsing JSON from request: {e}. Body: {req.data[:200]}")
            return https_fn.Response("Bad Request: Could not parse JSON", status=400)
    else:  # If not POST
        print(f"WEBHOOK: Received {req.method} request, expected POST.")
        return https_fn.Response("Method Not Allowed", status=405)

    if not update_json:  # Should be caught above, but as a safeguard
        print("WEBHOOK: No update_json after POST check.")
        return https_fn.Response("Bad Request: Empty update", status=400)

    update = telegram.Update.de_json(update_json, bot)

    chat_id_int = None
    user_id_str = None
    text_payload = None  # Can be command from text or callback_data

    if update.message:
        chat_id_int = update.message.chat_id
        user_id_str = str(update.message.from_user.id)
        text_payload = update.message.text
    elif update.callback_query:
        chat_id_int = update.callback_query.message.chat.id
        user_id_str = str(update.callback_query.from_user.id)
        text_payload = update.callback_query.data
        try:
            bot.answer_callback_query(callback_query_id=update.callback_query.id)
        except Exception as e_ack:
            print(f"WEBHOOK: Error answering callback query {update.callback_query.id}: {e_ack}")

    if not chat_id_int or not user_id_str:
        print("WEBHOOK ERROR: Could not determine chat_id or user_id from update.")
        return https_fn.Response("Bad Request: Invalid Telegram update structure", status=400)

    # Ensure user profile exists. This also handles sending the "pending approval" message.
    user_profile = ensure_user_profile_exists(user_id_str, chat_id_for_message=chat_id_int)
    current_user_status = user_profile.get('status', 'pending_approval')  # Default to pending if somehow missing
    is_current_user_admin = is_user_admin(user_id_str)

    # Parse command and args (for typed commands)
    command_action = ""  # This will be the core action/command identifier
    command_args_list = []  # For /cmd arg1 arg2
    full_arg_string = ""  # For /cmd <name with spaces>

    if text_payload:
        if text_payload.startswith('/'):  # Actual command from text
            parts = text_payload.split(maxsplit=1)
            command_action = parts[0].lower()
            if len(parts) > 1:
                full_arg_string = parts[1]
                command_args_list = full_arg_string.split()  # Simple split
        elif update.callback_query:  # Callback data acts as command_action
            command_action = text_payload
            # Args from callback_data usually need specific parsing if complex, e.g. "action:value"

    # --- Centralized Message Sending Function ---
    def send_tg_message(text: str, reply_markup=None, parse_mode=None):
        if not bot or not chat_id_int: return
        try:
            bot.send_message(chat_id=chat_id_int, text=text, reply_markup=reply_markup, parse_mode=parse_mode)
        except telegram.error.TelegramError as te:
            print(f"SEND_MSG_ERROR: Telegram API error sending to {chat_id_int}: {te}")
        except Exception as e:
            print(f"SEND_MSG_ERROR: Generic error sending to {chat_id_int}: {e}")

    # --- Admin Command Handling (Bypasses regular user status checks for admin actions) ---
    if is_current_user_admin:
        admin_action_handled = True  # Flag to see if an admin command was processed
        if command_action == '/adminhelp':
            help_text = ("Admin Commands:\n"
                         "/listusers [pending|approved|banned|all]\n"
                         "/approveuser <user_id>\n"
                         "/banuser <user_id>\n"
                         "/setuserstatus <user_id> <status>\n"
                         "/admincreategroup <group_name> [user_id1 user_id2 ...]\n"
                         "/addusertogroup <user_id> <group_id>\n"
                         "/removeuserfromgroup <user_id> <group_id>\n"
                         "/deletegroup <group_id>")
            send_tg_message(help_text)
        elif command_action == '/listusers':
            send_tg_message(admin_list_users(command_args_list[0] if command_args_list else "all"),
                            parse_mode='Markdown')
        elif command_action == '/approveuser':
            send_tg_message(admin_set_user_status(command_args_list[0],
                                                  "approved") if command_args_list else "Usage: /approveuser <user_id>",
                            parse_mode='Markdown')
        elif command_action == '/banuser':
            send_tg_message(admin_set_user_status(command_args_list[0],
                                                  "banned") if command_args_list else "Usage: /banuser <user_id>",
                            parse_mode='Markdown')
        elif command_action == '/setuserstatus':
            send_tg_message(admin_set_user_status(command_args_list[0], command_args_list[1]) if len(
                command_args_list) == 2 else "Usage: /setuserstatus <user_id> <approved|banned|pending_approval>",
                            parse_mode='Markdown')
        elif command_action == '/admincreategroup':
            # Need to parse group name and optional user IDs
            group_name_adm = ""
            initial_m_adm = []
            if full_arg_string:
                 # Try to parse quoted name first
                 quoted_match = re.match(r'["\'](.*?)["\']\s*(.*)', full_arg_string)
                 if quoted_match:
                     group_name_adm = quoted_match.group(1)
                     remaining_args = quoted_match.group(2).split()
                     initial_m_adm = [arg for arg in remaining_args if arg] # Filter empty strings
                 else: # Assume first word is name, rest are IDs
                     parts = full_arg_string.split(maxsplit=1)
                     group_name_adm = parts[0]
                     if len(parts) > 1:
                         initial_m_adm = [arg for arg in parts[1].split() if arg]

            if not group_name_adm:
                 send_tg_message("Usage: /admincreategroup \"Group Name\" [UserID1 UserID2 ...]")
            else:
                 _, msg_adm_create = create_group_in_firestore(user_id_str, group_name_adm, initial_member_ids=initial_m_adm)
                 send_tg_message(msg_adm_create, parse_mode='MarkdownV2') # Use V2 for backticks
            return https_fn.Response("OK", status=200)
            _, msg_adm_create = create_group_in_firestore(user_id_str, group_name_adm, initial_member_ids=initial_m_adm)
            send_tg_message(msg_adm_create, parse_mode='Markdown')
        elif command_action == '/addusertogroup':
            send_tg_message(admin_add_user_to_group(user_id_str, command_args_list[0], command_args_list[1]) if len(
                command_args_list) == 2 else "Usage: /addusertogroup <user_id> <group_id>", parse_mode='Markdown')
        elif command_action == '/removeuserfromgroup':
            send_tg_message(admin_remove_user_from_group(command_args_list[0], command_args_list[1]) if len(
                command_args_list) == 2 else "Usage: /removeuserfromgroup <user_id> <group_id>", parse_mode='Markdown')
        elif command_action == '/deletegroup':
            send_tg_message(
                admin_delete_group(command_args_list[0]) if command_args_list else "Usage: /deletegroup <group_id>",
                parse_mode='Markdown')
        # Admin menu buttons
        elif command_action == 'admin_menu':
            send_tg_message("Admin Panel:", reply_markup=build_admin_menu_keyboard())
        elif command_action == 'admin_list_pending':
            send_tg_message(admin_list_users("pending_approval"), parse_mode='Markdown')
        elif command_action == 'admin_list_approved':
            send_tg_message(admin_list_users("approved"), parse_mode='Markdown')
        elif command_action == 'admin_list_all':
            send_tg_message(admin_list_users("all"), parse_mode='Markdown')
        elif command_action == 'admin_approve_prompt':
            send_tg_message("Type: `/approveuser USER_ID_TO_APPROVE`")
        elif command_action == 'admin_ban_prompt':
            send_tg_message("Type: `/banuser USER_ID_TO_BAN`")
        elif command_action == 'admin_creategroup_prompt':
            send_tg_message("Type: `/admincreategroup \"Group Name\" OptionalUserID1 OptionalUserID2 ...`")
        elif command_action == 'admin_addtogroup_prompt':
            send_tg_message("Type: `/addusertogroup USER_ID_TO_ADD GROUP_ID`")
        elif command_action == 'admin_removefromgroup_prompt':
            send_tg_message("Type: `/removeuserfromgroup USER_ID_TO_REMOVE GROUP_ID`")
        elif command_action == 'admin_deletegroup_prompt':
            send_tg_message("Type: `/deletegroup GROUP_ID_TO_DELETE`")
        else:
            admin_action_handled = False  # Not an admin-specific command

        if admin_action_handled and not (
                command_action == '/start' or command_action == '/menu' or command_action == 'main_menu'):  # Avoid double processing if admin also typed /start
            return https_fn.Response("OK", status=200)  # Admin action processed

    # --- Access Control Block for Non-Admin Users (and for admin using user features) ---
    if command_action == '/start':  # Handles /start for all users after admin check
        if current_user_status == 'pending_approval' and not is_current_user_admin:
            # ensure_user_profile_exists already sent the message
            pass  # Message already sent by ensure_user_profile_exists
        elif current_user_status == 'banned' and not is_current_user_admin:
            send_tg_message("Your account access has been restricted.")
        else:  # Approved user or Admin typing /start
            command_action = 'main_menu'  # Fall through to show main menu
        # For /start, we typically don't want to fall through to other command processing immediately
        if not (command_action == 'main_menu'):  # if it wasn't re-routed to main_menu
            return https_fn.Response("OK", status=200)

    # For any other action by non-admin, check status (Admin bypasses this for their own actions)
    if not is_current_user_admin:
        if current_user_status == 'pending_approval':
            # Message should have been sent by ensure_user_profile_exists, or by /start handler above.
            # To be safe, can resend if command_action is not empty and not /start
            if command_action and command_action != '/start':
                send_tg_message(
                    f"Your account is pending approval. Your User ID is `{user_id_str}`. Please wait or contact the administrator.",
                    parse_mode='Markdown')
            return https_fn.Response("OK", status=200)
        elif current_user_status == 'banned':
            if command_action and command_action != '/start':  # Avoid double message if /start already handled it
                send_tg_message("Your account access has been restricted.")
            return https_fn.Response("OK", status=200)

    # --- Image Processing (only for approved users or admin) ---
    image_to_process_info = None  # Will store (file_id, file_name, mime_type_guess)

    if update.message and update.message.photo:
        # Largest photo is usually best quality among compressed versions
        photo_file_id = update.message.photo[-1].file_id
        # Telegram photos are typically JPEGs after compression
        image_to_process_info = (photo_file_id, "telegram_photo.jpg", "image/jpeg")
        print(f"WEBHOOK: Detected photo message. File ID: {photo_file_id}")

    elif update.message and update.message.document:
        doc = update.message.document
        print(
            f"WEBHOOK: Detected document. MIME Type: {doc.mime_type}, File Name: {doc.file_name}, File ID: {doc.file_id}")
        # Check if the document is an image type we want to process
        if doc.mime_type and doc.mime_type.startswith('image/'):  # e.g., image/jpeg, image/png
            # Use the document's reported MIME type if available, otherwise guess from extension
            file_mime_type = doc.mime_type
            image_to_process_info = (doc.file_id, doc.file_name, file_mime_type)
        else:
            send_tg_message(
                "You sent a file, but it doesn't appear to be an image. Please send a photo or an image file (JPEG, PNG).")

    if image_to_process_info:

        file_id_for_download, original_filename, detected_mime_type = image_to_process_info  # Unpack

        send_tg_message("Got your image! Analyzing with Gemini Vision...")

        try:
            photo_file = bot.get_file(file_id_for_download)
            image_response = requests.get(photo_file.file_path)
            image_response.raise_for_status()
            image_bytes = image_response.content

            # Use the detected_mime_type, or default if somehow still None
            mime_type_for_llm = detected_mime_type if detected_mime_type else "image/jpeg"
            print(f"IMGPROC: Processing image '{original_filename}' with MIME type '{mime_type_for_llm}'.")

            receipt_data_from_llm = call_llm_for_receipt(image_bytes, image_mime_type=mime_type_for_llm)

            # --- FEATURE: Use Today's Date ---
            todays_date_str = datetime.now().strftime("%Y-%m-%d")
            receipt_data_to_save = receipt_data_from_llm.copy() # Work with a copy
            original_llm_date = receipt_data_to_save.pop('date', None) # Remove LLM's date attempt
            receipt_data_to_save['date'] = todays_date_str # Set to today's date
            print(f"IMGPROC: Using today's date '{todays_date_str}'. LLM proposed: '{original_llm_date}'")
            # --- END FEATURE ---

            # Determine context for saving (personal or group)
            # Admin's receipts are personal unless they explicitly manage group settings (not implemented for receipt adding)
            # For simplicity, admin's own receipts are personal here.
            current_group_id = get_user_group_id(user_id_str)

            receipt_data_to_save['telegramUserId'] = user_id_str   # Always tag with individual user
            if current_group_id:
                receipt_data_to_save['groupId'] = current_group_id
            else:
                receipt_data_to_save.pop('groupId', None)

            receipt_data_to_save['uploadTimestamp'] = firestore.SERVER_TIMESTAMP
            receipt_data_to_save['lastUpdatedAt'] = firestore.SERVER_TIMESTAMP # Add a field for edits
            receipt_data_to_save['isVerifiedByUser'] = False # New field

            _, actual_doc_ref = db.collection('receipts').add(receipt_data_to_save)
            doc_id = actual_doc_ref.id # Get the ID of the newly created document
            print(f"IMGPROC: Receipt initial save. Doc ID: {doc_id}")

            # --- FEATURE: Show Formatted Result ---
            # Prepare the data that was actually saved (with today's date) for display
            formatted_message_to_user = format_single_receipt_for_view(receipt_data_to_save)
            
            # Telegram messages have a length limit (4096 chars).
            # If formatted_message_to_user is too long, we might need to split it or send a summary.
            if len(formatted_message_to_user) > 4000: # Leave some buffer
                summary_text = (f"🧾 Receipt Processed & Saved (Ref: `{doc_id}`)\n"
                                f"Store: {receipt_data_to_save.get('store_name', 'N/A')}\n"
                                f"Date: {receipt_data_to_save.get('date', 'N/A')}\n"
                                f"Total: {receipt_data_to_save.get('total_price', 0.0):.2f} {receipt_data_to_save.get('currency_code', '')}\n\n"
                                "The full details were too long to display here. "
                                "You can still edit it using the Ref ID and the JSON structure if needed. "
                                "To edit, see instructions after sending `/edithelp` or refer to the standard edit format.")
                send_tg_message(summary_text, parse_mode='Markdown')
            else:
                send_tg_message(formatted_message_to_user, parse_mode='Markdown')
            # --- END FEATURE ---

        except ValueError as ve:  # Catch specific error from call_llm_for_receipt or other validation
            send_tg_message(f"Receipt Processing Error: {str(ve)}")
        except telegram.error.TelegramError as te:
            print(f"WEBHOOK IMGPROC: Telegram error during image processing for {user_id_str}: {te}")
            send_tg_message("A Telegram error occurred while processing your receipt.")  # Generic to user
        except Exception as e_proc:
            print(f"WEBHOOK IMGPROC: Unexpected error for {user_id_str}: {e_proc}")
            import traceback
            traceback.print_exc()
            send_tg_message("An unexpected error occurred while processing the receipt image.")
        return https_fn.Response("OK", status=200)  # Image processing done

    # --- Regular User Menu and Command Handling (Approved users / Admin using user features) ---
    action_handled_for_user = True  # Assume action will be handled unless it falls to the end
    start_of_month, end_of_month, current_month_year_str = get_current_month_start_end()

    # --- NEW: Delete Receipts Command ---
    if command_action == '/deletereceipts':
         if not is_user_approved(user_id_str): # Check approval
             send_tg_message("Your account is not approved for this action.")
         else:
             receipts, context = get_recent_receipts_for_user(user_id_str)
             message_text, reply_markup = format_receipt_list_for_delete(receipts, context)
             send_tg_message(message_text, reply_markup=reply_markup, parse_mode='Markdown') # Use Markdown for list

    # --- NEW: Delete Confirmation Request Callback ---
    elif command_action and command_action.startswith('del_confirm_req_'):
         doc_id_to_confirm = command_action.replace('del_confirm_req_', '')
         if not doc_id_to_confirm:
             send_tg_message("Error: Invalid receipt reference in button.")
         else:
            receipt_to_confirm = get_receipt_details_for_view(doc_id_to_confirm)
            if not receipt_to_confirm:
                 send_tg_message(f"Error: Receipt Ref: `{doc_id_to_confirm}` not found (maybe already deleted?).", parse_mode='Markdown')
            else:
                # Authorization check (Uploader or Admin)
                is_admin_req = is_user_admin(user_id_str)
                user_is_uploader_req = (receipt_to_confirm.get('telegramUserId') == user_id_str)
                if not (is_admin_req or user_is_uploader_req):
                     send_tg_message(f"Error: You are not authorized to delete receipt Ref: `{doc_id_to_confirm}`.", parse_mode='Markdown')
                else:
                    # Send confirmation message with Yes/Cancel buttons
                    confirm_text = (f"❓ **Confirm Deletion**\n\n"
                                    f"Are you sure you want to permanently delete receipt:\n"
                                    f"Ref: `{doc_id_to_confirm}`\n"
                                    f"Store: {escape_markdown_v2(receipt_to_confirm.get('store_name', 'N/A'))}\n"
                                    f"Date: {receipt_to_confirm.get('date', 'N/A')}\n"
                                    f"Total: {receipt_to_confirm.get('total_price', 0.0):.2f} {receipt_to_confirm.get('currency_code', '')}\n\n"
                                    f"⚠️ This action cannot be undone!")
                    confirm_keyboard = build_delete_confirmation_keyboard(doc_id_to_confirm)
                    send_tg_message(confirm_text, reply_markup=confirm_keyboard, parse_mode='MarkdownV2')
    # --- NEW: Delete Execution Callback ---
    elif command_action and command_action.startswith('del_do_'):
        doc_id_to_delete = command_action.replace('del_do_', '')
        if not doc_id_to_delete:
            send_tg_message("Error: Invalid receipt reference in delete confirmation.")
        else:
            # We MUST re-check auth here before deleting
            delete_result_message = delete_receipt_from_firestore(user_id_str, doc_id_to_delete)
            # Edit the original confirmation message to show the result
            if update.callback_query and update.callback_query.message:
                 edit_tg_message(message_id_to_edit=update.callback_query.message.message_id,
                                 text=delete_result_message, # Show result
                                 reply_markup=None, # Remove buttons
                                 parse_mode='MarkdownV2') # Use V2 for backticks
            else: # Should have message context, but fallback to new message
                 send_tg_message(delete_result_message, parse_mode='MarkdownV2')
    # --- NEW: Delete Cancellation Callback ---
    elif command_action and command_action.startswith('del_cancel_'):
        doc_id_cancelled = command_action.replace('del_cancel_', '')
        cancel_message = f"Deletion cancelled for receipt Ref: `{doc_id_cancelled}`."
        # Edit the original confirmation message
        if update.callback_query and update.callback_query.message:
             edit_tg_message(message_id_to_edit=update.callback_query.message.message_id,
                             text=cancel_message,
                             reply_markup=None, # Remove buttons
                             parse_mode='MarkdownV2')
        else:
             send_tg_message(cancel_message, parse_mode='MarkdownV2')

    if command_action == 'main_menu':  # Also triggered by /start for approved users
        send_tg_message("Main Menu:", reply_markup=build_main_menu_keyboard(is_current_user_admin))

    # User Group Menu
    elif command_action == 'group_menu_user':
        send_tg_message("Group Options:", reply_markup=build_user_group_menu_keyboard(user_id_str))
    elif command_action == 'mygroup_info_action':
        send_tg_message(get_my_group_info(user_id_str), parse_mode='Markdown')
    elif command_action == 'leavegroup_action_user':  # User initiated leave (non-admin)
        send_tg_message(user_leave_group(user_id_str))

    # Summaries
    elif command_action == 'summary_current_month':
        send_tg_message(get_spending_by_date_range_for_user(user_id_str, start_of_month, end_of_month))
    elif command_action == 'summary_category_current_month':
        send_tg_message(get_spending_by_category_for_user(user_id_str, current_month_year_str))
    elif command_action == 'summary_store_current_month':
        send_tg_message(get_spending_by_store_for_user(user_id_str, current_month_year_str))
    elif command_action == 'summary_avg_receipt_current_month':
        send_tg_message(get_average_receipt_value_for_user(user_id_str, current_month_year_str))
    elif command_action == 'summary_date_range_prompt':
        send_tg_message("To get a summary for a custom date range, type:\n`/daterange YYYY-MM-DD YYYY-MM-DD`")
    elif command_action == '/daterange':
        if len(command_args_list) == 2:
            send_tg_message(
                get_spending_by_date_range_for_user(user_id_str, command_args_list[0], command_args_list[1]))
        else:
            send_tg_message("Usage: /daterange YYYY-MM-DD_START YYYY-MM-DD_END")

    # User trying to use admin-style group commands
    elif command_action == '/creategroup':
        send_tg_message("Group creation is managed by the administrator. You can ask them to create a group for you.")
    elif command_action == '/joingroup':
        send_tg_message("To join a group, please ask the administrator to add you using your User ID.")

    elif command_action == '/edit':
            # For multi-line input, full_arg_string will contain everything after "/edit "
            # If user just types /edit, full_arg_string will be empty or None.
            # If user types /edit then data on new lines, text_payload might only be /edit,
            # and the actual data is in update.message.text AFTER the first line.
            
            edit_data_payload = ""
            if full_arg_string: # Data was on the same line as /edit
                edit_data_payload = full_arg_string
            elif update.message and update.message.text:
                # Check if message text contains more than just /edit
                lines_in_message = update.message.text.strip().split('\n', 1)
                if len(lines_in_message) > 1 and lines_in_message[0].strip().lower() == '/edit':
                    edit_data_payload = lines_in_message[1].strip() # Use text from second line onwards
                elif lines_in_message[0].strip().lower() == '/edit': # Only /edit was sent
                    pass # Fall through to send instructions

            if not edit_data_payload:
                send_tg_message(
                    "To edit, send a message starting with `/edit` on a new line, then paste the `Ref:` line, followed by the corrected details in this format (one item per line):\n\n"
                    "`Ref: <PASTE_REF_ID_HERE>`\n"
                    "`Store: New Store Name (optional)`\n"
                    "`Date: YYYY-MM-DD (optional)`\n"
                    "`Total: <new_total_price>`\n"
                    "`Item Name 1; Price 1; Category 1; Quantity 1; Unit 1`\n"
                    "`...`\n\n"
                    "Use semicolons (;) to separate item details.",
                    parse_mode='Markdown'
                )
            else:
                edit_result_message = handle_simple_edit_receipt_command(user_id_str, edit_data_payload)
                send_tg_message(edit_result_message, parse_mode='Markdown')

    # Fallback for unrecognized text if not image and not a known command/callback, and not /start
    elif update.message and update.message.text and not command_action and text_payload != '/start':
        send_tg_message("I didn't understand that. Send a receipt image or use /menu for options.")
        action_handled_for_user = False  # Explicitly not handled

    # If it was a callback query but no specific handler matched above
    elif update.callback_query and not command_action:  # Should have been caught by specific handlers
        print(f"WEBHOOK: Unhandled callback_data '{text_payload}' for user {user_id_str}")
        action_handled_for_user = False

    elif not text_payload and not (update.message and update.message.photo):  # No text, no photo, no callback
        print(f"WEBHOOK: Received an empty or unhandled update type for user {user_id_str}.")
        action_handled_for_user = False

    # --- NEW: List Receipts Command ---
    elif command_action == 'list_receipts_action' or command_action == '/listreceipts':
         receipts, context = get_recent_receipts_for_user(user_id_str)
         message_text, reply_markup = format_receipt_list_for_display(receipts, context)
         send_tg_message(message_text, reply_markup=reply_markup, parse_mode='Markdown')
    # --- END NEW ---

    # Handler for Delete Receipts button
    elif command_action == 'delete_receipts_action' or command_action == '/deletereceipts':
         if not is_user_approved(user_id_str): send_tg_message("Your account is not approved.")
         else:
             receipts, context = get_recent_receipts_for_user(user_id_str)
             # Use the specific formatting function for delete buttons
             message_text, reply_markup = format_receipt_list_for_delete(receipts, context) 
             send_tg_message(message_text, reply_markup=reply_markup, parse_mode='Markdown')

    # --- NEW: View Receipt Callback Handling ---
    elif command_action and command_action.startswith('view_receipt_'):
        doc_id_to_view = command_action.replace('view_receipt_', '')
        if not doc_id_to_view:
             send_tg_message("Error: Invalid receipt reference in button.")
        else:
            receipt_details = get_receipt_details_for_view(doc_id_to_view)
            if not receipt_details:
                send_tg_message(f"Error: Could not find receipt details for Ref: `{doc_id_to_view}`", parse_mode='Markdown')
            else:
                # Authorization check (same logic as for edit)
                is_admin_viewing = is_user_admin(user_id_str)
                user_is_uploader_view = (receipt_details.get('telegramUserId') == user_id_str)
                user_group_id_view = get_user_group_id(user_id_str)
                receipt_group_id_view = receipt_details.get('groupId')
                user_in_receipt_group_view = (receipt_group_id_view is not None and user_group_id_view == receipt_group_id_view)
                if not (is_admin_viewing or user_is_uploader_view or user_in_receipt_group_view):
                     send_tg_message(f"Error: You are not authorized to view receipt `{doc_id_to_view}`.", parse_mode='Markdown')
                else:
                    # Format and send the detailed view
                    formatted_detail_view = format_single_receipt_for_view(receipt_details)
                    # Decide whether to edit the original list message or send a new one
                    # Sending new is simpler for now
                    send_tg_message(formatted_detail_view, parse_mode='Markdown') # Use V2 for better code block/backtick handling
    # --- END NEW ---

    if not action_handled_for_user and command_action:  # A command was parsed but no handler hit
        print(f"WEBHOOK: Parsed command_action '{command_action}' but no handler was matched for user {user_id_str}.")
        # send_tg_message(f"Sorry, I don't know how to handle '{command_action}'. Use /menu for options.") # Optional

    return https_fn.Response("OK", status=200)  # Default OK response if no specific error response sent