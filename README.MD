# Spendy Pants - Telegram Grocery Receipt Tracker 💸🧾

A Telegram bot designed to track grocery expenses by analyzing receipt images using AI (Google Gemini) and storing structured data in Firebase Firestore. It simplifies expense tracking through a convenient chat interface, supports group/family sharing, and includes admin controls.

## Features

*   **Receipt Upload:** Send receipt images directly to the Telegram bot.
*   **AI Data Extraction:** Uses Google Gemini (Vision model) to perform OCR, extract item details (name, price, quantity, unit), store name, date (defaults to upload date), and total amount. Handles Hebrew receipts with English translation.
*   **Automated Categorization:** AI assigns grocery categories to extracted items.
*   **Secure Storage:** Data is stored securely in Google Cloud Firestore.
*   **User Approval System:** New users require admin approval before using the bot.
*   **Group Functionality:** Users can be added to groups (e.g., families) by the admin to share and view combined expenses.
*   **Expense Reporting:**
    *   Summaries by date range (current month, custom range).
    *   Breakdowns by category and store for specific months.
    *   Calculation of average receipt value.
*   **Receipt Management:**
    *   List recent receipts (personal or group).
    *   View detailed receipt information.
    *   Edit processed receipts using a simple text format.
    *   Delete receipts (requires confirmation, restricted to uploader or admin).
*   **Admin Panel:** Commands for managing users (approve, ban, list) and groups (create, delete, add/remove members).
*   **Telegram Command Menu:** Configurable list of commands accessible via the `[/]` menu button in Telegram.

## Technology Stack

*   **Language:** Python 3.11
*   **Cloud Platform:** Google Cloud Platform (GCP)
*   **Backend Service:** Firebase Cloud Functions (2nd Gen, Python Runtime)
*   **Database:** Google Cloud Firestore
*   **AI / LLM:** Google Gemini API (e.g., `gemini-1.5-flash-latest`)
*   **Telegram Bot Library:** `python-telegram-bot` (v13.x recommended)
*   **Framework:** Google Cloud Functions Framework
*   **Deployment:** Firebase CLI

## Setup and Installation

**1. Prerequisites:**

*   [Python](https://www.python.org/) (Version 3.11 recommended) and `pip`.
*   [Node.js and npm](https://nodejs.org/) (for Firebase CLI).
*   [Google Cloud SDK (gcloud CLI)](https://cloud.google.com/sdk/docs/install) installed and initialized (`gcloud init`).
*   [Firebase CLI](https://firebase.google.com/docs/cli#setup_update_cli) installed (`npm install -g firebase-tools`).

**2. Firebase Project:**

*   Create a new Firebase project in the [Firebase Console](https://console.firebase.google.com/).
*   Upgrade the project to the "Blaze (pay as you go)" plan, as Cloud Functions (beyond the free tier) and external network requests (like calling Gemini API) require it.
*   Enable **Firestore** database in the Firebase Console. Choose a region (e.g., `me-west1` Tel Aviv, as discussed) during setup. Note this region.
*   Enable the following Google Cloud APIs for your project in the [GCP Console APIs & Services Library](https://console.cloud.google.com/apis/library):
    *   Cloud Functions API
    *   Cloud Build API
    *   Artifact Registry API
    *   Cloud Run Admin API (for 2nd Gen Functions)
    *   Generative Language API (for Gemini)
    *   Cloud Firestore API
    *   Identity and Access Management (IAM) API
    *   Cloud Resource Manager API
    *   Service Usage API (usually enabled by default)

**3. Local Setup:**

*   Clone this repository (if applicable) or place the code in your project directory.
*   Navigate to the project root (`spendy-pants`):
    ```bash
    cd path/to/spendy-pants
    ```
*   Log in to Firebase:
    ```bash
    firebase login
    ```
*   Link your local project to your Firebase project:
    ```bash
    firebase use --add # Select your project from the list
    ```
*   Navigate to the functions directory:
    ```bash
    cd functions
    ```
*   Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    # Windows PowerShell:
    .\venv\Scripts\activate
    # Linux/macOS:
    # source venv/bin/activate
    ```
*   Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
*   Set up local Application Default Credentials (ADC) for accessing GCP services (like Firestore) locally:
    ```bash
    gcloud auth application-default login
    ```

## Configuration (Environment Variables)

The deployed Cloud Function requires the following environment variables to be set:

*   `TELEGRAM_TOKEN`: Your Telegram bot token obtained from @BotFather on Telegram.
*   `GEMINI_API_KEY`: Your API key for the Google Gemini API. Obtainable from [Google AI Studio](https://aistudio.google.com/app/apikey) or the GCP Console (Generative Language API section).
*   `ADMIN_USER_ID`: The numerical Telegram User ID of the person who will act as the bot administrator. You can find your ID by messaging `@userinfobot` on Telegram.

**Setting Environment Variables:**

*   **For Deployment:** Set these in the Google Cloud Console when deploying or editing your Cloud Function:
    1.  Go to GCP Console -> Cloud Functions -> Select `telegram_webhook`.
    2.  Click "EDIT".
    3.  Navigate to "Runtime, build, connections and security settings".
    4.  Under "Runtime environment variables", add the key-value pairs.
    5.  Click "NEXT" and "DEPLOY" to save.
*   **For Local Debugging:**
    *   Set them in your terminal session before running `functions-framework`:
        ```bash
        # PowerShell
        $env:TELEGRAM_TOKEN="..."; $env:GEMINI_API_KEY="..."; $env:ADMIN_USER_ID="..."
        # Bash/Zsh
        export TELEGRAM_TOKEN="..."
        export GEMINI_API_KEY="..."
        export ADMIN_USER_ID="..."
        ```
    *   Or use a `.env` file in the `functions` directory if using an IDE debugger that supports it (e.g., VS Code `launch.json`).
    *   You might also need `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account JSON key for local Firestore access if `gcloud auth application-default login` doesn't work for your setup.

## Deployment

1.  Navigate to your project root directory (`spendy-pants`).
2.  Run the deployment command:
    ```bash
    firebase deploy --only functions
    ```
3.  **Check Deployment Status:** Monitor the deployment in the CLI. If it says "Deploy complete!" but you suspect issues:
    *   Check **Google Cloud Build** history in the GCP Console for build logs and errors.
    *   Check **Google Cloud Functions** in the GCP Console (filter by the correct region, e.g., `me-west1`) to see the function's status.
4.  **Get Trigger URL:** Once successfully deployed, find the HTTP Trigger URL in the GCP Console (Cloud Functions -> `telegram_webhook` -> Trigger tab). It will look like `https://<function-name>-<hash>-<region>.a.run.app`.
5.  **Set Telegram Webhook:** Use `curl` or a similar tool to tell Telegram where to send updates:
    ```bash
    curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook?url=<YOUR_DEPLOYED_FUNCTION_URL>"
    ```
    (Replace token and URL). Check for an `{"ok":true}` response.

## Usage (Bot Commands)

Interact with the bot directly on Telegram.

**For All Users:**

*   Send an **image** of a grocery receipt to have it processed.
*   `/start` or `/menu`: Show the main menu with available actions.

**For New/Pending Users:**

*   `/start`: Bot replies with the user's Telegram ID and instructions to contact the admin for approval.

**For Approved Users:**

*   **(Main Menu Buttons)** Access reports and options via inline keyboard buttons.
*   `/listreceipts`: Show a list of the ~20 most recent receipts (personal or group) with buttons to view details.
*   `/deletereceipts`: Show a list of recent receipts with buttons to request deletion (requires confirmation).
*   `/mygroup`: Display information about the group the user belongs to, if any.
*   `/leavegroup`: Allows the user to leave their current group (cannot be used by admin via this command).
*   `/edithelp`: Shows instructions on how to format the `/edit` command.
*   `/edit`: Followed by multi-line text in the specified format to correct a previously processed receipt. (Requires the `Ref: <ID>` from the initial processing message).
    ```
    /edit
    Ref: <RECEIPT_ID>
    Store: Corrected Store Name
    Date: YYYY-MM-DD
    Total: <correct_total>
    Item 1 Name; Price 1; Category 1; Qty 1; Unit 1
    Item 2 Name; Price 2; Category 2; Qty 2; Unit 2
    ...
    ```
*   `/daterange <YYYY-MM-DD> <YYYY-MM-DD>`: Get a spending summary for a specific date range.

**For Admin Users Only:**

*   Access to all user commands.
*   **(Admin Panel Button)** In the main menu, access admin-specific actions.
*   `/adminhelp`: List available admin commands.
*   `/listusers [pending|approved|banned|all]`: List users based on status.
*   `/approveuser <user_id>`: Approve a pending user.
*   `/banuser <user_id>`: Ban a user (restricts access).
*   `/setuserstatus <user_id> <status>`: Set status directly (`approved`, `banned`, `pending_approval`).
*   `/admincreategroup "<Group Name>" [UserID1 UserID2 ...]`: Create a new group and optionally add initial approved members. Admin's own group status is unaffected.
*   `/addusertogroup <user_id> <group_id>`: Add an existing user to a group (moves them if they were in another).
*   `/removeuserfromgroup <user_id> <group_id>`: Remove a user from a specific group.
*   `/deletegroup <group_id>`: Permanently delete a group and remove all member associations.

**Telegram Command Menu `[/]`:**

*   Run the separate `set_commands.py` script once locally (after setting `TELEGRAM_TOKEN` env var) to configure the commands that appear in the Telegram `[/]` menu next to the text input.

## Firestore Indexes

Firestore requires composite indexes for complex queries (e.g., filtering by `groupId` or `telegramUserId` *and* ordering/filtering by `date`). If a query fails due to a missing index, the error message in the **Google Cloud Logging** section for your Cloud Function will contain a direct link to the GCP Console to create the required index. This is expected behavior. Common indexes needed will be:

*   `receipts`: `groupId` (Asc), `date` (Desc), `uploadTimestamp` (Desc)
*   `receipts`: `telegramUserId` (Asc), `date` (Desc), `uploadTimestamp` (Desc)

Click the link in the logs and wait for the index to build (status changes from "Building" to "Enabled").

## Contributing

This is currently maintained as a personal project. Suggestions and feedback are welcome via issues (if hosted on a platform like GitHub).

## License

MIT License