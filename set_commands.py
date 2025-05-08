# set_commands.py
import telegram
import os
import sys
from telegram import BotCommand

# --- Configuration ---
# Load the bot token from an environment variable for security
# Set this environment variable in your terminal before running the script:
# Windows PowerShell: $env:TELEGRAM_TOKEN="YOUR_BOT_TOKEN"
# Linux/macOS Bash: export TELEGRAM_TOKEN="YOUR_BOT_TOKEN"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

if not TOKEN:
    print("Error: TELEGRAM_TOKEN environment variable not set.")
    print("Please set it before running the script (e.g., export TELEGRAM_TOKEN='your:token').")
    sys.exit(1)  # Exit if token is missing

# --- Define the list of commands for users ---
# These commands will appear in the standard Telegram command menu '[/]'
# Keep descriptions concise and user-friendly.
# Admin commands are intentionally omitted from this public list.
user_commands = [
    BotCommand("start", "▶️ Start / Show main menu"),
    BotCommand("menu", "📋 Show main options menu"),
    BotCommand("listreceipts", "📄 List recent receipts"),
    BotCommand("deletereceipts", "🗑️ Select a recent receipt to delete"),
    BotCommand("mygroup", "👥 Show your current group info"),
    BotCommand("leavegroup", "🚪 Leave your current group"),
    BotCommand("edithelp", "✍️ Help for editing receipt details"),
    BotCommand("daterange", "📅 Sum spending for a date range"),
    # Note: /edit itself is not listed as it requires arguments and is better explained by /edithelp or after processing.
]

# --- Set the commands ---
try:
    print(f"Attempting to set commands for bot with token ending in ...{TOKEN[-5:]}")
    bot = telegram.Bot(token=TOKEN)

    # Set commands for the default scope (all private chats and groups)
    # You can define different commands for different scopes (e.g., BotCommandScopeAllPrivateChats()) if needed
    success = bot.set_my_commands(commands=user_commands)

    if success:
        print("\nSuccessfully set bot commands!")
        print("Changes might take a short while to appear in all Telegram clients (try restarting Telegram).")
        print("\nCommands set:")
        for cmd in user_commands:
            print(f"  /{cmd.command} - {cmd.description}")
    else:
        print("\nFailed to set bot commands. The API call returned failure.")

except telegram.error.Unauthorized:
    print("\nTelegram API Error: Unauthorized. Please check if your bot token is correct and valid.")
except telegram.error.TelegramError as e:
    print(f"\nTelegram API Error: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")