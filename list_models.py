import os
import getpass
from google import genai

key = os.getenv("GOOGLE_API_KEY")
if not key:
    try:
        key = getpass.getpass("Enter your GOOGLE_API_KEY: ")
    except:
        key = input("Enter your GOOGLE_API_KEY: ")

if not key:
    print("No key provided.")
    exit(1)

client = genai.Client(api_key=key)

print("\nListing available models...")
try:
    for m in client.models.list(config={}):
        # Simple print to see what's available
        # The new SDK's model object structure is different
        print(f"- {m.name} ({m.display_name})")
except Exception as e:
    print(f"Error: {e}")
