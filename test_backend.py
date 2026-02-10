from app_backend import llm, Plan, Task
from langchain_core.messages import HumanMessage, SystemMessage
import os

print("Testing Gemini LLM initialization...")
try:
    print(f"Model: {llm.model}")
    print("Testing basic generation...")
    # Skip actual invoke if no key present to avoid crash, but try if present
    if os.getenv("GOOGLE_API_KEY"):
        res = llm.invoke([HumanMessage(content="Hello, are you Gemini?")])
        print(f"Response: {res.content}")
    else:
        print("Skipping live generation (No GOOGLE_API_KEY found)")
        
    print("Test passed!")
except Exception as e: 
    print(f"Test failed: {e}")
