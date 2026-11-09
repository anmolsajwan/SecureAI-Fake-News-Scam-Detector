
import os
import tomllib
from google import genai

api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    try:
        with open(".streamlit/secrets.toml", "rb") as file:
            secrets = tomllib.load(file)
            api_key = secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

if not api_key:
    print("API KEY NOT FOUND")
else:
    client = genai.Client(api_key=api_key)

    print("\nAvailable Gemini Models:\n")

    for model in client.models.list():
        print(model.name)
