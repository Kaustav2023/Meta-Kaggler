import google.generativeai as genai
from config import Config

# Configure Gemini
genai.configure(api_key=Config.GEMINI_API_KEY)

# List all available models
print("🔍 Available Gemini Models:\n")
for model in genai.list_models():
    print(f"  - {model.name}")
    print(f"    Display Name: {model.display_name}")
    print(f"    Supported Methods: {model.supported_generation_methods}")
    print()
