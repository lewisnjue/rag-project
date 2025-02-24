from google import genai
import dotenv
import os 
dotenv.load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)
print(response.text)