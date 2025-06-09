<<<<<<< HEAD
=======
# src/llm.py
>>>>>>> e10216675b09a9e6e5fe87885184f76e4f60b7a1
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
<<<<<<< HEAD
        google_api_key=os.getenv("AIzaSyBmahRQr-CKQwQqvH_3_Eayu9itHmWcJng")
=======
        google_api_key=os.getenv("AIzaSyAwHAkDEc2rMrMj_1Ga5dHFzBw1___5n3Y")
>>>>>>> e10216675b09a9e6e5fe87885184f76e4f60b7a1
    )
