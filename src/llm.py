import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=os.getenv("AIzaSyBmahRQr-CKQwQqvH_3_Eayu9itHmWcJng")
    )
