from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Initialize the model (it automatically uses the loaded OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini")

# Test the connection
response = llm.invoke("Hello, how are you?")
print(response.content)
