from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

# Load environment variables from .env
load_dotenv()

# Initialize the model (it automatically uses the loaded OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini")

# Test the connection
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are senior Agentic AI engineer"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)
# Langchain chain
chain = prompt | llm

# Chat History
history = ChatMessageHistory()

while True:
    user_input = input("Enter question (type 'quit' to exit): ")
    response = chain.invoke({"user_input": user_input, "history": history.messages})

    history.add_user_message(user_input)
    history.add_ai_message(response.content)

    print(response.content)
    if user_input.lower() == "quit":
        print("Exiting loop")
        break
