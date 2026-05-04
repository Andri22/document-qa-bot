# Document Q&A Bot

An AI-powered chatbot that answers questions from business documents.
Built with LangChain, FastAPI, and Streamlit.

## Tech Stack
- Python 3.11
- LangChain
- OpenAI GPT-4o-mini
- ChromaDB
- Streamlit
- FastAPI

## How to Run

1. Clone the repository
   git clone https://github.com/Andri22/document-qa-bot.git
   cd document-qa-bot

2. Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Set up environment variables
   cp .env.example .env
   # Add your OPENAI_API_KEY to .env

5. Run the chatbot
   python stage1_foundations/chat.py