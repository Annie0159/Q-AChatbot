# Personalized Document Q&A Chatbot
This project implements a Retrieval-Augmented Generation (RAG) based chatbot that can answer questions based on a collection of your own documents. It allows you to interact with your personal knowledge base in a conversational manner, providing accurate and context-aware responses.

## Features 
Data: Load text files (.txt) and PDF documents (.pdf) into the chatbot's knowledge base.
Intelligent Chunking: Documents are automatically split into smaller, semantically meaningful chunks to optimize retrieval.Vector Embeddings: Utilizes Google's gemini-embedding-001 model to convert text chunks into high-dimensional vectors, enabling efficient semantic search.
Local Vector Database: Uses ChromaDB to store and retrieve document embeddings locally.
Generative AI Responses: Leverages Google's Gemini Pro model to generate natural language answers grounded in the retrieved document context.
Conversational Interface: Simple command-line interface for asking questions and receiving answers.

## Technologies Used
- Python 3.9
- LangChain: Framework for building LLM applications.
- langchain_community.document_loaders: For loading various document types.
- langchain.text_splitter: For splitting documents into chunks.
- langchain_google_genai: For integrating Google's Gemini models (LLM and Embeddings).
- - langchain_community.vectorstores.Chroma: For the local vector database.
- pypdf: For parsing PDF documents.
- python-dotenv: (Optional but recommended) For securely loading environment variables.
- Google Gemini API: For generative AI capabilities and text embeddings.

## Setup Instructions
Follow these steps to get your chatbot up and running:

### 1. Clone the Repository
```bash
git clone https://github.com/Annie0159/Q-AChatbot.git
cd Q-AChatbot
```

### 2. Set up a Python Virtual Environment.
python -m venv rag_env
source rag_env/bin/activate # On Windows: .\rag_env\Scripts\activate

### 3. Install Dependencies 

### 4. Obtain Your Google API Key
- You'll need an API key to use Google's Gemini models.
- Go to Google AI Studio.Log in with your Google account.
- Click "Create API key in new project" (or select an existing one).
- Copy your generated API key immediately.

### 5. Prepare Your Documents
Create a folder named data/ in the root directory of your project.Place your .txt and .pdf documents that you want the chatbot to answer questions from into this data/ folder.
Example:Q-AChatbot/
├── .env
├── data/
│   ├── my_notes_on_ai.txt
│   └── project_manual.pdf
└──  templates
│   └── index.html
└──   app.py  # main Python script



## Usage
Run the script: python app.py

## License
This project is open-source and available under the MIT License.