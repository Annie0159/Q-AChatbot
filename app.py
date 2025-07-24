
from flask import Flask, render_template, request, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)

rag_chain = None
vectorstore = None
google_api_key = None

def initialize_rag_components():
    global rag_chain, vectorstore, google_api_key

    try:
        google_api_key = os.environ["GOOGLE_API_KEY"]
    except KeyError:
        print("Error: The 'GOOGLE_API_KEY' environment variable is not set.")
        exit() 

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key,
        task_type="RETRIEVAL_DOCUMENT"
    )

    chroma_db_path = "./chroma_db"

    if os.path.exists(chroma_db_path) and os.path.isdir(chroma_db_path):
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings_model)
    else:
        print("ChromaDB not found. Processing documents and creating new embeddings...")
        document_path = "data/"
        documents = []

        if not os.path.exists(document_path):
            print(f"Error: Document path '{document_path}' does not exist.")
            print("Please ensure your 'data/' directory with documents is in the same location as app.py.")
            exit()

        for file_name in os.listdir(document_path):
            file_path = os.path.join(document_path, file_name)
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_name.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            # Add more loaders for other file types if needed

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks.")

        # Create a ChromaDB instance and add the chunks
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=chroma_db_path
        )
        vectorstore.persist()
        print("Embeddings created and stored in ChromaDB.")

    # Setup the LLM and RAG chain
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.7, google_api_key=google_api_key)

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the provided context.
    If you don't know the answer based on the context, state that you don't know.
    Be concise and to the point.

    Context: {context}

    Question: {input}
    """)

    retriever = vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain initialized and ready.")

with app.app_context():
    initialize_rag_components()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    if rag_chain is None:
        return jsonify({"error": "RAG chain not initialized. Please check server logs."}), 500

    try:
        response = rag_chain.invoke({"input": user_question})
        bot_answer = response['answer']
        return jsonify({"answer": bot_answer})
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True) 
