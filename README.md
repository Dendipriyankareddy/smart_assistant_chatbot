Smart Assistant for Research Summarization
This project is a web-based, AI-powered assistant designed to help users quickly understand and interact with large documents. The assistant can read uploaded PDF or TXT files, provide concise summaries, answer free-form questions, and challenge the user with logic-based questions derived from the document's content.

This application is built using Streamlit, LangChain, and the Google Gemini Pro API.

Features
Document Upload: Supports both PDF and TXT file formats.

Auto-Summary: Instantly generates a concise summary (≤ 150 words) of the uploaded document.

Ask Anything Mode: Allows users to ask free-form questions about the document. The assistant provides answers grounded in the text, complete with source snippets for justification.

Challenge Me Mode: The assistant generates three unique, logic-based questions from the document and evaluates the user's answers, providing feedback and justification.

Intuitive UI: A clean and responsive user interface built with Streamlit.

Setup and Installation
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.8 or higher

A Google API Key for the Gemini API. You can obtain one from Google AI Studio.

1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name>

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries with a single command:

pip install streamlit pypdf google-generativeai langchain-google-genai langchain langchain-community langchain-core faiss-cpu

4. Run the Application
Once the dependencies are installed, you can run the Streamlit application.

streamlit run app.py

The application will open automatically in your web browser. Enter your Google API Key in the sidebar, upload a document, and begin interacting with the assistant.

Architecture and Reasoning Flow
The application is built with a simple yet powerful architecture, leveraging modern AI tools to deliver a seamless experience.

Frontend: Streamlit was chosen for its ability to rapidly create beautiful, interactive data and AI applications with pure Python. This allows for a focus on the core logic rather than complex web development.

AI Orchestration: LangChain serves as the backbone for all AI-related tasks. It simplifies the process of "chaining" together language models with document loaders, text splitters, and vector stores.

Language Model: Google Gemini Pro (gemini-1.0-pro) is used for all generative tasks, including summarization, question answering, question generation, and evaluation. It provides a strong balance of performance and cost.

Workflow
The application follows a logical flow based on the principles of Retrieval-Augmented Generation (RAG):

Document Upload & Text Extraction: The user uploads a PDF or TXT file. The text is extracted using the pypdf library.

Text Chunking: The extracted text is split into smaller, overlapping chunks using LangChain's RecursiveCharacterTextSplitter. This is necessary to fit the context within the language model's limits.

Embedding & Vector Storage: Each text chunk is converted into a numerical representation (embedding) using Google's embedding-001 model. These embeddings are then stored in a FAISS vector store, which is a highly efficient library for similarity search. This vector store acts as the document's searchable knowledge base.

Interaction Modes:

Auto-Summary: The initial text is passed to the Gemini model with a prompt to generate a concise summary.

Ask Anything (Q&A): When a user asks a question, it is embedded, and the FAISS vector store is searched to find the most relevant text chunks. These chunks (the context) and the user's question are passed to the Gemini model, which generates an answer based only on the provided context. The source chunks are provided as justification.

Challenge Me: The application retrieves relevant chunks from the document and uses the Gemini model to generate three unique questions. When the user submits an answer, the model evaluates it against the original context and provides reasoned feedback.
