import streamlit as st
import os
import io
import textwrap
# Note: Ensure you have installed pypdf -> pip install pypdf
from pypdf import PdfReader

# Import Google Generative AI and LangChain components
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
# Updated imports for new LangChain structure
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# --- Helper Functions for Core Logic ---

def get_text_from_file(uploaded_file):
    """
    Extracts text from an uploaded file (PDF or TXT).
    
    Args:
        uploaded_file: The file uploaded via Streamlit's file_uploader.

    Returns:
        A string containing the extracted text, or None if the file type is unsupported.
    """
    text = ""
    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    elif uploaded_file.type == "text/plain":
        # To read file as string:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
        text = stringio.read()
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return None
    return text

def get_text_chunks(text):
    """
    Splits a long text into smaller, manageable chunks.

    Args:
        text: The string of text to be split.

    Returns:
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource(show_spinner="Processing document...")
def get_vector_store(_text_chunks, api_key):
    """
    Creates and caches a FAISS vector store from text chunks.
    The caching prevents re-processing the same document.

    Args:
        _text_chunks: A list of text chunks from the document.
        api_key: The Google API key.

    Returns:
        A FAISS vector store object.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(_text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store. Have you entered your API key correctly? Error: {e}")
        return None

def get_conversational_chain(api_key):
    """
    Creates a conversational Q&A chain with a custom prompt for the Gemini model.

    Args:
        api_key: The Google API key.

    Returns:
        A LangChain conversational chain object.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- UI and Application Flow ---

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("🧠 Smart Assistant for Research Summarization")
st.markdown("Built with Google Gemini Pro")

# --- Sidebar for API Key and Instructions ---
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Enter your Google API Key:", type="password", key="google_api_key")
    
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("1. **Enter your API Key:** Get your key from the [Google AI Studio](https://makersuite.google.com/app/apikey).")
    st.markdown("2. **Upload a document:** PDF or TXT files are supported.")
    st.markdown("3. **Get a summary:** An automatic summary will be generated.")
    st.markdown("4. **Interact:** Choose a mode to either ask questions or take a challenge.")

# --- Main Application Logic ---

# We no longer need the global genai.configure, as we pass the key directly.
if google_api_key:
    st.session_state.api_key_configured = True
else:
    st.session_state.api_key_configured = False

# 1. Document Upload
st.header("1. Upload Your Document")
uploaded_file = st.file_uploader("Upload a PDF or TXT file to begin", type=["pdf", "txt"])

# Initialize session state variables
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "challenge_questions" not in st.session_state:
    st.session_state.challenge_questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0

# Process the document only if a new file is uploaded
if uploaded_file is not None:
    if not st.session_state.api_key_configured:
        st.warning("Please enter your Google API Key in the sidebar to proceed.")
    else:
        # Process the document and generate summary
        with st.spinner("Reading and processing your document..."):
            raw_text = get_text_from_file(uploaded_file)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks, google_api_key)
                st.session_state.document_processed = True
                
                # 2. Auto-Summary
                st.header("2. Auto-Summary")
                with st.spinner("Generating summary..."):
                    # Pass the API key directly and use .invoke()
                    model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=google_api_key)
                    summary_prompt = f"Summarize the following document concisely in no more than 150 words:\n\n{raw_text[:15000]}"
                    summary = model.invoke(summary_prompt)
                    st.success(textwrap.fill(summary, width=100))
        
        st.divider()

# 3. Interaction Modes
if st.session_state.document_processed:
    st.header("3. Interact with the Document")
    mode = st.radio("Choose an interaction mode:", ("Ask Anything", "Challenge Me"), horizontal=True)

    # --- "Ask Anything" Mode ---
    if mode == "Ask Anything":
        st.subheader("❓ Ask a Question")
        user_question = st.text_input("Ask anything about the content of your document:")
        
        if user_question:
            with st.spinner("Searching for the answer..."):
                vector_store = st.session_state.vector_store
                if vector_store:
                    docs = vector_store.similarity_search(user_question)
                    chain = get_conversational_chain(google_api_key)
                    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    
                    st.write("### Answer")
                    st.info(textwrap.fill(response["output_text"], width=100))

                    with st.expander("Show Justification (Source Snippets)"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Snippet {i+1}:**")
                            st.write(textwrap.fill(doc.page_content, width=100))
                            st.markdown("---")
                else:
                    st.error("Vector store not initialized. Please re-upload the document.")

    # --- "Challenge Me" Mode ---
    elif mode == "Challenge Me":
        st.subheader("🧠 Take a Challenge")
        
        # Generate questions if they don't exist
        if not st.session_state.challenge_questions:
            with st.spinner("Generating challenge questions..."):
                vector_store = st.session_state.vector_store
                if vector_store:
                    # Retrieve some random-ish chunks to generate questions from
                    docs = vector_store.similarity_search("Give me 3 key concepts from the document")
                    context_for_questions = "\n".join([doc.page_content for doc in docs])
                    
                    # Pass the API key directly and use .invoke()
                    model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=google_api_key)
                    question_gen_prompt = f"""
                    Based on the provided text, generate exactly three distinct, logic-based or comprehension-focused questions.
                    Each question should be on a new line. Do not number them.
                    
                    Context:
                    {context_for_questions}
                    """
                    generated_text = model.invoke(question_gen_prompt)
                    st.session_state.challenge_questions = [q for q in generated_text.strip().split('\n') if q]
                    st.session_state.current_question_index = 0

        # Display the current question and handle user's answer
        if st.session_state.challenge_questions:
            idx = st.session_state.current_question_index
            if idx < len(st.session_state.challenge_questions):
                current_question = st.session_state.challenge_questions[idx]
                st.markdown(f"**Question {idx + 1}/{len(st.session_state.challenge_questions)}:**")
                st.info(current_question)
                
                user_answer = st.text_area("Your Answer:", key=f"answer_{idx}")

                if st.button("Submit Answer", key=f"submit_{idx}"):
                    with st.spinner("Evaluating your answer..."):
                        # Retrieve context relevant to the question to evaluate the answer
                        docs = st.session_state.vector_store.similarity_search(current_question)
                        context_for_eval = "\n".join([doc.page_content for doc in docs])
                        
                        # Pass the API key directly and use .invoke()
                        model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=google_api_key)
                        eval_prompt = f"""
                        Evaluate the user's answer based on the provided context and question.
                        Provide feedback on whether the answer is correct, partially correct, or incorrect, and give a brief justification based ONLY on the context.

                        Context:
                        {context_for_eval}

                        Question:
                        {current_question}

                        User's Answer:
                        {user_answer}

                        Evaluation:
                        """
                        evaluation = model.invoke(eval_prompt)
                        st.markdown("### Feedback")
                        st.success(evaluation)

                        # Move to the next question
                        st.session_state.current_question_index += 1
                        st.rerun()

            else:
                st.success("You have completed the challenge! 🎉")
                if st.button("Start a New Challenge"):
                    st.session_state.challenge_questions = []
                    st.session_state.current_question_index = 0
                    st.rerun()
        else:
            st.warning("Could not generate challenge questions. Try uploading a different document.")
