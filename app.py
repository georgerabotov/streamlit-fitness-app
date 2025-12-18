import streamlit as st
import re
import hashlib
import time
import tiktoken
import PyPDF2
import excel_processor
from io import BytesIO
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set page title and configure layout
st.set_page_config(page_title="Fitness Data Assistant", layout="wide")
st.title("Fitness Data Assistant")

# Initialize session state for tracking progress
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'current_chunk' not in st.session_state:
    st.session_state.current_chunk = 0
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0
if 'chunks_to_process' not in st.session_state:
    st.session_state.chunks_to_process = []
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = None

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Pinecone client
pineconeRegion = st.secrets["PINECONE_ENV"]
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# Create index if it doesn't exist
if 'fitness-workouts' not in pc.list_indexes().names():
    pc.create_index(
        name='fitness-workouts',
        dimension=1536,  # Dimension for text-embedding-ada-002
        metric='cosine',  # Use cosine similarity for embeddings
        spec=ServerlessSpec(
            cloud='aws',
            region=pineconeRegion
        )
    )

# Connect to the existing index
index = pc.Index('fitness-workouts')

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by text-embedding-ada-002

# Helper function to count tokens
def count_tokens(text):
    """Count the number of tokens in a text string"""
    tokens = tokenizer.encode(text)
    return len(tokens)

# Helper function to generate ASCII-compliant IDs
def generate_safe_id(text, prefix="data"):
    """Generate a safe, ASCII-compliant ID for Pinecone vectors"""
    # Create a hash of the text
    hash_object = hashlib.md5(text.encode('utf-8'))
    # Get the hexadecimal representation (guaranteed to be ASCII)
    hex_dig = hash_object.hexdigest()
    # Return a prefixed ID to make it more identifiable
    return f"{prefix}-{hex_dig[:16]}"

# Helper function to split text into paragraphs
def split_into_paragraphs(text):
    """Split text into paragraphs using a consistent method"""
    # Split on two or more newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    # Filter out empty paragraphs
    return [p for p in paragraphs if p.strip()]

# Helper function to extract text from PDF
def extract_text_from_pdf(file_content):
    """Extract text from PDF file"""
    pdf_file = BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    # Extract text from each page
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n\n"
    
    return text

# Helper function to chunk text to fit within token limits
def chunk_text_by_tokens(text, max_tokens=4000):
    """Split text into chunks that fit within token limits"""
    # If text is already within token limit, return it as a single chunk
    if count_tokens(text) <= max_tokens:
        return [text]
    
    # Use RecursiveCharacterTextSplitter for more intelligent splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=200,  # Some overlap to maintain context
        length_function=count_tokens,
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

# Helper function to process chunks in batches
def process_chunks(chunks, prefix="data", batch_size=5):
    """Process text chunks in batches to prevent timeouts"""
    # Initialize or update session state
    if st.session_state.chunks_to_process != chunks:
        st.session_state.chunks_to_process = chunks
        st.session_state.current_chunk = 0
        st.session_state.total_chunks = len(chunks)
        st.session_state.processing_complete = False
        st.session_state.progress_bar = st.progress(0.0)
    
    # If processing is already complete, just return
    if st.session_state.processing_complete:
        return True
    
    # Process the next batch of chunks
    start_idx = st.session_state.current_chunk
    end_idx = min(start_idx + batch_size, st.session_state.total_chunks)
    
    for i in range(start_idx, end_idx):
        chunk = st.session_state.chunks_to_process[i]
        try:
            # Skip empty chunks
            if not chunk or not chunk.strip():
                continue
                
            # Check token count and split if necessary
            token_count = count_tokens(chunk)
            if token_count > 8000:  # Slightly below the 8192 limit for safety
                st.info(f"Chunk {i+1} is too large ({token_count} tokens). Breaking it into smaller pieces.")
                sub_chunks = chunk_text_by_tokens(chunk)
                
                # Process each sub-chunk
                for j, sub_chunk in enumerate(sub_chunks):
                    # Generate embedding for the sub-chunk
                    response = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=sub_chunk
                    )
                    embedding = response.data[0].embedding
                    
                    # Generate a safe ASCII ID
                    safe_id = generate_safe_id(sub_chunk, prefix=f"{prefix}-{i}-{j}")
                    
                    # Store embedding in Pinecone with the safe ID and metadata
                    index.upsert([
                        {
                            "id": safe_id,
                            "values": embedding,
                            "metadata": {"text": sub_chunk}  # Store full sub-chunk text in metadata
                        }
                    ])
                    
                    # Small delay to prevent rate limiting
                    time.sleep(0.1)
            else:
                # Generate embedding for the chunk
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk
                )
                embedding = response.data[0].embedding
                
                # Generate a safe ASCII ID
                safe_id = generate_safe_id(chunk, prefix=f"{prefix}-{i}")
                
                # Store embedding in Pinecone with the safe ID and metadata
                index.upsert([
                    {
                        "id": safe_id,
                        "values": embedding,
                        "metadata": {"text": chunk}  # Store full chunk text in metadata
                    }
                ])
                
                # Small delay to prevent rate limiting
                time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")
            # Continue with next chunk despite errors
            continue
    
    # Update progress
    st.session_state.current_chunk = end_idx
    progress = min(1.0, end_idx / st.session_state.total_chunks)
    st.session_state.progress_bar.progress(progress)
    
    # Check if all chunks have been processed
    if end_idx >= st.session_state.total_chunks:
        st.session_state.processing_complete = True
        return True
    else:
        # Not done yet, need to process more
        return False

# Create tabs for different functionalities
upload_tab, text_tab, query_tab, excel_tab = st.tabs(["Upload Document", "Enter Text", "Query Data", "Excel Upload"])


# Tab 1: Document Upload
with upload_tab:
    st.header("Upload Fitness Data Document")
    st.write("Upload a document containing fitness data to add to your knowledge base.")
    
    # File upload functionality
    file = st.file_uploader("Upload your fitness data", type=["txt", "csv", "json", "pdf"])
    
    if file:
        st.write(f"File '{file.name}' uploaded successfully!")
        
        # Process button to start processing
        if st.button("Process Document", key="process_file"):
            # Reset session state for new processing
            st.session_state.processing_complete = False
            st.session_state.current_chunk = 0
            st.session_state.total_chunks = 0
            st.session_state.chunks_to_process = []
            
            # Read file content as bytes
            file_content = file.read()
            
            # Extract text based on file type
            if file.name.lower().endswith('.pdf'):
                st.write("Processing PDF file...")
                try:
                    text_content = extract_text_from_pdf(file_content)
                    st.write(f"Successfully extracted text from PDF ({len(text_content)} characters).")
                except Exception as e:
                    st.error(f"Error extracting text from PDF: {str(e)}")
                    text_content = ""
            else:
                # For text files, decode with UTF-8
                try:
                    text_content = file_content.decode('utf-8', errors='replace')
                except Exception as e:
                    st.error(f"Error decoding file: {str(e)}")
                    text_content = ""
            
            if text_content:
                # Split into paragraphs
                paragraphs = split_into_paragraphs(text_content)
                
                st.write(f"Found {len(paragraphs)} paragraphs to process.")
                
                # Start processing in batches
                processing_done = process_chunks(paragraphs, prefix="file")
                
                if processing_done:
                    st.success("All document data has been processed and stored successfully!")
                else:
                    st.info("Processing in progress. Please wait and do not close this page.")
                    st.rerun()
        
        # Continue processing if already started but not complete
        if not st.session_state.processing_complete and st.session_state.total_chunks > 0:
            st.info(f"Processing chunks {st.session_state.current_chunk + 1} to {min(st.session_state.current_chunk + 5, st.session_state.total_chunks)} of {st.session_state.total_chunks}...")
            processing_done = process_chunks(st.session_state.chunks_to_process, prefix="file")
            
            if processing_done:
                st.success("All document data has been processed and stored successfully!")
            else:
                st.rerun()

# Tab 2: Manual Text Input
with text_tab:
    st.header("Enter Fitness Data Manually")
    st.write("Enter fitness data as text to add to your knowledge base.")
    
    # Add text input functionality
    manual_text = st.text_area("Enter fitness data manually (longer text supported):", height=300)
    
    if st.button("Process Text", key="process_manual"):
        if manual_text:
            # Reset session state for new processing
            st.session_state.processing_complete = False
            st.session_state.current_chunk = 0
            st.session_state.total_chunks = 0
            st.session_state.chunks_to_process = []
            
            # Split into paragraphs - same method as file upload
            paragraphs = split_into_paragraphs(manual_text)
            
            st.write(f"Found {len(paragraphs)} paragraphs to process.")
            
            # Start processing in batches
            processing_done = process_chunks(paragraphs, prefix="manual")
            
            if processing_done:
                st.success("All manual text has been processed and stored successfully!")
            else:
                st.info("Processing in progress. Please wait and do not close this page.")
                st.rerun()
        else:
            st.warning("Please enter some text before processing.")
    
    # Continue processing if already started but not complete
    if not st.session_state.processing_complete and st.session_state.total_chunks > 0:
        st.info(f"Processing chunks {st.session_state.current_chunk + 1} to {min(st.session_state.current_chunk + 5, st.session_state.total_chunks)} of {st.session_state.total_chunks}...")
        processing_done = process_chunks(st.session_state.chunks_to_process, prefix="manual")
        
        if processing_done:
            st.success("All manual text has been processed and stored successfully!")
        else:
            st.rerun()

# Tab 3: Query Interface
with query_tab:
    st.header("Ask Questions About Your Fitness Data")
    st.write("Ask questions about your fitness data and get personalized coaching advice.")
    
    # Query interface
    query = st.text_input("Ask a question about your fitness data:")
    
    if st.button("Get Answer", key="get_answer"):
        if query:
            with st.spinner("Generating response..."):
                query = query.replace("\n", " ")
                
                # Generate embedding for the query using OpenAI API
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=query
                )
                
                # Extract the embedding vector from the response
                query_embedding = response.data[0].embedding
                
                # Retrieve relevant chunks from Pinecone based on the query embedding
                results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
                
                # Extract retrieved texts and scores from metadata to construct a refined prompt
                retrieved_chunks = []
                for match in results["matches"]:
                    text = match["metadata"].get("text", "")
                    score = match["score"]
                    retrieved_chunks.append((text, score))
                
                # Sort chunks by relevance (score) and combine them into a single context
                retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)
                retrieved_texts = "\n".join([f"[Score: {score:.2f}] {text}" for text, score in retrieved_chunks])
                
                # Construct a detailed prompt for GPT-4
                prompt = f"""
You are a helpful fitness coach. Based on the following fitness data (ranked by relevance):

{retrieved_texts}

Answer the user's query: {query}
"""
                
                # Generate a coaching response using GPT-4
                chat_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert fitness coach providing detailed advice."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Display the response
                st.write("### Fitness Coach Response:")
                st.write(chat_response.choices[0].message.content)
                
                # Optionally show the retrieved data
                with st.expander("View retrieved data"):
                    st.write("The following data was used to generate your response:")
                    st.write(retrieved_texts)
        else:
            st.warning("Please enter a question to get a response.")

# Tab 4: Excel Upload
with excel_tab:
    excel_processor.excel_tab_content(pc, pineconeRegion, client)

# Add footer
st.markdown("---")
st.caption("Fitness Data Assistant powered by OpenAI and Pinecone")
