import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from datetime import datetime
import time
import base64  # For download transcript feature

# Custom CSS for VidGenie styling
st.markdown(
    """
    <style>
    /* General Styling */
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 12px 16px;
        font-size: 16px;
        color: #333333;
    }
    .stTextInput>div>div>input::placeholder {
        color: #999999;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff4a3d;
    }
    /* Response Cards */
    .response-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .response-card h4 {
        color: #ff6f61;
        font-size: 18px;
        margin-bottom: 12px;
    }
    .response-card p {
        color: #333333;
        font-size: 16px;
        margin: 8px 0;
    }
    .response-card a {
        color: #ff6f61;
        text-decoration: none;
    }
    .response-card a:hover {
        text-decoration: underline;
    }
    /* Chat History Sidebar */
    .sidebar-history {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .sidebar-history h3 {
        color: #ff6f61;
        font-size: 18px;
        margin-bottom: 12px;
    }
    .sidebar-history p {
        color: #666666;
        font-size: 14px;
        margin: 4px 0;
    }
    /* Dark Mode */
    [data-theme="dark"] .stApp {
        background-color: #1e1e1e;
    }
    [data-theme="dark"] .stTextInput>div>div>input {
        background-color: #2d2d2d;
        border-color: #444444;
        color: #ffffff;
    }
    [data-theme="dark"] .stTextInput>div>div>input::placeholder {
        color: #999999;
    }
    [data-theme="dark"] .response-card {
        background-color: #2d2d2d;
        border-color: #444444;
    }
    [data-theme="dark"] .response-card h4 {
        color: #ff6f61;
    }
    [data-theme="dark"] .response-card p {
        color: #ffffff;
    }
    [data-theme="dark"] .sidebar-history {
        background-color: #2d2d2d;
    }
    [data-theme="dark"] .sidebar-history h3 {
        color: #ff6f61;
    }
    [data-theme="dark"] .sidebar-history p {
        color: #cccccc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper function to convert seconds to minutes:seconds format
def seconds_to_min_sec(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02}"


# Initialize the Retriever
class Retriever:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(persist_directory=db_path, is_persistent=True))
        self.collection = self.client.get_collection(name="video_metadata")
        self.embedding_generator = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve the most relevant chunks based on the user's query.
        """
        query_embedding = self.embedding_generator.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Format the results
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "video_uri": results["metadatas"][0][i]["video_uri"],
                "start_time": results["metadatas"][0][i]["start_time"],
                "text": results["documents"][0][i],
            })

        return retrieved_chunks


# Function to create a download link for the transcript
def create_download_link(transcript: str, filename: str = "transcript.txt"):
    b64 = base64.b64encode(transcript.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Transcript</a>'
    return href


# Streamlit App
def main():
    # Title and Header
    st.markdown("<h1 style='text-align: center; color: #ff6f61;'>🎥 VidGenie</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666666;'>Your Smart Video Assistant</h3>", unsafe_allow_html=True)

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for Chat History and Clear Button
    with st.sidebar:
        st.markdown("<h3 style='color: #ff6f61;'>Chat History</h3>", unsafe_allow_html=True)
        for i, entry in enumerate(st.session_state.chat_history):
            st.markdown(
                f"""
                <div class="sidebar-history">
                    <p><strong>{i + 1}. {entry['question']}</strong></p>
                    <p><small>{entry['timestamp']}</small></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    # Search Bar with Magnifying Glass Emoji
    query = st.text_input("🔍 Ask a question...", key="query_input", placeholder="What are the main challenges of deploying ML models?")

    if query:
        # Add the question and timestamp to chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"question": query, "timestamp": timestamp})

        # Display spinner while processing
        with st.spinner("VidGenie is thinking..."):
            time.sleep(1)  # Simulate delay
            retriever = Retriever()
            results = retriever.retrieve(query, top_k=3)

        if results:
            # Display Results
            st.markdown("<h3 style='color: #333333;'>Results</h3>", unsafe_allow_html=True)
            for idx, result in enumerate(results, start=1):
                # Split the text into title, description, and transcript
                text_parts = result["text"].split("\n")
                title = text_parts[0].replace("Title: ", "") if text_parts[0].startswith("Title: ") else ""
                description = text_parts[1].replace("Description: ", "") if len(text_parts) > 1 and text_parts[1].startswith("Description: ") else ""
                transcript = text_parts[2].replace("Transcript: ", "") if len(text_parts) > 2 and text_parts[2].startswith("Transcript: ") else ""

                # Response Card with Response Number
                st.markdown(
                    f"""
                    <div class="response-card">
                        <h4>{idx}</h4>
                        <p><strong>Title:</strong> {title}</p>
                        <p><strong>Description:</strong> {description}</p>
                        <p><strong>Start Time:</strong> {seconds_to_min_sec(result['start_time'])}</p>
                        <p><a href="{result['video_uri']}" target="_blank">Watch Video ↗️</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Feedback Mechanism
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"👍 Helpful {idx}"):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button(f"👎 Not Helpful {idx}"):
                        st.error("We'll try to improve!")
        else:
            st.markdown("<p style='color: #666666;'>No results found.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()