# VidGenie: Your Smart Video Assistant

VidGenie is an intelligent video processing system designed to help you analyze, search, and understand video content through automated metadata extraction and semantic search capabilities. 

## Features
- **YouTube Video Downloading & Metadata Extraction**: Download and extract video metadata from YouTube URLs.
- **Automatic Speech-to-Text Transcription**: Transcribe video audio to text in 100+ languages using the Whisper model.
- **Visual Analysis**: Perform frame-by-frame captioning using BLIP image captioning to describe video content.
- **Semantic Search**: Search for specific video segments based on semantic queries.
- **Streamlit Web Interface**: User-friendly web interface for easy interaction with the system.

## Technologies Used
- **Core**: Python 3.9+
- **Video Processing**: yt-dlp, FFmpeg
- **AI Models**: Whisper (OpenAI), BLIP (Salesforce)
- **Database**: ChromaDB
- **NLP**: Sentence Transformers
- **Web Interface**: Streamlit

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/vidgenie.git
    cd vidgenie
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install FFmpeg**
    - **For Ubuntu/Debian:**
        ```bash
        sudo apt-get install ffmpeg
        ```
    - **For MacOS:**
        ```bash
        brew install ffmpeg
        ```

## 🚀 Quick Start with Streamlit

1. **Launch the Streamlit Interface**
    ```bash
    streamlit run app.py
    ```

2. **In the Web Interface:**
    - Ask any questions (related to thre 5 videos given below)
    - Click enter
    - You'll see the ouput with the Video URI, timestamps, description etc.
    - View matched video segments with timestamps


## How It Works

### Preprocessing Pipeline
- **Video Download**: Videos are downloaded using `yt-dlp`.
- **Audio Extraction & Transcription**: Audio is extracted and transcribed using the Whisper model to generate text from video.
- **Visual Analysis**: Frames of the video are analyzed with BLIP image captioning to generate descriptions.
- **Metadata Storage**: Extracted metadata (transcriptions, captions) is stored in a ChromaDB vector database for easy retrieval.

### Retrieval System
- **Semantic Search**: Query text is processed with Sentence Transformers to retrieve relevant video segments.
- **Timestamp-based Matching**: Results are matched with timestamps for quick access to specific video sections.
- **Multi-modal Search**: Users can search using both text and visual context.

### Streamlit Interface
- **Interactive UI**: Easily input video URLs and manage video processing.
- **Real-time Processing**: Track the status of video processing in real-time.
- **Search Results**: View results of search queries with corresponding timestamps. 

