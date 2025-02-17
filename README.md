# VidGenie: Your Smart Video Assistant

VidGenie is an intelligent video processing system designed to help you analyze, search, and understand video content through automated metadata extraction and semantic search capabilities. 

## What is RAG?

RAG stands for Retrieval Augmented Generation.

Each phase can be approximately divided into: 

  1. Retrieval - Act of searching for relevant information from a source based on a specific query. For instance, retrieving relevant snippets of Wikipediacontent from            database in response to a certain question.
        
  2. Augmented - Process of utilizing relevant returned information to alter an input for a generative model, such as an LLM.
        
  3. Generation - Process of producing an output based on a particular input. For instance, while considering an LLM, the task involves creating a written piece based            on a specified prompt.
  

● The main goal of RAG is to improve the generation outputs of LLMs.



## Workflow of RAG

Now let’s understand the basic workflow of a RAG based LLM architecture in three steps:

1. Ingestion
2. Synthesis
3. Retrieval

<img src="https://github.com/Ananya0104/Basic-RAG-Implementation/blob/main/rag.jpeg">

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
    streamlit run main_app.py
    ```

2. **In the Web Interface:**
    - Ask any questions (related to thre 5 videos given below)
    - Click enter
    - You'll see the ouput with the Video URI, timestamps, description etc.
    - View matched video segments with timestamps

### Example Videos:
- [Video 1](https://www.youtube.com/watch?v=ftDsSB3F5kg)
- [Video 2](https://www.youtube.com/watch?v=kKFrbhZGNNI)
- [Video 3](https://www.youtube.com/watch?v=6qUxwZcTXHY)
- [Video 4](https://www.youtube.com/watch?v=MspNdsh0QcM)
- [Video 5](https://www.youtube.com/watch?v=Kf57KGwKa0w)


## How It Works

### Preprocessing Pipeline
- **Video Download**: Videos are downloaded using `yt-dlp`.
- **Audio Extraction & Transcription**: Audio is extracted and transcribed using the Whisper model to generate text from video.
- **Visual Analysis**: Frames of the video are analyzed with BLIP image captioning to generate descriptions.
- **Metadata Storage**: Extracted metadata (transcriptions, captions) is stored in a ChromaDB vector database for easy retrieval.

### Retrieval System
- **Semantic Search**: Query text is processed with Sentence Transformers to retrieve relevant answers.
- **Timestamp-based Matching**: Results are matched with timestamps for quick access to specific video sections.
- **Multi-modal Search**: Users can search using both text and visual context.

### Streamlit Interface
- **Interactive UI**: Easily input any question.
- **Real-time Processing**: Track the status of video processing in real-time.
- **Search Results**: View results of search queries with corresponding timestamps. 

