# 📄 Paper to Voice Assistant 🎙️

> Transform research papers into engaging podcast conversations using AI-powered analysis and natural voice synthesis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io)

## 🌟 Overview

Paper to Voice Assistant is an innovative AI-powered application that revolutionizes how we consume academic research. It automatically converts dense research papers into engaging, conversational podcast-style audio content featuring natural dialogue between a host (Jane) and an expert guest (Dr. Sharma).

### ✨ What Makes It Special?

- **🧠 AI-Powered Analysis**: Uses Google's Gemini AI to understand and extract key insights from research papers
- **🎭 Natural Conversations**: Creates realistic dialogue between podcast characters with distinct voices
- **🔊 High-Quality Audio**: Leverages MeloTTS for natural-sounding voice synthesis
- **🎵 Professional Production**: Adds background music for a polished podcast experience
- **📊 Visual Processing**: Converts PDF pages to images for comprehensive AI analysis

## 🚀 Key Features

### 📖 Intelligent Paper Analysis
- Processes multi-page research PDFs
- Extracts key research steps and methodologies
- Identifies important findings and contributions
- Structures content for conversational presentation

### 🎤 Podcast Generation
- **Two-Character Format**: Host (Jane) interviews expert (Dr. Sharma)
- **Natural Speech Patterns**: Includes fillers, interruptions, and authentic conversation flow
- **Configurable Tones**: Choose between Formal and Conversational styles
- **Multi-language Support**: Currently supports English with room for expansion

### 🔊 Audio Production
- **Dual Voice Synthesis**: Different accents for host (EN-US) and guest (EN-India)
- **Background Audio**: Ambient guitar tracks for professional atmosphere
- **Audio Consolidation**: Combines multiple voice tracks into single podcast file
- **Retry Logic**: Robust error handling for voice generation

## 🏗️ Architecture

### Project Structure
```
Paper-to-voice-assistant/
├── 📁 src/paper_to_voice/           # Core package
│   ├── 📁 core/                     # Configuration & models
│   │   ├── config.py               # API keys, model settings
│   │   └── models.py               # TypedDict data structures
│   ├── 📁 utils/                   # Utility functions
│   │   └── pdf_processor.py        # PDF → Images conversion
│   ├── 📁 workflow/                # AI workflow components
│   │   ├── steps.py                # Research analysis
│   │   ├── dialog.py               # Conversation generation
│   │   └── orchestrator.py         # LangGraph workflow
│   └── 📁 audio/                   # Audio processing
│       ├── tts.py                  # Text-to-speech
│       └── processor.py            # Audio consolidation
├── 📁 config/                      # Configuration files
├── 📁 tests/                       # Testing framework
├── 📁 examples/                    # Sample papers
├── main.py                         # Streamlit app
└── app.py                          # Legacy monolithic version
```

### 🔄 Workflow Pipeline

1. **📄 PDF Upload & Processing**
   - User uploads research paper PDF
   - System converts each page to high-resolution images
   - Images encoded as base64 for AI consumption

2. **🧠 AI Analysis Phase**
   ```python
   PDF → Images → AI Analysis → Research Steps → Q&A Pairs
   ```
   - Gemini AI analyzes paper structure and content
   - Extracts research methodology and key findings
   - Generates question-answer pairs for each section

3. **🎭 Dialog Generation**
   - AI creates natural conversation between host and expert
   - Incorporates research insights into engaging dialogue
   - Maintains academic accuracy while ensuring accessibility

4. **🔊 Voice Synthesis**
   - Text-to-speech conversion using MeloTTS
   - Different voice characteristics for each speaker
   - Background music integration

5. **🎵 Audio Production**
   - Combines voice tracks with ambient audio
   - Exports final podcast as MP3 format
   - Ready for streaming or download

## 🛠️ Technology Stack

### Core AI Components
- **Google Gemini AI**: Research paper analysis and content extraction
- **LangChain**: AI workflow orchestration and prompt management
- **LangGraph**: State-based workflow execution
- **MeloTTS**: High-quality text-to-speech synthesis

### Application Framework
- **Streamlit**: Interactive web interface
- **Python 3.8+**: Core programming language
- **pypdfium2**: PDF processing and image extraction
- **pydub**: Audio manipulation and processing

### Data Processing
- **Base64 Encoding**: Image processing for AI consumption
- **JSON Parsing**: Structured data extraction
- **TypedDict**: Type-safe data structures

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher**
- **Google AI API Key** (for Gemini access)
- **Stable internet connection** (for TTS services)
- **Sufficient disk space** for temporary audio files

## 🚀 Quick Start

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/priyanshiranawat15/Paper-to-voice-assistant.git
cd Paper-to-voice-assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy environment template
cp config/.env.example .env

# Edit .env with your API keys
# Required:
GOOGLE_API_KEY=your_google_gemini_api_key_here
GOOGLE_MODEL_NAME=gemini-1.5-flash
```

### 3. Launch Application
```bash
# Start the Streamlit app
streamlit run main.py
```

### 4. Generate Your First Podcast
1. **Upload PDF**: Use the file uploader to select your research paper
2. **Configure Settings**: Choose podcast tone from the sidebar
3. **Process**: Watch the progress bar as AI analyzes your paper
4. **Listen**: Play the generated podcast directly in your browser

## 🎯 Usage Guide

### Supported Paper Types
- ✅ Academic research papers
- ✅ Technical reports
- ✅ Conference proceedings
- ✅ Journal articles
- ✅ Multi-page scientific documents

### Best Practices
- **PDF Quality**: Use high-resolution PDFs for better analysis
- **File Size**: Optimal performance with papers under 20 pages
- **Content Structure**: Papers with clear sections work best
- **Language**: Currently optimized for English-language papers

## ⚙️ Configuration Options

### Model Settings
```python
# In src/paper_to_voice/core/config.py
GOOGLE_MODEL_NAME = "gemini-1.5-flash"  # AI model for analysis
TTS_MODEL = "myshell-ai/MeloTTS-English"  # Voice synthesis model
```

### Audio Settings
```python
# Voice characteristics
LIGHT_GUITAR_FREQ = 440  # Background music frequency
AMBIENT_GUITAR_FREQ = 220  # Ambient sound frequency
GUITAR_DURATION = 1000  # Background audio duration (ms)
```

### Podcast Characters
- **Jane (Host)**: EN-US accent, interviewer role
- **Dr. Sharma (Expert)**: EN-India accent, subject matter expert

## 🧪 Testing

```bash
# Run basic functionality tests
python -m pytest tests/

# Run specific test file
python tests/test_basic.py

# Test with coverage
pytest --cov=src tests/
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### API Key Errors
```bash
ValueError: GOOGLE_API_KEY environment variable is not set
```
**Solution**: Ensure your `.env` file contains valid Google AI API key

#### Voice Generation Failures
```
Could not generate voice for part: Connection timeout
```
**Solution**: Check internet connection and HuggingFace service status

#### Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: Process smaller PDF files or reduce image resolution

#### Audio Processing Errors
```
Could not process audio file: Invalid format
```
**Solution**: Ensure proper audio codecs are installed

### Debug Mode
Enable detailed logging by setting:
```python
# In main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Fork the repository
git fork https://github.com/priyanshiranawat15/Paper-to-voice-assistant

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements.txt
pip install pytest black isort flake8

# Make your changes and test
pytest tests/
black src/
isort src/
```

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Use descriptive commit messages

## 📊 Performance Metrics

### Processing Times (Average)
- **5-page paper**: ~2-3 minutes
- **10-page paper**: ~4-6 minutes  
- **20-page paper**: ~8-12 minutes

## 🛡️ Reliability, Validation, and Concurrency Architecture

### 1. Pydantic Schema Validation
The research plan generated by Gemini is validated through structured Pydantic v2 schemas:
- `SubStepItem`: Ensures non-empty `key` and `value` strings for each substep query.
- `ResearchStep`: Enforces a non-empty `step` description and at least one substep.
- `ResearchPlan`: Enforces a non-empty list of validated research steps.
- **Validation Pipeline**: Markdown code blocks (` ```json ... ``` `) are automatically cleaned and parsed with `json.loads`. On malformed JSON or `ValidationError`, a bounded corrective model recovery is triggered before failing explicitly.

### 2. Exponential Backoff with Randomized Full Jitter
API calls to Gemini and MeloTTS are wrapped with configurable exponential backoff and randomized jitter:
$$\text{delay} = \text{random.uniform}(0, \min(\text{max\_delay}, \text{base\_delay} \times 2^{\text{attempt}}))$$
- **Transient Error Classification**: Automatically retries HTTP 429, 500, 502, 503, 504, `ResourceExhausted`, rate limits, and network connection timeouts.
- **Permanent Errors**: Immediately rejects non-retryable errors (HTTP 400, 401, 403, `ValidationError`, `ValueError`) without wasting attempts.
- **Header Support**: Respects `Retry-After` headers where available.
- **Sanitized Logging**: Redacts API keys and sensitive tokens from structured log outputs.

### 3. LangGraph Send-Based Fan-Out Stages
- **Stage 1 (Research-step Fan-out)**: `continue_to_substeps()` validates the research plan and creates one isolated `solve_substeps` task per step using LangGraph's native `Send` API. Each task receives image context and a stable `step_index`.
- **Stage 2 (Dialogue Fan-out)**: `continue_to_substeps_voice()` pairs steps and solutions using deterministic indices. Detects step/solution count mismatches and prevents silently dropped results.
- **Deterministic Aggregation**: Task results are tagged with stable indices and re-sorted to preserve original document order regardless of asynchronous completion order.

### 4. Distributed Task Processing with Celery & Redis
When distributed processing is desired (`USE_CELERY=true`), independent tasks in the two fan-out stages execute in separate worker processes via Celery and Redis:
- **Broker & Backend**: Redis (`redis://redis:6379/0` in Docker or `redis://localhost:6379/0` locally).
- **Dedicated Task Queues**:
  - `research_steps`: Consumed by workers for solving research substeps (`solve_substeps_task`).
  - `dialog_tasks`: Consumed by workers for dialogue script generation (`generate_dialog_task`).
- **Clean LangGraph Integration**: LangGraph's Send-based fan-out dispatches tasks with stable IDs to the Celery queue and awaits their results with configurable timeouts (`CELERY_TASK_TIMEOUT=120`), ensuring full state consistency before moving to downstream nodes.
- **Task Lifecycle & Status Tracking**: The `TaskTracker` maintains live status across `PENDING`, `RUNNING`, `SUCCEEDED`, and `FAILED` states.
- **Idempotency & Redelivery Safety**: Tasks use deterministic IDs (e.g., `workflow_{id}_step_{index}` and `workflow_{id}_dialog_{index}`). If redelivered due to network disconnects or worker crashes (`task_acks_late=True`), the worker returns the cached result without repeating costly LLM calls.
- **Storage Safety**: Redis is strictly used as a task broker and metadata store. Heavy binary files (PDFs, raw audio chunks) are never stored in Redis.

### 5. Running with Docker Compose (Local Multi-Process Cluster)
The repository includes a complete multi-container Docker Compose configuration featuring Redis, two independent Celery worker processes, and the orchestrator web app:

```bash
# 1. Provide your Google API key
export GOOGLE_API_KEY="your-api-key-here"

# 2. Build and start Redis, two Celery workers, and the Streamlit orchestrator
docker compose up --build

# 3. Access the Streamlit application
open http://localhost:8501
```

To run Celery workers manually on a local host:
```bash
# Start Redis (if installed locally)
redis-server

# Run Celery Worker 1
celery -A src.paper_to_voice.tasks.celery_app worker -Q research_steps,dialog_tasks -c 2 -l info -n worker1@%h

# Run Celery Worker 2
celery -A src.paper_to_voice.tasks.celery_app worker -Q research_steps,dialog_tasks -c 2 -l info -n worker2@%h
```

### 6. Audio Correctness & Verification
- **Speaker Voices**: Preserves designated speaker profiles (`**Jane:**` $\to$ EN-US, `**Dr. Sharma:**` $\to$ EN-India).
- **Format Verification**: Consolidated audio is normalized and exported to 44,100 Hz stereo (channels=2). The exported MP3 metadata is inspected via `verify_audio_metadata` to guarantee delivery standards.
- **Safe Cleanup**: Temporary audio chunks are safely managed and cleaned up to prevent orphaned files.

### 7. Running Tests
Run the automated test suite with:
```bash
pytest -v tests/
```

## 🔮 Roadmap

### Upcoming Features
- [ ] **Multi-language Support**: Support for non-English papers
- [ ] **Custom Voice Cloning**: Upload your own voice samples
- [ ] **Batch Processing**: Process multiple papers simultaneously
- [ ] **Export Options**: Download scripts, subtitles, and transcripts
- [ ] **Integration APIs**: Connect with podcast platforms

### Long-term Vision
- [ ] **Real-time Processing**: Live paper-to-podcast conversion
- [ ] **Interactive Podcasts**: Q&A with AI about the paper
- [ ] **Visual Enhancements**: Generate accompanying slide decks
- [ ] **Community Features**: Share and discover generated podcasts


## 🙏 Acknowledgments

### Core Technologies
- **Google AI**: For powerful Gemini language models
- **Meta AI**: For MeloTTS voice synthesis technology
- **Streamlit Team**: For the amazing web app framework
- **LangChain**: For AI workflow orchestration tools

### Inspiration
This project was inspired by the need to make academic research more accessible and engaging for broader audiences.

---

<div align="center">

**Made with ❤️ for researchers, students, and curious minds everywhere**

*Transform knowledge into conversations, one paper at a time*

</div>
