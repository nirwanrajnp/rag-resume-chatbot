# Resume Chatbot

A self-hosted AI-powered resume chatbot that enables natural language queries about professional background. Built with FastAPI, Ollama, and ChromaDB for efficient local deployment.

## Quick Start

> **Note**: Detailed hosting and deployment instructions will be available soon. Currently in development and testing phase. (Not Ready for Production Use)

### Local Development Setup

#### Prerequisites
- Python 3.10+
- Git

#### Basic Setup

1. **Clone Repository**
```bash
git clone https://github.com/nirwanrajnp/rag-resume-chatbot
cd resume-chatbot
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model
ollama pull deepseek-r1:1.5b
```

4. **Prepare Your Resume**
```bash
# Copy your resume PDF to data directory
cp /path/to/your/resume.pdf data/

# Step 1: Parse resume with universal parser
python backend/universal_parser.py data/your-resume.pdf

# Step 2: Ingest parsed resume into vector database
python backend/universal_ingest.py data/your-resume.pdf
```

5. **Start Services**
```bash
# Local development
cd backend && python main.py
```

6. **Test the API**
```bash
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL_NAME` | `deepseek-r1:1.5b` | LLM model to use |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB storage path |

## API Endpoints

### Health Check
```bash
GET /health
```
Returns service status and component health.

### Chat
```bash
POST /chat
Content-Type: application/json

{
  "question": "What is your experience with React?"
}
```

Response:
```json
{
  "answer": "I have extensive experience with React...",
  "sources": ["resume", "structured_info"]
}
```

## Deployment & Hosting

> **Coming Soon**: Comprehensive hosting guides, cost analysis, and scaling strategies will be available after thorough testing and validation.

This section will include:
- VPS deployment guides
- Cost optimization strategies
- Performance tuning recommendations
- Scaling options for different use cases