# 🧠 LitGraph AI - Research Literature Knowledge Graph

A sophisticated Graph-RAG system that extracts knowledge graphs from research papers and enables intelligent querying through AI-powered reasoning.

![LitGraph AI](https://img.shields.io/badge/LitGraph-AI-blue?style=for-the-badge)
![React](https://img.shields.io/badge/React-19-blue?style=for-the-badge&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?style=for-the-badge&logo=typescript)
![Node.js](https://img.shields.io/badge/Node.js-18-green?style=for-the-badge&logo=node.js)
![Ollama](https://img.shields.io/badge/Ollama-Local-black?style=for-the-badge)

## ✨ Features

- 📄 **PDF Upload & Text Extraction** - Process research papers with automatic text extraction
- 🕸️ **Knowledge Graph Extraction** - AI-powered entity and relationship identification
- 🎨 **Interactive Visualization** - D3.js force-directed graph with zoom, pan, and drag
- 💬 **Graph-Aware Q&A** - Intelligent chat with reasoning traces
- 🎯 **Modern UI** - Dark theme with responsive design
- 🔍 **Entity Types** - Papers, Authors, Concepts, Methods, Results, Datasets, Problems
- 🧠 **Advanced Graph-RAG** - Sub-graph extraction with contextual AI prompting
- 📊 **Vector Embeddings** - Semantic search capabilities for enhanced retrieval
- 🎨 **Dynamic Chat Modal** - Centered interface that preserves sidebar navigation

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Ollama running locally
- `qwen3:4b-instruct` and `nomic-embed-text` installed in Ollama
- npm or yarn

### Installation

```bash
git clone https://github.com/yourusername/litgraph-ai.git
cd litgraph-ai
npm install
```

### Environment Setup

Create `.env` in `server/`:
```env
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_GENERATION_MODEL=qwen3:4b-instruct
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
GRAPHRAG_DENSE_RETRIEVAL_ENABLED=false
PORT=3100
```

Sparse BM25 retrieval is the safe default for an existing Gemini-vector corpus. After backing up the configured database, run `npm run reindex:local --workspace=server` to replace stored vectors with local Ollama embeddings, then set `GRAPHRAG_DENSE_RETRIEVAL_ENABLED=true` to restore hybrid dense+sparse retrieval.

### Integration assurance modes

`POST /api/integration/retrieve` accepts `verification_mode` as `fast`, `standard`, or `thorough`. The interactive modes verify the highest-ranked material claim without a retry; `thorough` verifies up to four claims and permits one evidence retry. Claim-specific evidence excerpts use bounded Ollama requests, and failures or malformed results become `INSUFFICIENT`. Responses include `timings_ms` for retrieval, verification, and total time.

### Run Development Servers

```bash
# Terminal 1 - Backend
cd server && npm run dev

# Terminal 2 - Frontend
cd client && npm run dev
```

Access at http://localhost:5173. The API listens on `http://localhost:3100` by default so it can run beside the SovereignAI frontend on port 3000.

## 📖 Usage

1. **Upload PDF**: Drag and drop research papers or paste text
2. **Explore Graph**: Navigate to interactive knowledge graph
3. **Ask Questions**: Query the graph with AI-powered reasoning

## 🏗️ Architecture

```
Frontend (React) ←→ Backend (Express) ←→ Local AI (Ollama) ←→ Vector DB (Supabase)
     ↓                    ↓                      ↓
PDF Processing      Graph Storage        Knowledge Extraction
Graph Viz          API Routes           Question Answering
Chat Interface     Data Persistence     Reasoning Traces
```

## 📁 Project Structure

```
litgraph-ai/
├── client/                 # React frontend
│   ├── components/        # UI components
│   │   └── GraphCanvas.tsx    # D3.js graph visualization
│   ├── graph/            # Graph-related components
│   │   ├── GraphWorkspace.tsx # Main graph view
│   │   └── NodeInspector.tsx # Node details panel
│   ├── ingestion/        # PDF upload & text input
│   │   └── IngestionPanel.tsx # PDF processing interface
│   ├── chat/            # Chat interface
│   │   └── ChatPanel.tsx     # Q&A interface
│   ├── layout/          # Layout components
│   │   ├── Header.tsx       # App header
│   │   └── Sidebar.tsx      # Navigation sidebar
│   ├── services/        # API services
│   │   ├── graphRagService.ts # Backend API calls
│   │   └── pdfService.ts    # PDF processing
│   ├── types.ts         # TypeScript type definitions
│   ├── App.tsx          # Main application component
│   └── index.tsx        # Application entry point
├── server/               # Express backend
│   ├── server.ts         # Main server file
│   ├── embedding.ts      # Vector embedding generation
│   ├── supabase.ts     # Database integration
│   └── graph_db.json     # Graph storage
├── README.md            # Documentation
└── LICENSE              # MIT License
```

## 🔌 API

### Graph Endpoints

#### `GET /api/graph`
Retrieve current knowledge graph data.

**Response:**
```json
{
  "nodes": [
    {
      "id": "node1",
      "label": "Machine Learning",
      "type": "Concept",
      "description": "A field of AI..."
    }
  ],
  "links": [
    {
      "source": "node1",
      "target": "node2", 
      "relationship": "uses"
    }
  ]
}
```

#### `POST /api/graph/extract`
Extract knowledge graph from text.

**Request:**
```json
{
  "text": "Research paper content..."
}
```

#### `POST /api/graph/query`
Query the knowledge graph with AI reasoning.

**Request:**
```json
{
  "query": "What methods are used?",
  "graph": { /* current graph data */ }
}
```

**Response:**
```json
{
  "text": "The papers use several methods including...",
  "reasoningTrace": ["node1", "node2", "node3"]
}
```

#### `POST /api/graph/clear`
Clear all graph data.

## 🛠️ Tech Stack

**Frontend**: React, TypeScript, Vite, TailwindCSS, D3.js, PDF.js  
**Backend**: Node.js, Express, TypeScript, Supabase
**AI**: Local Ollama models, Graph-RAG, vector embeddings

## 🧠 How It Works

### 1. Text Processing Pipeline
```
PDF/Text Input → Text Extraction → AI Analysis → Graph Generation → Vector Storage
```

1. **PDF Processing**: PDF.js extracts raw text from uploaded files
2. **AI Analysis**: A local Ollama model identifies entities and relationships
3. **Graph Generation**: Structured nodes and links are created
4. **Vector Storage**: Embeddings stored for semantic search

### 2. Knowledge Graph Structure
- **Nodes**: Research entities (papers, authors, concepts, etc.)
- **Links**: Relationships between entities
- **Types**: Categorization for better organization
- **Metadata**: Additional information for context

### 3. Query Processing
```
User Question → Graph Context → AI Reasoning → Answer + Trace
```

1. **Question Analysis**: Understand user intent
2. **Graph Context**: Include relevant graph structure
3. **AI Reasoning**: The local Ollama model processes the retrieved graph evidence
4. **Response Generation**: Answer with reasoning path

## 🎨 UI Components

### IngestionPanel
- **PDF Upload**: Drag-and-drop file interface
- **Text Input**: Large textarea for manual text entry
- **Loading States**: Visual feedback during processing
- **Error Handling**: User-friendly error messages

### GraphCanvas
- **D3.js Visualization**: Force-directed graph layout
- **Interactive Controls**: Zoom, pan, drag functionality
- **Node Styling**: Color-coded by entity type
- **Link Labels**: Relationship type display
- **Event Handling**: Click events for node selection

### ChatPanel
- **Message History**: Conversation display
- **Input Interface**: Question submission
- **Reasoning Traces**: Path visualization showing AI's logical progression
- **Loading Indicators**: Processing feedback
- **Dynamic Modal**: Centered overlay with sidebar preservation

## 🔍 Graph-RAG Pipeline

### Knowledge Extraction
```typescript
// Enhanced entity extraction with detailed prompts
const prompt = `
Extract a comprehensive knowledge graph from following research text.
REQUIREMENTS:
1. Node labels should be COMPLETE and DESCRIPTIVE
2. Include specific entity types: Paper, Author, Concept, Method, Result, Dataset, Problem
3. Add descriptions where helpful for context
4. Create meaningful relationships between entities
`;
```

### Query Processing
```typescript
// Sub-graph extraction for enhanced relevance
const seedNodes = graph.nodes.filter((node: GraphNode) => 
  query.toLowerCase().includes(node.label.toLowerCase())
);

// Contextual prompting with detailed graph structure
const nodeSummary = graph.nodes.map((node: GraphNode) => 
  `ID: ${node.id}, Label: ${node.label}, Type: ${node.type}`
).join('\n');
```

## 🚀 Development

### Running Tests

```bash
# Frontend tests
cd client && npm test

# Backend tests  
cd server && npm test
```

### Building for Production

```bash
# Frontend build
cd client && npm run build

# Backend build
cd server && npm run build
```

## 🐛 Troubleshooting

### Common Issues

#### PDF Upload Not Working
- Check file size (max 10MB)
- Verify file is valid PDF format
- Ensure browser supports File API
- Check console for JavaScript errors

#### Graph Not Displaying
- Verify backend server is running
- Check API endpoints are accessible
- Ensure graph data exists in database
- Look for D3.js console errors

#### AI Responses Not Working
- Verify Ollama is running at `OLLAMA_URL`
- Verify `OLLAMA_GENERATION_MODEL` and `OLLAMA_EMBEDDING_MODEL` are installed
- Run `ollama list` and review the server logs
- Re-ingest content created with another embedding provider

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Code review and merge

### Guidelines
- Follow TypeScript best practices
- Write meaningful commit messages
- Update documentation for changes
- Test thoroughly before submission

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local model inference
- **D3.js** for interactive data visualization
- **PDF.js** for PDF text extraction
- **React & TypeScript** for modern web development
- **TailwindCSS** for utility-first CSS framework

---

**Built with ❤️ for research literature analysis**
