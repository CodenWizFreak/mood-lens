# MoodLens — Conversational Movie Recommender with Machine Unlearning

A full-stack movie recommendation system where you talk to an AI that learns your taste, forgets what you hate, and lets you explore moods without permanently changing your profile.

Powered by **LightGCN graph embeddings**, **GNNDelete** (Tier 1 permanent unlearning), **Influence Functions** (Tier 2 session unlearning), and a **Groq LLM** backend with a live graph visualizer.

---

## Directory Structure

```
project/
├── backend/                    ← Python FastAPI server
│   ├── api.py                  ← Main FastAPI entry point
│   ├── main.py                 ← Legacy CLI entry point
│   ├── llm_client.py           ← Groq client wrapper
│   ├── intent_parser.py        ← LLM-first intent classification
│   ├── scoring_engine.py       ← Hybrid LightGCN + Bayesian scorer
│   ├── state_manager.py        ← Two-tier memory coordinator
│   ├── embedder.py             ← Sentence-transformer plot embeddings
│   ├── graph/
│   │   ├── preference_graph.py ← Permanent taste graph
│   │   ├── session_graph.py    ← Ephemeral mood session graph
│   │   └── graph_builder.py    ← Merges both graphs for the visualizer
│   ├── models/
│   │   ├── lightgcn.py         ← LightGCN implementation
│   │   ├── gnn_delete.py       ← Tier 1: GNNDelete permanent unlearning
│   │   ├── influence.py        ← Tier 2: Influence function session unlearning
│   │   └── train_lightgcn.py   ← Training script
│   ├── requirements.txt
│   ├── .env.example            ← Copy this to .env and fill in your keys
│   └── data/
│       ├── movies_metadata.csv
│       ├── credits.csv
│       └── ratings.csv
│
└── frontend/                   ← Next.js 14 App Router
    ├── app/
    │   ├── chat/page.tsx       ← Main chat UI
    │   └── api/                ← Next.js route handlers (proxy to Python)
    ├── components/
    │   ├── chat/
    │   │   ├── GnnVisualizer.tsx    ← Live memory map graph
    │   │   ├── UnlearningPanel.tsx  ← Memory surgery log
    │   │   ├── EmbeddingDriftChart.tsx
    │   │   └── NewMoodButton.tsx
    │   └── ui/                 ← Glass UI components
    └── .env.local              ← BACKEND_URL=http://localhost:8000
```

---

## Quick Start

### 1. Backend

```bash
cd backend

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set your GROQ_API_KEY (free at console.groq.com)

# Start the server
uvicorn api:app --reload --port 8000
```

First startup takes ~30–60 seconds to load the movie database and build the embedding index. You'll see `[API] Ready ✓` when done.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000/chat

---

## Environment Variables

### Backend `.env`

```env
# Required — get your free key at console.groq.com
GROQ_API_KEY=gsk_...

# Model choice (llama-3.1-8b-instant is the recommended default)
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.8
GROQ_MAX_TOKENS=1024

# Dataset paths (relative to backend/)
METADATA_CSV=data/movies_metadata.csv
CREDITS_CSV=data/credits.csv
RATINGS_CSV=data/ratings.csv

# Embedding backend ("local" is free, uses sentence-transformers)
EMBEDDING_BACKEND=local
EMBEDDINGS_CACHE=embeddings_cache.npy
```

See `.env.example` for the full list including LightGCN paths.

### Frontend `.env.local`

```env
BACKEND_URL=http://localhost:8000
```

---

## How It Works

### Two-Tier Memory System

| Tier | Trigger | What it does |
|------|---------|--------------|
| **Tier 1 — Permanent** | "Block X forever", "never recommend horror" | GNNDelete modifies the LightGCN model weights. That movie/genre is carved out of the embedding space. |
| **Tier 2 — Session** | "New Mood" → Forget / Keep | Influence Functions either roll back or commit the current session signals into the permanent checkpoint. |

### Node Types in the Graph Visualizer

| Node color | Meaning |
|------------|---------|
| 🟢 Green | You (center node) |
| 🟡 Amber | Session movies (temporary mood) |
| 🔵 Blue | Permanent likes (long-term taste) |
| 🟣 Purple | Genre bridge nodes |
| 🔴 Red | Disliked or permanently blocked |

### Chat Flow

1. You type a message
2. Intent parser classifies it (LLM at temp=0, regex fallback)
3. Preference/session graph updates
4. LightGCN + Bayesian scorer ranks movies
5. Groq LLM streams the response token-by-token
6. Graph visualizer updates live

---

## Machine Unlearning — Example Phrases

**Soft dislike (session filter, red node):**
- "I don't like Opening Night"
- "Harold and Maude isn't for me"

**Permanent block (GNNDelete, model weight update):**
- "Block Amadeus"
- "Never recommend horror again"
- "BLOCK COMEDY GENRE"

**Session mood boundary:**
- Click **New Mood** → choose **Forget** (rollback) or **Keep in my Profile** (commit)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Backend health + LightGCN status |
| GET | /state | Full user preference state |
| GET | /graph | Graph viz payload |
| GET | /greet | Opening greeting |
| GET | /session | Current session info |
| GET | /embedding-drift | Drift history for the timeline chart |
| POST | /chat | Send message → SSE stream |
| POST | /new-mood | End session (discard or commit) |
| POST | /reset | Full state reset |

---

## Troubleshooting

**Backend won't start:**
- Make sure `GROQ_API_KEY` is set in `backend/.env`
- Verify the three CSV files exist in `backend/data/`

**"Backend Offline" in the UI:**
- Python server must be running on port 8000
- Frontend must be on port 3000 (CORS is configured for this)

**Slow first startup:**
- Normal — it's building the plot embedding cache (`embeddings_cache.npy`)
- Subsequent starts are fast

**Recommendations not showing:**
- Train LightGCN first: `python models/train_lightgcn.py --quick`
- Without the checkpoint, the system falls back to Bayesian scoring (still works)