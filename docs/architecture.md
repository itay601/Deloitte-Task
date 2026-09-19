# Airport Investment Intelligence Agent — Architecture Document

## Overview

The Airport Investment Intelligence Agent is an AI-powered system that helps analysts identify US airports with the highest need for modernization and renovation investment. It combines **deterministic scoring logic** with **LLM-powered conversational Q&A** through an interactive chat interface.

## Architecture

```
User (Browser)
    │
    ▼
┌──────────────────────────────┐
│    Streamlit Chat UI         │  ← Text input, message history
│    (app.py)                  │     Sidebar with data status
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  LangGraph ReAct Agent       │  ← System prompt, conversation memory
│  (agent/agent.py)            │     Mistral LLM (mistral-large-latest)
│                              │
│  Tools:                      │
│  ├─ score_airports()         │  → Rank airports by region
│  ├─ compare_airports()       │  → Side-by-side comparison
│  ├─ get_airport_profile()    │  → Full airport details
│  ├─ get_flight_mix()         │  → Short/medium/long haul %
│  ├─ analyze_demand()         │  → Capacity utilization
│  └─ search_airports_by_region│  → Region → airport lookup
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Scoring Engine              │  ← Pure Python, deterministic
│  (scoring/engine.py)         │     No LLM in scoring logic
│                              │
│  KPIs:                       │
│  ├─ Congestion (25%)         │  Passengers / runway
│  ├─ Growth (20%)             │  YoY passenger growth
│  ├─ Delay (20%)              │  Delay + cancellation metrics
│  ├─ Facility Constraint (20%)│  Runway count, length, surface
│  └─ Demand Gap (15%)         │  Throughput / capacity ratio
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Data Layer                  │  ← Cached pandas DataFrames
│  (data_layer/)               │
│                              │
│  Static Data:                │
│  ├─ airports.csv             │  OurAirports
│  ├─ runways.csv              │  OurAirports
│  ├─ t100_data.csv            │  BTS T-100 Domestic Market
│  └─ ontime_data.csv          │  BTS On-Time Performance
│                              │
│  Optional Live APIs:         │
│  ├─ AviationStack            │  Current flight schedules
│  └─ AeroDataBox              │  Delay statistics
└──────────────────────────────┘
```

## Key Design Decisions

### 1. Separation of Scoring and LLM

The scoring engine is entirely deterministic — it uses percentile ranking across all US commercial airports with no LLM involvement. The LLM (Mistral) is used only for:
- Understanding natural language queries
- Selecting which tools to call
- Interpreting and presenting results conversationally

This separation ensures reproducibility and auditability of investment scores.

### 2. Percentile-Based Scoring

Each KPI is computed as a raw metric, then percentile-ranked (0–100) across all scored airports. This means:
- A score of 90 means the airport is in the 90th percentile for that KPI
- Scores are relative to the entire US airport population
- The composite score is a weighted sum of percentile scores

### 3. Tool-Calling Agent Architecture

We use LangGraph's `create_react_agent` pattern, which:
- Gives the LLM access to 6 structured tools
- Lets the LLM decide which tools to call based on the user's question
- Supports multi-step reasoning (e.g., search region → score airports)
- Maintains conversation memory for follow-up questions

### 4. Data Strategy

- **Static-first**: Core functionality uses pre-downloaded CSV data
- **Graceful degradation**: Live APIs are optional enrichment
- **Synthetic demo data**: T-100 and on-time data are generated with realistic distributions for demo purposes. In production, actual BTS data would be used.

## Scoring Methodology

| KPI | Weight | Metric | Source |
|-----|--------|--------|--------|
| Congestion | 0.25 | Passengers per runway per year | T-100 + runways |
| Growth | 0.20 | Year-over-year passenger change | T-100 multi-year |
| Delay | 0.20 | Composite: avg delay, % delayed >15min, cancellation rate | On-time performance |
| Facility Constraint | 0.20 | Composite: runway count, avg length, surface quality | Runways |
| Demand Gap | 0.15 | Throughput ÷ estimated capacity (25M pax/runway/year) | T-100 + runways |

### Assumptions and Limitations

1. **Capacity estimate**: We use 25M passengers/runway/year as a capacity proxy based on FAA planning guidelines. Actual capacity varies by runway configuration, ATC, and airspace.
2. **Synthetic data**: The demo uses generated data with realistic distributions. Actual BTS data would improve accuracy.
3. **Domestic only**: T-100 domestic market data doesn't capture international traffic, which is significant for hub airports.
4. **Point-in-time**: Scores reflect the most recent year in the dataset. Trend analysis uses 2 years.

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Best AI/data ecosystem |
| LLM | Mistral (mistral-large-latest) | Strong tool-calling, cost-effective |
| Agent | LangChain + LangGraph | Structured tool loops, memory |
| Data | pandas | Standard tabular processing |
| Frontend | Streamlit | Built-in chat, zero build step |
| Charts | plotly | Interactive (when used) |

## Running the Application

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download/generate data
python scripts/download_data.py

# 4. Set API key
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY

# 5. Run
streamlit run app.py
```

## Testing

```bash
pytest tests/ -v
```

Tests verify:
- KPI weight validity
- Individual KPI computations with known inputs
- Scoring determinism (same input → same output)
- Engine end-to-end ranking
- Tool functions (without LLM)
- Region resolution
