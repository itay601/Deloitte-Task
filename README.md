# Deloitte-Task



Airport Investment Intelligence Agent — Implementation Plan
                                                                                                           
 Context                                                                                                 

 This is a 24-hour Deloitte Digital exam project. We need to build an AI-powered agent that helps analysts
  identify promising US airport investment opportunities for modernization/renovation. The agent must
 combine deterministic scoring logic with LLM-powered conversational Q&A, and include a chat interface.

 ---
 Tech Stack

 ┌────────────────┬────────────────────────────────────┬──────────────────────────────────────────────┐
 │     Layer      │               Choice               │                     Why                      │
 ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Language       │ Python 3.11+                       │ Best AI/data ecosystem for rapid prototyping │
 ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────┤
 │ LLM            │ Mistral (mistral-large-latest) via │ User preference; strong tool-calling support │
 │                │  Mistral API                       │                                              │
 ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Agent          │ LangChain (langchain-mistralai) +  │ Structured tool calling, conversation        │
 │ Framework      │ LangGraph                          │ memory, automatic tool loops                 │
 ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Data           │ pandas                             │ Standard for tabular data                    │
 │ Processing     │                                    │                                              │
 ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Frontend       │ Streamlit                          │ Built-in chat components, zero build step,   │
 │                │                                    │ st.audio_input for voice bonus               │
 ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Visualization  │ plotly                             │ Interactive charts inside Streamlit          │
 ├────────────────┼────────────────────────────────────┼──────────────────────────────────────────────┤
 │ Voice (bonus)  │ Mistral or browser-based Web       │ Voice-to-text for chat input                 │
 │                │ Speech API                         │                                              │
 └────────────────┴────────────────────────────────────┴──────────────────────────────────────────────┘

 ---
 Data Strategy

 Static Data (pre-downloaded, stored in data/)

 ┌──────────────────────┬───────────────────────────────┬────────────────────────────────────────────┐
 │       Dataset        │            Source             │                  Purpose                   │
 ├──────────────────────┼───────────────────────────────┼────────────────────────────────────────────┤
 │ airports.csv         │ OurAirports (free, unlimited) │ Airport master list, location, region      │
 │                      │                               │ mapping                                    │
 ├──────────────────────┼───────────────────────────────┼────────────────────────────────────────────┤
 │ runways.csv          │ OurAirports                   │ Runway infrastructure assessment           │
 ├──────────────────────┼───────────────────────────────┼────────────────────────────────────────────┤
 │ T-100 Domestic       │ BTS Transtats (free,          │ Passenger volumes, route distances, demand │
 │ Market               │ unlimited)                    │                                            │
 ├──────────────────────┼───────────────────────────────┼────────────────────────────────────────────┤
 │ On-Time Performance  │ BTS Transtats                 │ Delay and congestion metrics               │
 └──────────────────────┴───────────────────────────────┴────────────────────────────────────────────┘

 Live APIs (optional enrichment, cached)

 ┌───────────────┬────────────────┬──────────────────────────┐
 │      API      │   Free Tier    │           Use            │
 ├───────────────┼────────────────┼──────────────────────────┤
 │ AviationStack │ 100 req/month  │ Current flight schedules │
 ├───────────────┼────────────────┼──────────────────────────┤
 │ AeroDataBox   │ 2400 req/month │ Delay statistics         │
 └───────────────┴────────────────┴──────────────────────────┘

 Fallback: System works fully with static BTS + OurAirports data alone. If API keys are missing or
 rate-limited, it degrades gracefully and tells the user.

 Data approach: Ship a pre-processed sample CSV in the repo for quick start, plus scripts/download_data.py
  to download full datasets. The sample covers enough airports/routes to demo all features without any
 download step.

 ---
 Scoring Methodology (Deterministic — No LLM)

 Five KPIs, each scored 0–100 via percentile rank across all US commercial airports:

 ┌────────────────────┬────────┬──────────────────────────────────────────────────────────────────────┐
 │        KPI         │ Weight │                             How Computed                             │
 ├────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
 │ Congestion         │ 0.25   │ Passengers per runway (from T-100 + runways data)                    │
 ├────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
 │ Growth             │ 0.20   │ Year-over-year passenger growth rate (T-100 multi-year)              │
 ├────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
 │ Delay              │ 0.20   │ Avg departure delay + % flights delayed >15 min + cancellation rate  │
 ├────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
 │ Facility           │ 0.20   │ Runway count, lengths, surface type (fewer/shorter = more            │
 │ Constraint         │        │ constrained)                                                         │
 ├────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
 │ Demand Gap         │ 0.15   │ Throughput vs estimated capacity ratio (capacity = runway-based      │
 │                    │        │ proxy)                                                               │
 └────────────────────┴────────┴──────────────────────────────────────────────────────────────────────┘

 Composite Score = weighted sum. Every result includes raw values, data source, assumptions, and
 confidence level.

 ---
 Architecture

 User (Browser)
     │
     ▼
 [Streamlit Chat UI] ← text input + voice input (bonus)
     │
     ▼
 [LangGraph Agent with Mistral] ← system prompt, conversation memory
     │
     ├── Tool: score_airports(region?, top_n?)      → ScoringEngine
     ├── Tool: compare_airports(iata_a, iata_b)     → side-by-side scores
     ├── Tool: get_airport_profile(iata_code)        → full airport details
     ├── Tool: get_flight_mix(iata_code)             → long-haul % breakdown
     ├── Tool: analyze_demand(iata_code)             → capacity utilization
     └── Tool: search_airports_by_region(region)     → region → IATA codes
     │
     ▼
 [Scoring Engine] ← pure Python, deterministic
     │
     ▼
 [Data Layer] ← pandas DataFrames from CSVs + optional live API calls

 ---
 Project Structure

 airport-investment-agent/
 ├── app.py                      # Streamlit entry point (chat UI)
 ├── requirements.txt
 ├── .env.example
 ├── .gitignore
 ├── agent/
 │   ├── __init__.py
 │   ├── agent.py                # LangGraph agent setup + system prompt
 │   └── tools.py                # @tool decorated functions
 ├── scoring/
 │   ├── __init__.py
 │   ├── engine.py               # ScoringEngine class
 │   ├── kpis.py                 # Individual KPI computation functions
 │   └── config.py               # Weights, thresholds, constants
 ├── data_layer/
 │   ├── __init__.py
 │   ├── loader.py               # Load + cache static CSVs
 │   ├── api_client.py           # AviationStack/AeroDataBox wrappers
 │   ├── regions.py              # "New England" → state codes mapping
 │   └── constants.py            # Region definitions, IATA lookups
 ├── scripts/
 │   └── download_data.py        # Fetch static datasets
 ├── data/                       # Static CSVs (gitignored if large)
 ├── docs/
 │   └── architecture.md         # Design document (deliverable)
 └── tests/
     ├── test_scoring.py
     └── test_tools.py

 ---
 Implementation Order

 Phase 1: Foundation

 1. Create project scaffolding (requirements.txt, .env.example, .gitignore, directory structure)
 2. Build scripts/download_data.py to fetch OurAirports CSVs and prepare BTS sample data
 3. Build data_layer/loader.py with cached DataFrame loading
 4. Build data_layer/regions.py with region-to-state mapping

 Phase 2: Scoring Engine

 5. Implement KPI functions in scoring/kpis.py (congestion, growth, delay, facility, demand gap)
 6. Build ScoringEngine class in scoring/engine.py — orchestrate KPIs, apply weights, return ranked
 results with breakdowns
 7. Write unit tests in tests/test_scoring.py

 Phase 3: Agent + Tools

 8. Implement tool functions in agent/tools.py (6 tools bridging agent to scoring engine)
 9. Set up LangGraph agent in agent/agent.py with system prompt, tool binding, conversation memory

 Phase 4: Chat UI

 10. Build app.py — Streamlit chat with st.chat_input, message history, agent invocation
 11. Add rich output rendering (DataFrames as tables, plotly charts for comparisons)
 12. Add sidebar (data status, about section)

 Phase 5: Polish + Bonus (if time permits)

 13. Voice input via st.audio_input + speech-to-text
 14. Live API integration (AviationStack/AeroDataBox) with rate limiting
 15. Write docs/architecture.md (deliverable)
 16. Error handling and edge cases

 What to skip if time runs out

 - Live API integration (BTS data is sufficient for demo)
 - Voice input
 - Plotly charts (text explanations work)
 - Deployment (localhost is fine)

 What never to skip

 - Deterministic scoring engine with transparent methodology
 - Working chat interface with conversational follow-up
 - Clear communication of assumptions and data sources
 - Architecture documentation

 ---
 Verification

 1. Run scripts/download_data.py and confirm CSVs are downloaded
 2. Run pytest tests/ to verify scoring engine determinism
 3. Run streamlit run app.py and test these queries:
   - "Which airports in New England are strong candidates for terminal expansion?"
   - "Compare LA and Santa Ana airport congestion levels"
   - "What is the percentage of long-haul flights out of Anchorage airport?"
   - "What is the unmet flight demand in SFO airport and why?"
 4. Verify follow-up questions work (conversation memory)
 5. Verify scores include raw values, assumptions, and confidence levels


 All tasks complete. Here's a summary of what was built:                                                  
                                                                                                           
  Project structure:                                                                                       
  airport-investment-agent/                                                                                
  ├── app.py                      # Streamlit chat UI                                                      
  ├── requirements.txt            # All dependencies                                                       
  ├── .env.example                # API key template                                                       
  ├── .gitignore                                                                                           
  ├── agent/                                                                                               
  │   ├── agent.py                # LangGraph ReAct agent + Mistral                                        
  │   └── tools.py                # 6 @tool functions                                                      
  ├── scoring/                                                                                             
  │   ├── config.py               # Weights, thresholds                                                    
  │   ├── kpis.py                 # 5 KPI computation functions
  │   └── engine.py               # ScoringEngine class                                                    
  ├── data_layer/                                                                                          
  │   ├── loader.py               # CSV loading + caching
  │   ├── regions.py              # Region → state mapping
  │   ├── constants.py            # Airport/state constants
  │   └── api_client.py           # Optional live API wrappers
  ├── scripts/
  │   └── download_data.py        # Data download + synthetic generation
  ├── data/                       # CSVs (airports, runways, T-100, on-time)
  ├── docs/
  │   └── architecture.md         # Design document
  └── tests/
      ├── test_scoring.py         # 23 tests — all passing
      └── test_tools.py           # 12 tests — all passing

  To run:
  cd /mnt/c/Myprojects/Deloitte-Task
  source venv/bin/activate
  cp .env.example .env   # Add your MISTRAL_API_KEY
  streamlit run app.py

  All 35 tests pass. The scoring engine is fully deterministic, the agent uses Mistral via LangGraph with
  conversation memory, and the Streamlit UI provides a chat interface with a data status sidebar.