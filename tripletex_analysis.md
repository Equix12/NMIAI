# Tripletex — AI Accounting Agent Analysis

## 1. Task Summary

Build an HTTPS endpoint (`/solve`) that receives accounting tasks in natural language (Norwegian, potentially 7 languages), interprets them, and executes them via the Tripletex REST API. Each submission gets a **fresh Tripletex sandbox** — you always start from scratch. Scored on correctness (field-by-field verification) and efficiency.

**You are building an AI agent** that:
1. Receives a Norwegian-language prompt describing an accounting task
2. Optionally processes attached files (PDFs, images with invoices/contracts)
3. Makes the right Tripletex API calls to complete the task
4. Returns `{"status": "completed"}`

## 2. Architecture

```
Competition Server → POST /solve → Your Agent → Tripletex API (via proxy)
                     ↓                ↓
                  prompt +          LLM interprets
                  files +           prompt, decides
                  credentials       API calls, executes
```

### Request Format
```json
{
  "prompt": "Opprett en kunde med navn 'Ola Nordmann'...",
  "files": [
    {"filename": "faktura.pdf", "content_base64": "...", "mime_type": "application/pdf"}
  ],
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "abc123..."
  }
}
```

### Authentication
- **Basic Auth** with username `0` and `session_token` as password
- All API calls go through proxy URL, NOT standard Tripletex URL

### Key API Endpoints
| Endpoint | Methods | Purpose |
|---|---|---|
| `/employee` | GET, POST | Manage employees |
| `/customer` | GET, POST | Manage customers |
| `/product` | GET, POST | Manage products |
| `/invoice` | GET, POST | Create and query invoices |
| `/order` | GET, POST | Manage orders |
| `/travelExpense` | GET, POST, PUT, DELETE | Travel expense reports |
| `/ledger/posting` | GET | Query ledger postings |
| `/ledger/voucher` | GET, POST, DELETE | Manage vouchers |

API docs: https://kkpqfuj-amager.tripletex.dev/v2-docs/

## 3. Scoring Breakdown

### Task Tiers
| Tier | Multiplier | Examples |
|---|---|---|
| Tier 1 | ×1 | Create employee, create customer |
| Tier 2 | ×2 | Create invoice, register payment, credit notes |
| Tier 3 | ×3 | Bank reconciliation from CSV, error correction in ledger, year-end closing |

### Score Range
- **0.0** — Failed
- **Up to 6.0** — Perfect Tier 3 + best efficiency

### Verification
- **Field-by-field checks** against expected values
- **Efficiency bonus** for completing tasks quickly/with fewer API calls
- **Leaderboard** = sum of best scores across all task types
- More task types handled well → higher potential score

### Rate Limits
- Different limits for verified vs unverified teams

## 4. Theoretical Background

### AI Agent Design Patterns
- **ReAct pattern:** Reason about the task, Act (make API call), Observe result, repeat
- **Tool use:** LLM selects which API endpoint to call, with what parameters
- **Planning:** For multi-step tasks, plan the sequence of operations before executing
- **Error recovery:** Parse error messages, adjust parameters, retry

### Norwegian Language Understanding
- Prompts are in Norwegian (bokmål)
- Key accounting terms:
  - Kunde = Customer, Ansatt = Employee, Faktura = Invoice
  - Kreditnota = Credit note, Bilag = Voucher, Reskontro = Ledger
  - Reiseregning = Travel expense, Prosjekt = Project
  - Betaling = Payment, Produkt = Product, Ordre = Order

### PDF/Image Processing
- Some tasks include PDF invoices, contracts, expense reports
- Need to: decode base64 → extract text (OCR if image-based) → parse structured data
- Libraries: PyPDF2/pdfplumber for text PDFs, Tesseract for image-based OCR
- Or: send to multimodal LLM (Claude/Gemini) for direct extraction

### Accounting Basics
- **Double-entry bookkeeping:** Every transaction has debit and credit entries
- **Invoice lifecycle:** Create → Send → Payment received → Reconciled
- **Credit notes:** Reverse/correct a previous invoice
- **Bank reconciliation:** Match bank statement entries to ledger transactions
- **Chart of accounts:** Standard Norwegian account numbering system

## 5. Strategy for Maximum Points

### Priority 1: Cover All Tier 1 Tasks (foundation)
- Map all basic CRUD operations: employee, customer, product
- These are straightforward API calls — should be near-perfect
- Worth 1× multiplier each, but forms the base

### Priority 2: Handle Tier 2 Reliably (multiplier)
- Invoice creation with line items, payment registration
- Multi-step but well-documented
- Worth 2× — good ROI for effort

### Priority 3: Tackle Tier 3 (high value)
- Bank reconciliation, error correction, year-end closing
- Complex multi-step workflows
- Worth 3× — highest potential score
- Requires deep understanding of accounting workflows

### Architecture Recommendations
1. **Use Claude API** as the LLM brain — excellent at Norwegian, tool use, and reasoning
2. **Pre-map common tasks** to API call sequences (template matching)
3. **Implement verification loops** — after creating entities, GET them back to confirm
4. **Robust error handling** — parse Tripletex error messages, retry with corrections
5. **File processing pipeline** — PDF text extraction + LLM interpretation

### Efficiency Optimization
- Minimize unnecessary API calls
- Use field filtering (`?fields=id,name`) to reduce response size
- Batch operations where possible
- Cache API schema knowledge in the agent's context

## 6. Difficulty Assessment

**Rating: 4/10** (Medium-Low) — **This is Claude Code's sweet spot**

### What Claude Code can do exceptionally well:
- **Scaffold the entire endpoint** (FastAPI/Flask + Docker)
- **Write the LLM integration** (Claude API for prompt interpretation)
- **Map prompts to API calls** — this is essentially tool-use, Claude's core strength
- **Handle Norwegian** — Claude is excellent at Norwegian language
- **PDF parsing** — straightforward with libraries or multimodal LLM
- **Error handling logic** — parse and retry patterns
- **Write comprehensive test suites**

### What requires human effort:
- **Deployment** — need an HTTPS endpoint (Google Cloud Run recommended, europe-north1)
- **API exploration** — use sandbox to understand Tripletex API quirks
- **Edge case discovery** — some tasks may have unexpected requirements
- **Authentication setup** — get sandbox account, JWT tokens

### Why this is the easiest task:
- It's literally "build an AI agent with tool use" — the exact thing LLMs are best at
- Well-documented REST API with clear schemas
- Sandbox available for testing
- Clear scoring criteria (field-by-field checks)
- Claude is excellent at Norwegian
- No ML training required — just API orchestration
- Fast iteration cycle (deploy, submit, check score, iterate)

## 7. Step-by-Step Implementation Plan

### Phase 1: Explore & Understand (2-3 hours)
1. Get sandbox account from competition page
2. Explore Tripletex web UI — create employees, customers, invoices manually
3. Read API docs: https://kkpqfuj-amager.tripletex.dev/v2-docs/
4. Test key API endpoints with curl/Python using sandbox credentials
5. Document the API call patterns for each task type

### Phase 2: Build the Agent (3-5 hours)
6. Create FastAPI app with `/solve` endpoint
7. Implement request parsing (prompt, files, credentials)
8. Integrate Claude API for Norwegian prompt interpretation
9. Build tool-use framework: define Tripletex API tools for Claude
10. Implement basic task handlers: create employee, create customer, create product

### Phase 3: Expand Coverage (3-5 hours)
11. Add invoice creation with line items
12. Add payment registration
13. Add travel expense handling
14. Implement PDF/image file processing
15. Add verification loops (GET after POST to confirm)

### Phase 4: Deploy & Test (2-3 hours)
16. Dockerize the application
17. Deploy to Google Cloud Run (europe-north1)
18. Submit endpoint URL at app.ainm.no/submit/tripletex
19. Analyze scoring feedback, fix issues
20. Iterate on failing task types

### Phase 5: Advanced (2-4 hours)
21. Add Tier 3 task handling (bank reconciliation, ledger corrections)
22. Optimize efficiency (fewer API calls)
23. Add robust error recovery with retry logic
24. Tune LLM prompts for edge cases

**Total estimated time: 12-20 hours** for a competitive solution
