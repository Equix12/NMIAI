# Tripletex AI Accounting Agent

## Quick Start

```bash
cd /home/haava/NMAI/tripletex-agent
source .venv/bin/activate

# Start server
uvicorn main:app --host 0.0.0.0 --port 8080 &

# Start tunnel (serveo — works with Tailscale DNS, unlike cloudflared)
ssh -o StrictHostKeyChecking=no -R 80:localhost:8080 serveo.net
# Grab the https://xxx.serveousercontent.com URL from output

# Submit URL at https://app.ainm.no/submit/tripletex
```

## Architecture

```
Competition Server --> POST /solve --> FastAPI (main.py)
                                         |
                                     solve_task() in agent.py
                                         |
                                     LLM loop (OpenRouter / gpt-5.4-mini)
                                         |
                                     Tool calls --> tripletex_actions.py
                                                        |
                                                    TripletexClient --> Tripletex API (via proxy)
```

### Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI endpoint, logging setup (50MB rotating to `/tmp/tripletex-agent-logs/`) |
| `agent.py` | LLM agent loop, system prompt, 60 tool definitions, ACTION_MAP |
| `tripletex_actions.py` | Deterministic API wrappers (~3000 lines, 60 actions) |
| `tripletex_client.py` | HTTP client with full request/response logging |
| `api_reference.py` | OpenAPI schema helpers |
| `api_schemas.json` | Cached Tripletex schemas |
| `test_scenarios.py` | Test scenarios: `python test_scenarios.py list` / `python test_scenarios.py 0` |
| `test_local.py` | Simple local test runner |

## Environment

```bash
# Required
OPENROUTER_API_KEY=sk-or-...   # in .env, loaded via python-dotenv
```

## LLM

- Provider: OpenRouter
- Model: `openai/gpt-5.4-mini`
- Temperature: 0.1
- Max iterations: 25 per task

## Logging

Logs go to both console (INFO) and file (DEBUG):
- **File**: `/tmp/tripletex-agent-logs/agent.log` (50MB rotation, 5 backups)
- Every API call numbered (`API #1`, `API #2`, ...) with full request/response bodies
- Every tool call logged with full args and results (no truncation)
- Task summary at end: iterations, tool calls, API calls, mutations, errors
- Per-task ID (`[a1b2c3d4]`) for tracing

```bash
# Tail logs
tail -f /tmp/tripletex-agent-logs/agent.log

# Search for errors in a specific task
grep '\[a1b2c3d4\].*ERROR' /tmp/tripletex-agent-logs/agent.log
```

## Tunnel

Tailscale's MagicDNS blocks `trycloudflare.com` resolution, so cloudflared quick tunnels don't work from this machine. Use serveo.net instead:

```bash
# Serveo (works, no binary needed)
ssh -o StrictHostKeyChecking=no -R 80:localhost:8080 serveo.net

# Cloudflared (broken — Tailscale DNS issue)
# ~/cloudflared tunnel --url http://localhost:8080
```

The serveo URL is stable for the session but changes on reconnect. Resubmit at app.ainm.no after reconnect.

## Task Tiers & Scoring

| Tier | Multiplier | Examples |
|------|-----------|----------|
| 1 | x1 | Create employee, customer, product, supplier, department |
| 2 | x2 | Invoice + payment, credit notes, travel expenses, timesheet, projects |
| 3 | x3 | Bank reconciliation from CSV, year-end closing, error correction, payroll |

Score = field-by-field verification + efficiency bonus. Leaderboard = sum of best scores across all task types.

## Key Actions (Tier 3)

- `year_end_closing(date)` — fully deterministic, zeros revenue/expense accounts to equity
- `import_bank_statement(fileContent)` + `create_bank_reconciliation` + `suggest_bank_matches` + `match_bank_transactions` + `close_bank_reconciliation` — full bank recon flow
- `get_account_balances(dateFrom, dateTo, accountNumberFrom, accountNumberTo)` — aggregated balances for error correction
- `run_payroll(employeeId, date, salaryLines)` — handles division, employment, salary types
- `create_supplier_invoice(supplierId, invoiceNumber, ...)` — tries incomingInvoice API, falls back to voucher

## Testing

```bash
source .venv/bin/activate

# List scenarios
python test_scenarios.py list

# Run specific scenario
python test_scenarios.py 0

# Run range
python test_scenarios.py 0-4

# Direct test against sandbox
python test_local.py 0
```

Sandbox: `https://kkpqfuj-amager.tripletex.dev/v2` (token expires March 31, 2026)

## Known Issues

- Tailscale DNS blocks `trycloudflare.com` — use serveo.net tunnel instead
- Supplier invoice `POST /incomingInvoice` returns 403 on some sandbox/proxy configs — falls back to ledger voucher
- Large PDF files may slow down extraction — pdfplumber is used, not OCR

<claude-mem-context>
</claude-mem-context>