"""Test the agent locally against the Tripletex sandbox."""

import json
import os
import sys

from agent import solve_task

TX_BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjMyNDc2LCJ0b2tlbiI6Ijc5YWRlZWRkLTZiOWItNDljNi04Yzc5LWE3NDRlNzI2OTkyNSJ9"

# Test tasks simulating competition prompts
TEST_TASKS = [
    {
        "name": "Tier 1: Create customer",
        "prompt": "Opprett en kunde med navn 'Nordvik Consulting AS', e-post 'post@nordvik.no' og telefonnummer '98765432'.",
        "files": [],
    },
    {
        "name": "Tier 1: Create employee",
        "prompt": "Registrer en ny ansatt med fornavn 'Kari', etternavn 'Hansen', e-post 'kari.hansen@firma.no'.",
        "files": [],
    },
    {
        "name": "Tier 2: Create invoice",
        "prompt": "Opprett en faktura til kunde 'Berg Elektro AS' med følgende linjer:\n- 5 stk 'Konsulenttime' à kr 1200\n- 2 stk 'Reisekostnad' à kr 500\nFakturadato: 2026-03-19, forfallsdato: 2026-04-19.",
        "files": [],
    },
    {
        "name": "Tier 1: Create product",
        "prompt": "Create a product called 'Premium Support Package' with a price of 2500 NOK excluding VAT.",
        "files": [],
    },
    {
        "name": "Tier 1: Create project",
        "prompt": "Opprett et prosjekt med navn 'Nettside Redesign' for kunde 'Nordvik Consulting AS'. Startdato: 2026-04-01.",
        "files": [],
    },
]


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    task_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if task_index >= len(TEST_TASKS):
        print(f"Task index {task_index} out of range (0-{len(TEST_TASKS)-1})")
        sys.exit(1)

    task = TEST_TASKS[task_index]
    print(f"\n{'='*60}")
    print(f"Testing: {task['name']}")
    print(f"Prompt: {task['prompt']}")
    print(f"{'='*60}\n")

    result = solve_task(
        prompt=task["prompt"],
        files=task["files"],
        base_url=TX_BASE_URL,
        session_token=TX_TOKEN,
        openrouter_api_key=api_key,
    )

    print(f"\n{'='*60}")
    print(f"Result: {json.dumps(result, indent=2)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
