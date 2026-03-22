"""Test scenarios based on real competition tasks we've encountered."""

import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from agent import solve_task

TX_BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjMyNDc2LCJ0b2tlbiI6Ijc5YWRlZWRkLTZiOWItNDljNi04Yzc5LWE3NDRlNzI2OTkyNSJ9"

SCENARIOS = [
    # === TIER 1: Simple creation tasks ===
    {
        "name": "T1: Create customer (English)",
        "prompt": "Create the customer Clearwater Ltd with organization number 898695476. The address is Torggata 146, 4006 Stavanger. Email: post@clearwater.no.",
        "files": [],
    },
    {
        "name": "T1: Create employee (Portuguese)",
        "prompt": "Temos um novo funcionário chamado André Almeida, nascido em 30. May 1992. Crie-o como funcionário com o e-mail andre.almeida@example.org e data de início 4. February 2026.",
        "files": [],
    },
    {
        "name": "T1: Create employee as admin (Norwegian)",
        "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
        "files": [],
    },
    {
        "name": "T1: Create product (English)",
        "prompt": "Create a product called 'Premium Support Package' with a price of 2500 NOK excluding VAT.",
        "files": [],
    },
    {
        "name": "T1: Create department (Norwegian)",
        "prompt": "Opprett en avdeling med navn 'Salgsavdeling' og avdelingsnummer 3.",
        "files": [],
    },

    # === TIER 1/2: Invoice tasks ===
    {
        "name": "T2: Create invoice (German)",
        "prompt": "Erstellen und senden Sie eine Rechnung an den Kunden Waldstein GmbH (Org.-Nr. 925346519) über 25100 NOK ohne MwSt. Die Rechnung betrifft Systementwicklung.",
        "files": [],
    },
    {
        "name": "T2: Create invoice with lines (Norwegian)",
        "prompt": "Opprett en faktura til kunde 'Berg Elektro AS' med følgende linjer:\n- 5 stk 'Konsulenttime' à kr 1200\n- 2 stk 'Reisekostnad' à kr 500\nFakturadato: 2026-03-19, forfallsdato: 2026-04-19.",
        "files": [],
    },

    # === TIER 2: Travel expenses ===
    {
        "name": "T2: Travel expense (German)",
        "prompt": "Erfassen Sie eine Reisekostenabrechnung für Johanna Hoffmann (johanna.hoffmann@example.org) für \"Kundenbesuch Bergen\". Die Reise dauerte 3 Tage mit Tagegeld (Tagessatz 800 NOK). Auslagen: Flugticket 3500 NOK, Taxi 450 NOK.",
        "files": [],
    },

    # === TIER 2: Timesheet ===
    {
        "name": "T2: Register timesheet (English)",
        "prompt": "Log 29 hours for Ella Williams (ella.williams@example.org) on the activity \"Testing\" in the project \"Platform Integration\" for Windmill Ltd (org no. 839360274). Hourly rate: 1350 NOK/h. Generate a proforma invoice.",
        "files": [],
    },

    # === TIER 2: Reverse/Credit ===
    {
        "name": "T2: Reverse payment (Spanish)",
        "prompt": "El pago de Dorada SL (org. nº 849807021) por la factura \"Mantenimiento\" (8900 NOK sin IVA) fue devuelto por el banco. Revierta el pago para que la factura vuelva a mostrar el importe pendiente.",
        "files": [],
    },

    # === TIER 1: Supplier ===
    {
        "name": "T1: Create supplier (Norwegian)",
        "prompt": "Opprett en leverandør med navn 'Kontorrekvisita AS', org.nr 987654321, e-post bestilling@kontorrekvisita.no. Adresse: Industriveien 5, 3050 Mjøndalen.",
        "files": [],
    },

    # === TIER 1: Contact ===
    {
        "name": "T1: Create contact (French)",
        "prompt": "Créez un contact pour le client Montagne SA : prénom Jean, nom Dupont, e-mail jean.dupont@montagne.fr, téléphone portable +33612345678.",
        "files": [],
    },

    # === TIER 2: Project ===
    {
        "name": "T2: Create project (Norwegian)",
        "prompt": "Opprett et prosjekt med navn 'Nettside Redesign' for kunde 'Nordvik Consulting AS'. Startdato: 2026-04-01.",
        "files": [],
    },
]


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            for i, s in enumerate(SCENARIOS):
                print(f"  {i:>2}: {s['name']}")
            return

        indices = []
        for arg in sys.argv[1:]:
            if "-" in arg:
                start, end = arg.split("-")
                indices.extend(range(int(start), int(end) + 1))
            else:
                indices.append(int(arg))
    else:
        indices = [0]

    for idx in indices:
        if idx >= len(SCENARIOS):
            print(f"Scenario {idx} out of range (0-{len(SCENARIOS)-1})")
            continue

        scenario = SCENARIOS[idx]
        print(f"\n{'='*70}")
        print(f"Scenario {idx}: {scenario['name']}")
        print(f"Prompt: {scenario['prompt'][:100]}...")
        print(f"{'='*70}\n")

        try:
            result = solve_task(
                prompt=scenario["prompt"],
                files=scenario["files"],
                base_url=TX_BASE_URL,
                session_token=TX_TOKEN,
                openrouter_api_key=api_key,
            )
            print(f"\n{'='*70}")
            print(f"Result: {json.dumps(result, indent=2)}")
            print(f"{'='*70}")
        except Exception as e:
            print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
