"""AID NAVIGATOR — DC LIHEAP eligibility triage agent.

Entry point: runs a batch of applicant intake cases (some text-only, some
with a supporting document) through the full pipeline — extraction,
schema validation, next-best-action, cross-source contradiction checking,
deterministic eligibility/crisis rules — and prints the resulting cases
in caseworker triage order: most urgent/important first.

Run with:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

from agent.extraction_agent import run_extraction_pipeline
from triage.ranking import rank_cases


PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Demo case set
# ---------------------------------------------------------------------------
# Five applicants covering every branch the system handles:
#   Maria   - complete, crisis-eligible (text only)
#   Robert  - urgent but missing required fields (text only)
#   Angela  - complete, regular-eligible, not crisis (text only)
#   James   - complete, ineligible on income (text only)
#   Denise  - complete, but text/document income disagree beyond
#             tolerance -> flagged for caseworker review, not auto-decided
#
# All figures are synthetic. No real applicant data is used anywhere in
# this project.

DEMO_CASES = [
    {
        "name": "Maria Lopez",
        "text": (
            "My name is Maria Lopez. I have a household of 3 people. "
            "My annual income is $42,000. I live in a single family home "
            "and use natural gas. My heat is included in rent. I am "
            "disconnected and owe $480 in arrears. My fuel tank is 8%. "
            "I need help by Friday."
        ),
        "document": None,
    },
    {
        "name": "Robert",
        "text": (
            "This is Robert. Things have been rough since I got put on "
            "reduced hours — my paycheck last month was $2,100 but that's "
            "not steady. There's four of us at home including my mother. "
            "We're in a rowhouse, been renting the same unit for six years. "
            "The gas company sent a shutoff notice, not sure exactly when "
            "it takes effect but soon. We owe something like $600, maybe "
            "more with fees. Honestly I just need this handled whenever you "
            "can get to it."
        ),
        "document": None,
    },
    {
        "name": "Angela",
        "text": (
            "This is Angela. Household of 4. Annual income $35,000. "
            "Single-family home, gas heat. Not disconnected, everything's "
            "current, but the utility bills have been tight."
        ),
        "document": None,
    },
    {
        "name": "James",
        "text": (
            "This is James. Household of 2, my wife and me. Combined "
            "income is $150,000 a year. We're in a single-family home, "
            "gas heat. No issues with the utility, just checking what "
            "programs might be available for next year."
        ),
        "document": None,
    },
    {
        "name": "Denise Carter",
        "text": (
            "Hi, this is Denise. There's just me and my two kids here — "
            "household of 3. We're in an apartment, electric heat. I make "
            "about $28,000 a year. The electric company already cut us off "
            "last week — no power since Tuesday. I owe them $340."
        ),
        "document": str(PROJECT_ROOT / "income_verification.txt"),
    },
]


STATUS_LABELS = {
    "ready_for_eligibility": "DECIDED",
    "needs_review": "NEEDS CASEWORKER REVIEW (contradiction)",
    "needs_followup": "NEEDS FOLLOW-UP (missing information)",
}


def _print_case(rank: int, case: dict) -> None:
    triage = case["triage"]
    status = case["status"]
    print(f"\n{rank}. [{STATUS_LABELS.get(status, status)}]  tier={triage['tier']}  "
          f"confidence={triage['confidence']:.2f}")
    print(f"   reason: {triage['reason']}")

    if status == "ready_for_eligibility":
        eligibility = triage["eligibility"]
        regular = eligibility["regular"]
        crisis = eligibility["crisis"]
        print(f"   regular eligible: {regular['eligible']}  "
              f"(benefit: ${regular['benefit_amount']:.2f})")
        print(f"   crisis eligible:  {crisis['eligible']}")
    elif status == "needs_review":
        contradictions = case.get("contradictions", [])
        for c in contradictions:
            print(f"   contradiction: {c['field']} — text={c['text_value']!r} "
                  f"vs document={c['document_value']!r}")
    elif status == "needs_followup":
        for q in case["next_action"]["followup_questions"]:
            print(f"   ask: {q}")


def main() -> None:
    print("AID NAVIGATOR — running demo case batch through the pipeline...\n")

    results = []
    for demo_case in DEMO_CASES:
        pipeline_result = run_extraction_pipeline(
            text=demo_case["text"],
            document_path_or_text=demo_case["document"],
        )
        results.append(pipeline_result)

    ranked = rank_cases(results)

    print("=" * 70)
    print("CASEWORKER TRIAGE QUEUE (highest priority first)")
    print("=" * 70)

    for i, case in enumerate(ranked, start=1):
        _print_case(i, case)

    print("\n" + "=" * 70)
    print(f"{len(ranked)} cases processed.")


if __name__ == "__main__":
    main()