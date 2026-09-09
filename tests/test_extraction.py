import json

from agent import extraction_agent


def test_strands_agent_returns_evidence_and_workflow_recommendation(monkeypatch):
	expected = {
		"text_evidence": {"partial_fields": {"household_size": 2}},
		"document_evidence": {"partial_fields": {"annual_income": 24000}},
		"contradictions": [],
		"workflow_recommendation": {
			"workflow": "needs_followup",
			"missing_fields": ["annual_income"],
		},
	}

	class FakeAgent:
		def __call__(self, prompt):
			assert "Use the registered tools autonomously" in prompt
			return json.dumps(expected)

	monkeypatch.setattr(
		extraction_agent,
		"build_extraction_agent",
		lambda model=None: FakeAgent(),
	)

	text_output, document_output, contradictions, trace = extraction_agent._orchestrate_extraction(
		text="Household of two; income is documented separately.",
		document_path_or_text="income_verification.txt",
		model=object(),
	)

	assert text_output == expected["text_evidence"]
	assert document_output == expected["document_evidence"]
	assert contradictions == []
	assert trace[-1] == "Strands recommended needs_followup"
