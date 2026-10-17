"""Per-request usage rows for the trace modal."""

from __future__ import annotations

import unittest

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.usage import RequestUsage

from spatial_intelligence.agent.runner import request_usage


class RequestUsageTests(unittest.TestCase):
    def test_a_model_response_becomes_one_usage_row(self):
        response = ModelResponse(
            parts=[
                TextPart("checking"),
                ToolCallPart("list_files", {}, tool_call_id="t1"),
            ],
            usage=RequestUsage(
                input_tokens=120,
                output_tokens=15,
                cache_read_tokens=80,
                cost=0.0012,
            ),
            model_name="openai:test",
        )

        rows = request_usage([response])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_tokens"], 120)
        self.assertEqual(rows[0]["output_tokens"], 15)
        self.assertEqual(rows[0]["cache_read_tokens"], 80)
        self.assertEqual(rows[0]["requests"], 1)
        self.assertAlmostEqual(rows[0]["cost"], 0.0012)
        self.assertEqual(rows[0]["model"], "openai:test")
        self.assertEqual(rows[0]["tools"], ["list_files"])
        self.assertIsInstance(rows[0]["timestamp"], str)

    def test_non_response_messages_are_ignored(self):
        self.assertEqual(request_usage([]), [])


if __name__ == "__main__":
    unittest.main()
