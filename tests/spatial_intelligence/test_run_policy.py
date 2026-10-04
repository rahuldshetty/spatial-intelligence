"""Run policy: transient-failure classification, backoff, plan snapshots.

These are the decisions the run loop makes between attempts. They used to be
module-private helpers beside a hardcoded replay list; here they are the public
seams `spatial_intelligence.agent.runner` exposes, and the replay decision is
derived from the registry rather than restated.
"""

import errno
import socket
import ssl
import unittest
import urllib.error

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart

from spatial_intelligence.agent.capabilities import (
    TRANSIENT_TOOL_ATTEMPTS,
    NormalizeDuplicateToolNames,
    deduplicate_tool_name,
    is_transient_tool_error,
    tool_failure_text,
)
from spatial_intelligence.agent.runner import (
    MAX_BACKOFF_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    describe_run_error,
    is_context_overflow_error,
    is_transient_run_error,
    latest_plan_items,
    retry_delay,
)
from spatial_intelligence.contracts.errors import ToolInputError, WorkspaceError
from spatial_intelligence.tools.registry import ToolRegistry


class TransientErrorTests(unittest.TestCase):
    def test_provider_and_transport_failures_are_retryable(self):
        self.assertTrue(is_transient_run_error(ModelAPIError("model", "connection reset")))
        self.assertTrue(is_transient_run_error(ModelHTTPError(429, "model")))
        self.assertTrue(is_transient_run_error(ModelHTTPError(408, "model")))
        self.assertTrue(is_transient_run_error(ModelHTTPError(503, "model")))
        self.assertTrue(is_transient_run_error(ModelHTTPError(500, "model")))

    def test_client_errors_and_application_bugs_are_not_retryable(self):
        self.assertFalse(is_transient_run_error(ModelHTTPError(401, "model")))
        self.assertFalse(is_transient_run_error(ModelHTTPError(400, "model")))
        self.assertFalse(is_transient_run_error(UnexpectedModelBehavior("bad tool name")))
        self.assertFalse(is_transient_run_error(ValueError("application failure")))


class ContextOverflowTests(unittest.TestCase):
    """Telling a provider's length refusal apart from any other failure.

    Compaction already targets half the window before every request, so a refusal
    means the window was wrong for this endpoint — worth saying plainly instead of
    reporting the provider's raw text as a failed run.
    """

    def test_the_wordings_providers_use_are_recognised(self):
        messages = (
            "This model's maximum context length is 128000 tokens. However, your "
            "messages resulted in 131067 tokens.",
            "context_length_exceeded",
            "Input is too long for the requested model.",
            "prompt is too long: 210000 tokens > 200000 maximum",
            "too many tokens; please reduce the length of the messages",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertTrue(
                    is_context_overflow_error(ModelHTTPError(400, message))
                )

    def test_an_unrelated_failure_is_not_read_as_an_overflow(self):
        for error in (
            ModelHTTPError(500, "upstream is down"),
            ModelHTTPError(400, "invalid tool schema"),
            ModelAPIError("model", "connection reset by peer"),
            UnexpectedModelBehavior("the model produced no output"),
            ValueError("bad shape"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(is_context_overflow_error(error))


class RetryDelayTests(unittest.TestCase):
    def test_backoff_grows_and_is_capped(self):
        error = ModelAPIError("model", "reset")

        self.assertEqual(retry_delay(error, 1), 0.5)
        self.assertEqual(retry_delay(error, 2), 1.0)
        self.assertEqual(retry_delay(error, 3), 2.0)
        self.assertEqual(retry_delay(error, 20), MAX_BACKOFF_SECONDS)

    def test_a_provider_retry_after_is_honoured_but_bounded(self):
        self.assertEqual(
            retry_delay(ModelHTTPError(429, "model", headers={"Retry-After": "2"}), 5), 2.0
        )
        self.assertEqual(
            retry_delay(ModelHTTPError(429, "model", headers={"Retry-After": "45"}), 1),
            MAX_RETRY_AFTER_SECONDS,
        )


class TransientToolErrorTests(unittest.TestCase):
    """A blip earns a retry; the tool's own answer about the request does not."""

    def test_transport_failures_are_retryable(self):
        for error in (
            urllib.error.URLError("connection reset"),
            urllib.error.HTTPError(
                "https://example.com/a.tif", 503, "unavailable", {}, None
            ),
            ConnectionResetError("reset by peer"),
            ConnectionRefusedError("refused"),
            TimeoutError("timed out"),
            OSError(errno.ECONNRESET, "reset by peer"),
            ssl.SSLError("handshake failure"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(is_transient_tool_error(error))

    def test_a_wrapped_transport_failure_is_retryable(self):
        # Libraries re-raise their own error with the transport failure as the cause.
        wrapper = WorkspaceError("could not read the scene")
        wrapper.__cause__ = ConnectionResetError("reset by peer")

        self.assertTrue(is_transient_tool_error(wrapper))

    def test_the_tools_own_answers_are_not_retryable(self):
        for error in (
            ToolInputError("bounds are outside the Web Mercator range"),
            FileNotFoundError("data/missing.tif"),
            ValueError("no such column"),
            urllib.error.HTTPError(
                "https://example.com/missing.tif", 404, "Not Found", {}, None
            ),
            urllib.error.URLError(socket.gaierror("name does not resolve")),
            ssl.SSLCertVerificationError("certificate verify failed"),
            UnexpectedModelBehavior("Tool 'x' exceeded max retries count of 3"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(is_transient_tool_error(error))


class ToolFailureTextTests(unittest.TestCase):
    def test_an_exhausted_transient_failure_names_the_retries(self):
        text = tool_failure_text(
            urllib.error.URLError("connection reset"), attempts=TRANSIENT_TOOL_ATTEMPTS
        )

        self.assertIn("URLError: <urlopen error connection reset>", text)
        self.assertIn("transient remote failure", text)
        self.assertIn(f"retried {TRANSIENT_TOOL_ATTEMPTS - 1} time(s)", text)

    def test_a_failure_that_was_not_retried_reads_as_plain_detail(self):
        self.assertEqual(
            tool_failure_text(ToolInputError("bad bounds"), attempts=TRANSIENT_TOOL_ATTEMPTS),
            "ToolInputError: bad bounds",
        )
        self.assertEqual(
            tool_failure_text(urllib.error.URLError("reset"), attempts=1),
            "URLError: <urlopen error reset>",
        )


def registry(*names: str) -> ToolRegistry:
    """A registry holding ``names`` as tools implemented elsewhere."""
    built = ToolRegistry()
    for name in names:
        built.add_external(name, category="test", origin="tests")
    return built


class DescribeRunErrorTests(unittest.TestCase):
    """Retry exhaustion is the one failure a tool call can still end a run with."""

    def test_a_repeated_uncallable_call_names_the_tool_and_the_fix(self):
        error = UnexpectedModelBehavior(
            "Tool 'read_file' exceeded max retries count of 3. Consider raising the retry limit, "
            "or see the docs on tool retries: https://pydantic.dev/docs/ai"
        )

        text = describe_run_error(error, registry("read_file"))

        self.assertIn("kept calling 'read_file'", text)
        self.assertIn("search_tools", text)
        self.assertNotIn("pydantic.dev", text)

    def test_a_repeated_unknown_name_says_it_does_not_exist(self):
        error = UnexpectedModelBehavior("Tool 'bogus_tool' exceeded max retries count of 3.")

        text = describe_run_error(error, registry("read_file"))

        self.assertIn("kept calling 'bogus_tool'", text)
        self.assertIn("no tool by that name exists", text)

    def test_any_other_failure_is_reported_verbatim(self):
        error = ModelHTTPError(401, "model", body="invalid api key")

        self.assertEqual(describe_run_error(error, registry()), str(error))


class PlanSnapshotTests(unittest.TestCase):
    def test_the_latest_plan_snapshot_is_restored(self):
        steps = [
            {"type": "plan", "items": [{"id": "old", "content": "Old task"}]},
            {"type": "text", "content": "working"},
            {
                "type": "plan",
                "items": [
                    {"id": "current", "content": "Load selected scenes", "status": "in_progress"}
                ],
            },
        ]

        restored = latest_plan_items(steps)

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].id, "current")
        self.assertEqual(restored[0].status.value, "in_progress")

    def test_no_plan_step_means_an_empty_snapshot(self):
        self.assertEqual(latest_plan_items([{"type": "text", "content": "hi"}]), [])
        self.assertEqual(latest_plan_items([]), [])

    def test_a_malformed_snapshot_does_not_raise(self):
        steps = [{"type": "plan", "items": [{"nonsense": True}, {"id": "ok", "content": "Fine"}]}]

        restored = latest_plan_items(steps)

        self.assertEqual([item.id for item in restored], ["ok"])

    def test_a_non_list_snapshot_restores_nothing(self):
        self.assertEqual(latest_plan_items([{"type": "plan", "items": "nope"}]), [])


class ToolNameNormalizationTests(unittest.TestCase):
    def test_only_a_doubled_name_is_collapsed(self):
        self.assertEqual(deduplicate_tool_name("fit_bounds__fit_bounds"), "fit_bounds")
        self.assertEqual(deduplicate_tool_name("fit_bounds"), "fit_bounds")
        self.assertEqual(deduplicate_tool_name("a__b"), "a__b")
        self.assertEqual(deduplicate_tool_name("__x"), "__x")

    def test_the_capability_rewrites_doubled_names_before_dispatch(self):
        import asyncio

        response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="add_raster__add_raster",
                    args={"path": "a.tif", "name": "A"},
                    tool_call_id="call-1",
                )
            ]
        )

        normalized = asyncio.run(
            NormalizeDuplicateToolNames().after_model_request(
                None, request_context=None, response=response
            )
        )

        self.assertEqual(normalized.parts[0].tool_name, "add_raster")


if __name__ == "__main__":
    unittest.main()
