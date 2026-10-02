"""The local web search tool: result parsing, request shape, and limits."""

import unittest
from unittest.mock import patch

from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence import websearch

#: A DuckDuckGo-shaped page: a redirect-wrapped result with markup and entities,
#: a direct-URL result with no snippet, and an unrelated link that must be ignored.
PAGE = b"""<html><body>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%2Fstac&amp;rut=abc">STAC &amp; the &lt;spec&gt;</a>
  </h2>
  <a class="result__snippet" href="https://example.com/docs/stac">A <b>catalog</b> spec &#x27;for&#x27; imagery.</a>
  <a class="result__url" href="https://example.com/docs/stac">example.com</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.org/plain">Plain result</a>
</div>
</body></html>"""


class FakeResponse:
    def __init__(self, body: bytes, content_length: int | None = None):
        self._body = body
        self._offset = 0
        length = len(body) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body)
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ParseResultsTests(unittest.TestCase):
    def test_results_carry_decoded_title_url_and_snippet(self):
        results = websearch.parse_results(PAGE.decode("utf-8"))

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0],
            {
                "title": "STAC & the <spec>",
                "url": "https://example.com/docs/stac",
                "snippet": "A catalog spec 'for' imagery.",
            },
        )

    def test_a_redirect_wrapper_is_unwrapped_and_its_absence_tolerated(self):
        results = websearch.parse_results(PAGE.decode("utf-8"))

        self.assertEqual(results[1]["url"], "https://example.org/plain")
        self.assertEqual(results[1]["snippet"], "")

    def test_a_challenge_page_yields_nothing(self):
        self.assertEqual(websearch.parse_results("<html><body>challenge</body></html>"), [])


class WebSearchTests(unittest.TestCase):
    def serve(self, body: bytes, content_length: int | None = None):
        """Patch ``urlopen`` and record the request it was handed."""
        seen = {}

        def open_url(request, timeout=0):
            seen["url"] = request.full_url
            seen["body"] = request.data
            seen["timeout"] = timeout
            return FakeResponse(body, content_length)

        return seen, open_url

    def test_the_query_is_posted_to_the_duckduckgo_html_endpoint(self):
        seen, open_url = self.serve(PAGE)
        with patch("urllib.request.urlopen", open_url):
            results = websearch.web_search("stac spec")

        self.assertEqual(seen["url"], websearch.SEARCH_URL)
        self.assertEqual(seen["body"], b"q=stac+spec")
        self.assertEqual(results[0]["url"], "https://example.com/docs/stac")

    def test_max_results_clamps_and_slices(self):
        seen, open_url = self.serve(PAGE)
        with patch("urllib.request.urlopen", open_url):
            results = websearch.web_search("stac", max_results=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "STAC & the <spec>")

    def test_an_empty_query_is_rejected_before_any_request(self):
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(ToolInputError):
                websearch.web_search("   ")
        urlopen.assert_not_called()

    def test_a_non_integer_max_results_is_rejected(self):
        with self.assertRaises(ToolInputError):
            websearch.web_search("stac", max_results="lots")

    def test_no_parsed_results_is_reported_as_a_search_failure(self):
        _, open_url = self.serve(b"<html><body>no results here</body></html>")
        with patch("urllib.request.urlopen", open_url):
            with self.assertRaises(ToolInputError):
                websearch.web_search("stac")

    def test_an_oversized_response_is_refused(self):
        _, open_url = self.serve(b"x", content_length=websearch.MAX_SEARCH_BYTES + 1)
        with patch("urllib.request.urlopen", open_url):
            with self.assertRaises(ToolInputError):
                websearch.web_search("stac")


if __name__ == "__main__":
    unittest.main()
