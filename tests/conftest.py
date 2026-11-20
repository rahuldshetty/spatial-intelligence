"""Keep the test suite independent of provider credentials.

The shipped default model is a hosted one, so a test that opens a workspace
would otherwise try to construct its provider and fail on a machine without
``OPENAI_API_KEY``.  Pydantic AI's built-in ``test`` model is the intended
substitute; individual tests still override ``GEOAI_MODEL`` when they exercise
environment or settings behavior.
"""

import os

# Force rather than setdefault: a developer's shell may already export the real
# model, and the suite must not depend on that machine state.
os.environ["GEOAI_MODEL"] = "test"
