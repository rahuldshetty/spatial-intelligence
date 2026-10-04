"""Model residency: what loads, what is kept, and what is given back.

The catalog ships one model, so these tests patch in a second entry: eviction
is only observable when something else is competing for the memory.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_intelligence.ai import catalog, store
from spatial_intelligence.ai.manager import (
    EMBEDDING_CACHE_SIZE,
    MANAGER_SERVICE,
    ModelManager,
    ModelSession,
    get_manager,
)
from spatial_intelligence.contracts.errors import ToolInputError

MODEL = "slimsam-77"
TWIN = "slimsam-77-twin"


class ManagerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(
            "os.environ",
            {"GEOAI_MODELS_DIR": str(Path(self._tmp.name) / "models")},
            clear=False,
        )
        self._env.start()
        self.twin = dataclasses.replace(
            catalog.SLIMSAM, id=TWIN, repo="example/twin", revision="rev2", files=()
        )
        self._catalog = patch.object(catalog, "MODELS", (catalog.SLIMSAM, self.twin))
        self._catalog.start()
        self.loads: list[str] = []
        self._downloaded = patch.object(store, "is_downloaded", lambda spec: True)
        self._downloaded.start()
        self._loader = patch.object(ModelManager, "_load", self._fake_load)
        self._loader.start()

    def tearDown(self):
        self._loader.stop()
        self._downloaded.stop()
        self._catalog.stop()
        self._env.stop()
        self._tmp.cleanup()

    def _fake_load(self, spec):
        """Stand in for building ONNX sessions, with a delay two threads can race."""
        self.loads.append(spec.id)
        time.sleep(0.05)
        now = time.monotonic()
        return ModelSession(
            spec=spec,
            encoder=object(),
            decoder=object(),
            provider="CPUExecutionProvider",
            threads=1,
            loaded_at=now,
            last_used=now,
        )


class ResidencyTests(ManagerTestCase):
    def test_a_model_loads_once_and_is_reused(self):
        manager = ModelManager()

        with manager.reserve(MODEL) as first:
            pass
        with manager.reserve(MODEL) as second:
            pass

        self.assertIs(first, second)
        self.assertEqual(self.loads, [MODEL])

    def test_a_second_model_evicts_the_idle_first(self):
        manager = ModelManager(cache_size=1)

        with manager.reserve(MODEL):
            pass
        with manager.reserve(TWIN):
            pass

        self.assertEqual(manager.loaded_ids(), [TWIN])
        self.assertEqual(self.loads, [MODEL, TWIN])

    def test_a_model_in_use_survives_the_cap(self):
        manager = ModelManager(cache_size=1)

        with manager.reserve(MODEL):
            with manager.reserve(TWIN):
                self.assertEqual(sorted(manager.loaded_ids()), sorted([MODEL, TWIN]))

    def test_an_idle_model_past_the_ttl_is_dropped(self):
        manager = ModelManager(cache_size=4, ttl_seconds=0.0)

        with manager.reserve(MODEL):
            pass
        with manager.reserve(TWIN):
            pass

        self.assertEqual(manager.loaded_ids(), [TWIN])

    def test_embeddings_are_cached_within_a_cap(self):
        session = self._fake_load(catalog.find(MODEL))
        for index in range(EMBEDDING_CACHE_SIZE + 2):
            session.cache_embedding(f"tile-{index}", index)

        self.assertEqual(len(session.embeddings), EMBEDDING_CACHE_SIZE)
        self.assertIsNone(session.cached_embedding("tile-0"))
        self.assertEqual(
            session.cached_embedding(f"tile-{EMBEDDING_CACHE_SIZE + 1}"),
            EMBEDDING_CACHE_SIZE + 1,
        )


class UnloadTests(ManagerTestCase):
    def test_unload_frees_one_model(self):
        manager = ModelManager()
        with manager.reserve(MODEL):
            pass

        result = manager.unload(MODEL)

        self.assertEqual(result["unloaded"], [MODEL])
        self.assertEqual(manager.loaded_ids(), [])

    def test_unload_refuses_a_model_in_use(self):
        manager = ModelManager()

        with manager.reserve(MODEL):
            with self.assertRaises(ToolInputError):
                manager.unload(MODEL)

    def test_unload_all_leaves_a_model_in_use_alone(self):
        manager = ModelManager(cache_size=4)
        with manager.reserve(TWIN):
            pass

        with manager.reserve(MODEL):
            result = manager.unload()

            self.assertEqual(result["unloaded"], [TWIN])
            self.assertEqual(manager.loaded_ids(), [MODEL])

    def test_close_clears_everything(self):
        manager = ModelManager()
        with manager.reserve(MODEL):
            pass

        manager.close()

        self.assertEqual(manager.loaded_ids(), [])


class StatusTests(ManagerTestCase):
    def test_status_merges_the_catalog_row_with_residency(self):
        manager = ModelManager()

        before = manager.status()[0]
        self.assertEqual(before["id"], MODEL)
        self.assertEqual(before["task"], "segmentation")
        self.assertFalse(before["loaded"])

        with manager.reserve(MODEL):
            during = manager.status()[0]
            self.assertTrue(during["loaded"])
            self.assertTrue(during["in_use"])
            self.assertEqual(during["provider"], "CPUExecutionProvider")
            self.assertEqual(during["threads"], 1)

        after = manager.status()[0]
        self.assertTrue(after["loaded"])
        self.assertFalse(after["in_use"])
        self.assertIsNotNone(after["idle_seconds"])

    def test_status_can_filter_by_task(self):
        manager = ModelManager()

        self.assertEqual(
            [row["id"] for row in manager.status(task="segmentation")], [MODEL, TWIN]
        )
        self.assertEqual(manager.status(task="detection"), [])


class FailureTests(ManagerTestCase):
    def test_an_undownloaded_model_is_named_before_loading(self):
        manager = ModelManager()
        with patch.object(store, "is_downloaded", lambda spec: False):
            with self.assertRaises(ToolInputError) as caught:
                with manager.reserve(MODEL, download=False):
                    pass

        self.assertIn("ai_pull_model", str(caught.exception))
        self.assertEqual(self.loads, [])

    def test_an_unknown_model_is_reported(self):
        manager = ModelManager()

        with self.assertRaises(ToolInputError):
            with manager.reserve("no-such-model"):
                pass

    def test_a_missing_onnxruntime_explains_the_extra(self):
        manager = ModelManager()
        self._loader.stop()  # this one needs the real loader
        try:
            with patch.dict(sys.modules, {"onnxruntime": None}):
                with self.assertRaises(ToolInputError) as caught:
                    with manager.reserve(MODEL):
                        pass
        finally:
            self._loader.start()

        self.assertIn("spatial-intelligence[ai]", str(caught.exception))


class ConcurrencyTests(ManagerTestCase):
    def test_two_callers_share_one_load(self):
        manager = ModelManager()
        ready = threading.Barrier(3)
        held: list[str] = []

        def worker():
            ready.wait()
            with manager.reserve(MODEL):
                held.append(MODEL)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        ready.wait()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(len(held), 2)
        self.assertEqual(self.loads, [MODEL])

    def test_the_manager_is_the_sessions_singleton(self):
        class Bag:
            def __init__(self):
                self.services = {}

            def service(self, key, factory):
                return self.services.setdefault(key, factory())

        runtime = Bag()

        self.assertIs(get_manager(runtime), get_manager(runtime))
        self.assertIn(MANAGER_SERVICE, runtime.services)


if __name__ == "__main__":
    unittest.main()
