"""Unit and integration tests.  Run:  python -m unittest discover tests"""

import tempfile
import unittest
from itertools import islice

import numpy as np

from streampulse.config import DEVICES
from streampulse.feature_store import FeatureStore
from streampulse.ingest import DeadLetterQueue, MicroBatcher, QueueSource, validate
from streampulse.monitoring import PageHinkley, PSIMonitor
from streampulse.pipeline import StreamPulsePipeline
from streampulse.registry import ModelRegistry
from streampulse.simulator import StreamingPlatformSimulator
from streampulse.tasks.recommender import RecommenderTask


def small_sim(hours=2, **kw):
    return StreamingPlatformSimulator(hours=hours, traffic=0.6, **kw)


class TestSimulator(unittest.TestCase):
    def test_reproducible(self):
        a = list(islice(small_sim(seed=3), 500))
        b = list(islice(small_sim(seed=3), 500))
        self.assertEqual(a, b)

    def test_session_lifecycle_is_consistent(self):
        started, ended = set(), set()
        for ev in small_sim(hours=3, malformed_rate=0):
            if ev["type"] == "session_start":
                started.add(ev["session_id"])
            elif ev["type"] == "heartbeat":
                self.assertIn(ev["session_id"], started)
                self.assertNotIn(ev["session_id"], ended)
            else:
                ended.add(ev["session_id"])
        self.assertTrue(ended and ended <= started)


class TestIngest(unittest.TestCase):
    def test_validation_catches_bad_events(self):
        good = next(e for e in small_sim(malformed_rate=0) if e["type"] == "heartbeat")
        self.assertIsNone(validate(good))
        self.assertEqual(validate({**good, "throughput_mbps": -1}), "range:throughput_mbps")
        self.assertEqual(validate({**good, "cdn": "nope"}), "value:cdn")
        bad = dict(good)
        bad.pop("bitrate_mbps")
        self.assertEqual(validate(bad), "missing:bitrate_mbps")

    def test_microbatches_are_ordered_30s_windows(self):
        dlq = DeadLetterQueue()
        batches = list(MicroBatcher(dlq).batches(QueueSource(small_sim(hours=1))))
        idx = [b.window_index for b in batches]
        self.assertEqual(idx, sorted(idx))
        self.assertGreater(len(batches), 100)
        self.assertGreater(sum(b.size for b in batches), 1000)


class TestFeatureStore(unittest.TestCase):
    def test_features_are_finite_and_shaped(self):
        fs = FeatureStore()
        n = 0
        for ev in small_sim(malformed_rate=0):
            if ev["type"] == "session_start":
                fs.on_start(ev)
            elif ev["type"] == "heartbeat":
                s = fs.on_heartbeat(ev)
                x = fs.abandonment_features(ev, s)
                self.assertEqual(len(x), len(FeatureStore.ABANDON_FEATURES))
                self.assertTrue(np.isfinite(x).all())
                f = fs.forecast_features(ev, s)
                self.assertEqual(len(f), len(FeatureStore.FORECAST_FEATURES))
                n += 1
            else:
                fs.on_end(ev)
        self.assertGreater(n, 1000)
        self.assertEqual(sum(1 for d in DEVICES if f"device_{d}" in FeatureStore.ABANDON_FEATURES), len(DEVICES))


class TestMonitoring(unittest.TestCase):
    def test_page_hinkley_detects_shift(self):
        rng = np.random.default_rng(0)
        ph = PageHinkley(threshold=5, min_instances=100)
        values = np.r_[rng.normal(0.1, 0.02, 500), rng.normal(0.2, 0.02, 300)]
        hits = [i for i, v in enumerate(values) if ph.update(v)]
        self.assertTrue(hits and hits[0] >= 500, hits)

    def test_psi_zero_for_same_distribution_and_large_for_shift(self):
        rng = np.random.default_rng(1)
        m = PSIMonitor(["a"], reference_size=5000, window=5000)
        m.update(rng.normal(0, 1, (5000, 1)))
        m.update(rng.normal(0, 1, (5000, 1)))
        self.assertLess(m.compute()["a"], 0.05)
        m.update(rng.normal(1.5, 1, (5000, 1)))
        self.assertGreater(m.compute()["a"], 0.25)


class TestRecommender(unittest.TestCase):
    def test_learns_a_viewer_preference(self):
        rec = RecommenderTask(n_viewers=5, n_titles=50)
        for _ in range(40):
            rec.on_session_start({"viewer_id": 0, "title_id": 7, "genre": "sports"})
        self.assertIn(7, rec.recommend(0, "mf").tolist())
        self.assertIn(7, rec.recommend(0, "content").tolist())


class TestRegistry(unittest.TestCase):
    def test_publish_and_load_latest(self):
        with tempfile.TemporaryDirectory() as d:
            reg = ModelRegistry(d)
            self.assertEqual(reg.publish("m", {"w": 1}, {"auc": 0.7}), "v0001")
            self.assertEqual(reg.publish("m", {"w": 2}, {"auc": 0.8}), "v0002")
            model, meta = reg.load("m")
            self.assertEqual((model["w"], meta["version"]), (2, "v0002"))
            self.assertEqual(reg.load("m", "v0001")[0]["w"], 1)


class TestPipeline(unittest.TestCase):
    def test_end_to_end_short_run(self):
        with tempfile.TemporaryDirectory() as d:
            p = StreamPulsePipeline(registry_dir=d, publish_every_hours=2, verbose=False)
            res = p.run(QueueSource(small_sim(hours=6)))
            self.assertEqual(len(res.hourly), 5)
            s = res.summary
            self.assertGreater(s["throughput"]["events"], 10_000)
            self.assertGreater(s["abandonment"]["abandon_online_roc_auc"], 0.6)
            self.assertGreater(s["recommender"]["rec_hit@10_content"], s["recommender"]["rec_hit@10_random"])
            self.assertTrue(ModelRegistry(d).versions("abandonment"))


if __name__ == "__main__":
    unittest.main()
