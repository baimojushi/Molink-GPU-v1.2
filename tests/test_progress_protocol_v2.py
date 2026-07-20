import unittest
from unittest.mock import patch

from worker.progress_protocol import ProgressEmitter


class ProgressProtocolV2Test(unittest.TestCase):
    def test_sequence_and_stage_transition(self):
        job = {
            "job_id": "job_1",
            "order_id": "order_1",
            "pipeline_version": "hanging-main-v1",
            "job_kind": "hang_in_home",
            "eta_features": {"candidate_limit": 3},
        }
        with patch("worker.progress_protocol.time.monotonic", side_effect=[100.0, 101.0, 110.0]):
            emitter = ProgressEmitter(job, {"gpu_model": "RTX 4090"})
            first = emitter.transition("metric3d", "running")
            second = emitter.transition("semantic", "running")

        self.assertEqual(first[0]["seq"], 1)
        self.assertEqual(second[0]["stage"]["state"], "completed")
        self.assertEqual(second[1]["stage"]["id"], "semantic")
        self.assertEqual(second[1]["seq"], 3)
        self.assertEqual(second[1]["input_features"]["candidate_limit"], 3)

    def test_upload_alias_and_heartbeat_snapshot(self):
        emitter = ProgressEmitter({"job_id": "job_2"}, {})
        event = emitter.transition("r2_upload", "running")[0]
        self.assertEqual(event["stage"]["id"], "upload")
        self.assertEqual(emitter.heartbeat_snapshot()["event_id"], event["event_id"])


if __name__ == "__main__":
    unittest.main()
