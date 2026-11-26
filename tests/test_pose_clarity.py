import unittest

from core import pose_clarity as pc


def kp(visible):
    """17 keypoints; indices in `visible` get confidence 0.9, the rest 0.1."""

    return [[0.0, 0.0, 0.9 if i in visible else 0.1] for i in range(17)]


class PoseClarity(unittest.TestCase):
    def test_usable_body_needs_shoulder_hip_knee(self):
        self.assertTrue(pc.usable_body(kp({5, 12, 14})))
        self.assertFalse(pc.usable_body(kp({5, 6, 11, 12})))  # no knee
        self.assertFalse(pc.usable_body(kp({11, 13})))  # no shoulder
        self.assertEqual(pc.visible_joints(kp({1, 2, 3})), 3)

    def test_height_bucket_edges(self):
        self.assertEqual(pc.height_bucket(39.9), "under40")
        self.assertEqual(pc.height_bucket(40), "40-60")
        self.assertEqual(pc.height_bucket(500), "over120")

    def test_match_is_one_to_one_and_thresholded(self):
        ref = [[0, 0, 10, 10], [20, 20, 30, 30]]
        cand = [[21, 21, 31, 31], [0, 0, 10, 11], [100, 100, 110, 110]]
        self.assertEqual(pc.match_persons(ref, cand), [1, 0])
        self.assertEqual(pc.match_persons([[0, 0, 10, 10]], [[8, 8, 18, 18]]), [None])
        # one candidate cannot serve two references
        self.assertEqual(pc.match_persons([[0, 0, 10, 10], [0, 0, 10, 10]], [[0, 0, 10, 10]]).count(None), 1)

    def test_plan_is_deterministic_and_one_frame_per_video(self):
        def person(h):
            return {"box": [0, 0, 10, h], "keypoints": kp(set())}

        recs = [{"source_video_id": f"v{i % 7}", "source_frame_index": i, "persons": [person(30 + 20 * (i % 4))] if i % 5 else []}
                for i in range(200)]
        a = pc.plan_frames(recs, per_bucket=2, n_empty=2, seed=3)
        self.assertEqual(a, pc.plan_frames(recs, per_bucket=2, n_empty=2, seed=3))
        vids = [p["video"] for p in a]
        self.assertEqual(len(vids), len(set(vids)))

    def test_score_condition_counts_missing_person_as_not_usable(self):
        ref = [{"box": [0, 0, 10, 50]}, {"box": [50, 0, 60, 50]}]
        found = [{"box": [0, 0, 10, 50], "keypoints": kp({5, 11, 13, 1})}]
        rows = pc.score_condition(ref, found)
        self.assertEqual([r["matched"] for r in rows], [True, False])
        self.assertEqual([r["usable"] for r in rows], [True, False])
        self.assertEqual([r["joints"] for r in rows], [4, 0])

    def test_bootstrap_and_verdict(self):
        base = {f"v{i}": [{"usable": False}] for i in range(30)}
        better = {f"v{i}": [{"usable": True}] for i in range(30)}
        s = pc.paired_video_bootstrap(better, base, n_boot=200, seed=1)
        self.assertEqual(s["diff"], 1.0)
        self.assertEqual(pc.verdict(s), "helps")
        same = pc.paired_video_bootstrap(base, base, n_boot=200, seed=1)
        self.assertEqual(pc.verdict(same), "inside the noise")


if __name__ == "__main__":
    unittest.main()
