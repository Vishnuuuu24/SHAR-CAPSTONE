# Pod console logs

`kubectl logs` output for every DGX pod this project has run, captured 2026-09-01
before the completed pod objects were considered for deletion.

**Why this directory exists.** A completed pod's stdout lives only in the pod object.
`kubectl delete pod` destroys it, and none of the run artifact directories ever captured
it — the runners write structured JSON (`run_manifest`, `epoch_metrics`, `aggregate`,
`verdict`) but not the console stream. So everything here was one `kubectl delete` away
from being gone, and it is not reconstructible from the artifacts: staging times,
page-cache recovery figures (K-15), per-phase timing and the ordering of events appear
nowhere else.

| Pod | Experiment | Artifacts |
|---|---|---|
| `shar-baseline011-20260827193456` | `P1C-VIDEO-FRAME-BASELINE-011-AR-FOLD1` | `p1c-video-frame-baseline-011-ar-fold1-s{0,1,2}-*` |
| `shar-screen013-rerun-20260828084833` | `P1C-VIDEO-SCREEN-013-LR3E5-AR-FOLD1` | `p1c-video-screen-013-lr3e5-ar-fold1-s0-20260828T085150Z` |
| `shar-screen014-aug-20260828112653` | `P1C-VIDEO-SCREEN-014-AUGMENTATION-AR-FOLD1` | `p1c-video-screen-014-augmentation-ar-fold1-s0-20260828T113003Z` |
| `shar-confirm015-frozen-20260828145718` | `P1C-VIDEO-FROZEN-CONFIRM-015-AR-FOLD1` (D-42) | `p1c-video-frozen-confirm-015-ar-fold1-s{0,1,2}-*` |
| `shar-featcache-verify2-20260828223317` | B-1 feature-cache verification | `p1c-frozen-feature-cache-ar-fold1/` |

One pod can cover three seed directories: `shar-baseline011` and `shar-confirm015-frozen`
each ran their full registered seed set in a single pod, which is why the mapping is not
one-to-one.

**Note for future runs:** capture the log at the end of the job rather than relying on the
pod object surviving. Pods are also evicted or garbage-collected without warning, so this
is not only about deliberate deletion.
