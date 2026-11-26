# Confusion-guided divide-and-conquer: E0-E2 report

Status (2026-10-19): **E0-E2 are run-complete (CPU, on stored probabilities). E3-E5 (training specialists) are code-complete and unit-tested but NOT run** - both DGX workers were NotReady (unreachable taint) when the pod was launched; the pod stayed Pending and was deleted at the owner's request. Nothing below about specialist accuracy is measured. Validation only; sealed test untouched. Nothing pushed.

## A. Baseline (E0)

- Source: `results/p1c/p1c-late-fusion/dinov2_probs.csv` (sha256 `00da0e81...5211`), frozen DINOv2 ViT-B/14-reg + `step_permuted` attention head, 4 AR folds x 3 seeds, 56 validation videos/fold (4/class).
- Recomputed macro-F1 **0.46261** = reference 0.46261 (`reproduces_reference: true`). Per fold 0.5188 / 0.4364 / 0.4705 / 0.4247.
- Macro-F1 = unweighted mean over all 14 classes; a class with no true positives scores 0.
- Protocol caveats that bind everything below: (1) 56 videos/fold means one video moves a class F1 a lot; (2) validation sets overlap across folds (e.g. folds 2&3 share 38, 3&4 share 40), so pooled out-of-fold counts double-count videos; (3) the gate (+0.04 on each of 4 folds) is a project gate, not a significance test.
- `--repro-check` in the specialist script retrains the general model for the first (fold, seed) and compares to the CSV; not yet run.

## B. Confusion analysis (E1)

- Pooled over 672 run-rows: 353 errors; **94.9% anomaly->anomaly**, 2.8% to Normal, 2.3% from Normal. Normal is not where the loss is.
- Per-class F1: Normal 0.81, Explosion 0.64, Arson 0.63, Shoplifting 0.59, Vandalism 0.59, Stealing 0.53, Arrest 0.51, RoadAccidents 0.51, Robbery 0.50, Burglary 0.33, Abuse 0.30, Fighting 0.29, Assault 0.21, Shooting 0.02.
- Top recurring pairs: Assault->Burglary 13 (per-fold 3/5/2/3), RoadAccidents->Shooting 11, Robbery->Assault 11, Shooting->Shoplifting 11, Abuse->Fighting 10. Fold consistency is mixed: most pairs are driven by 1-2 folds.
- Group selection (rule fixed before looking at ceilings): anomaly classes only, sizes 2-4, closure >= 0.5, internal confusion >= 6, best coverage per size.
  - **Leave-one-fold-out (LOFO)**: for each scored fold, groups come only from the other folds' confusion minus videos shared with the scored fold. Size-4 groups: f1 Abuse/Assault/RoadAccidents/Shooting; f2 Abuse/Fighting/RoadAccidents/Stealing; f3 Explosion/Fighting/RoadAccidents/Shooting; f4 Abuse/Assault/Fighting/Robbery. **They differ on every fold - itself a sign the confusion structure is partly noise at this sample size.**
  - Global (pooled, biased) size-4: Assault/Burglary/Fighting/Robbery. Reported only as a biased upper bound.
  - Manual groups (violent_trio, violent_quad, fight_assault, property) were written knowing the pooled confusion: **model-selection bias**, documented here, not hidden.

## C. Oracle ceilings (E2)

Perfect within-group specialist, delta in macro-F1 over baseline. `oracle_route` (uses the true class to pick rows) is trivially large and uninformative; `hard_route` (specialist only replaces rows the baseline already predicts in G, truth in G) is the realistic ceiling.

| Group | hard_route mean | per fold | folds >= +0.04 | decision |
|---|---|---|---|---|
| violent_trio (Shooting, Assault, Fighting) | +0.026 | .006/.030/.039/.031 | 0 | rejected |
| fight_assault | +0.005 | .006/.005/.000/.009 | 0 | rejected |
| property (Burglary, Robbery, Shoplifting, Stealing) | +0.037 | .029/.024/.034/.063 | 1 | rejected |
| **violent_quad** (+Abuse) | +0.069 | .056/.069/.072/.081 | 4 | kept (manual, biased) |
| LOFO size 2 (folds 2,3 only) | +0.013 | .004/.023 | 0 | rejected |
| LOFO size 3 | +0.014 | .006/.000/.023/.029 | 0 | rejected |
| **LOFO size 4** | +0.056 | .054/.057/.056/.058 | 4 | kept (leak-free) |
| global size 4 (BIASED) | +0.083 | .058/.105/.076/.094 | 4 | reference only |

Reading: the gate is reachable only with 4-class groups, and only if the specialist is close to perfect. The kept groups' margin over +0.04 is 0.016-0.03 of macro-F1. Earlier project evidence put within-group accuracy near 0.60 at best for 3 classes (chance 0.33), far from "nearly perfect".

## D. Implementation (code-complete, tested)

`core/confusion_groups.py`, `core/specialist.py`, `core/specialist_eval.py`, `scripts/run_confusion_analysis.py`, `scripts/run_specialist_experiment.py`, `configs/divide_and_conquer/specialists.json`, `tests/test_specialist.py` (8), `tests/test_specialist_eval.py` (3). Design: same frozen features and head with |G| outputs; "holdout" mode (25% of group training videos monitor the epoch; leak-free) vs "valmon" sensitivity (baseline convention); strategies `hard_route`, `fusion_w0.5` (w fixed before any run, not tuned), and `oracle_route_trained` diagnostic; guard of >= 20 training videos/class; strong-class regression reported for Normal, Explosion, Robbery, Arson. Hierarchy (E6) not justified by the ceilings and not built.

## E. Experiment results

E0, E1, E2: above. **E3-E5: not run.**

## F. Conclusion (only what the evidence supports)

1. Does divide-and-conquer improve 14-class macro-F1? **Unknown - not tested.** The ceiling says at most +0.056 (LOFO) / +0.069 (manual) with a perfect specialist.
2. Best grouping strategy? By ceiling: 4-class groups only; LOFO size 4 is the only leak-free one that clears the gate. Specialist-level comparison not run.
3. Do specialist errors complement the general model? **Not measured.**
4. Hard routing vs fusion? **Not measured.** Hard routing has a known ceiling above; fusion's is bounded by it.
5. Meets +0.04 on all four folds? **Not measured.** Only the perfect-specialist ceiling does, with a thin margin.
6. Which classes improve/regress? Not measured. Ceiling gains concentrate in Shooting (+0.33 F1), Abuse (+0.27), Assault (+0.23), Fighting (+0.14) under violent_quad.
7. Why would it fail? Evidence so far: groups unstable across folds; only 4-class groups survive; the gate needs a near-perfect specialist while prior within-group accuracy is ~0.6; 56-video folds make small-class F1 very noisy; Shooting is at F1 0.02 so the group's largest gain depends on a class the features barely separate. These are reasons for pessimism, not a measured failure.
8. Next most informative experiment? Run E3-E5 on the two kept groups only (LOFO size 4 in holdout mode as primary; violent_quad as secondary): one command, CPU-only head training, ~30-60 min of slice (the Kyverno policy still claims the GPU). Its single decisive number is within-group accuracy vs the ~0.9+ a pass would need.

## G. Reproduce

```
./.venv/bin/python scripts/run_confusion_analysis.py        # E0-E2 -> results/divide-and-conquer/e0-e2/
./.venv/bin/python -m unittest discover -s tests -p "test_*.py"
# E3-E5 (needs the cluster; state ETA first; do not edit/commit while Running):
scripts/dgx/start_pod_job.sh dac-specialists './.venv/bin/python scripts/run_specialist_experiment.py --repro-check'
# -> results/divide-and-conquer/e3-e5/{summary.json,specialist_probs.csv,config.json}
# save the pod log to results/p1c/pod-logs/ before deleting the pod
```
