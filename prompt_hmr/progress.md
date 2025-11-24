# PromptHMR Video Evaluation Progress

## Completed
- Added world-coordinate metric utilities (`w_mpjpe100`, `wa_mpjpe100`, `rte`) in `prompt_hmr/utils/vid_eval_utils.py`.
- Rebuilt `prompt_hmr/vid_evaluator.py` to:
  - Run/consume the multi-stage video pipeline for EMDB sequences.
  - Convert predicted SMPL-X parameters to SMPL space, scatter them to the full video timeline, and compute both camera/world metrics.
  - Aggregate per-video statistics and expose them via the evaluator API.
- `scripts/eval_emdb.py` now only handles inference/caching; added `scripts/eval_emdb_metrics.py` to compute metrics from cached results so evaluation can be split into two steps.
- Created `tests/test_eval_emdb.py` as unit test to verify evaluator correctness against cached EMDB results and unit-test the axis-angle→rotmat conversions without re-running inference.
- Added `scripts/vis_emdb_compare.py` to visualize GT vs predicted world-coordinate meshes for any cached EMDB sequence via Viser.

## TODO / Next Steps
- Extend the evaluation framework into a general one, for fairly comparing Human Body Reconstruction methods.
