# Text-Driven Digital Human Integration

This directory contains the integration layer used to explore a text-driven
upper-body digital-human pipeline:

1. **HandMDM** converts a natural-language hand-motion prompt into a 274-D
   motion sequence at 25 fps.
2. `prepare_pose_for_echomimic.py` resamples the sequence to 24 fps, converts
   HandMDM's 6-D hand rotations into 3-D joints, projects the joints into the
   reference-image coordinate system, and exports per-frame pose data.
3. **EchoMimic** performs audio-conditioned portrait rendering using the
   prepared pose sequence, subject to the pose interface of the selected
   EchoMimic revision.

## What is included

- `run_pipeline.py` - one command-line entry point for orchestration.
- `prepare_pose_for_echomimic.py` - the HandMDM-to-pose conversion bridge.
- `handmdm2echomimic.py` - an earlier experimental all-in-one prototype,
  retained for comparison.

Model weights, generated data, SMPL assets, DWPose ONNX files, and upstream
source trees are intentionally excluded.

## Upstream projects

- HandMDM: <https://github.com/leorebensabath/HandMDM>
- EchoMimic: <https://github.com/antgroup/echomimic>
- EchoMimicV2: <https://github.com/antgroup/echomimic_v2>

Install and configure the upstream projects according to their own READMEs and
licenses. This integration code does not redistribute their checkpoints or
datasets.

## Usage

```bash
python run_pipeline.py \
  --text "the right hand waves hello" \
  --reference assets/reference.png \
  --audio assets/speech.wav \
  --handmdm-root /path/to/HandMDM \
  --echomimic-root /path/to/echomimic \
  --checkpoint /path/to/last.ckpt \
  --output-dir outputs/demo \
  --device cuda
```

Add `--dry-run` to print the full command sequence without loading models.

The local EchoMimic checkout reviewed for this project consumes pickled facial
landmark frames, while the bridge exports DWPose-style hand frames. Because
this interface differs across EchoMimic revisions, automatic rendering is
deliberately opt-in through `--render-command`. This avoids presenting an
unverified adapter as a fully working end-to-end implementation.

## Validation status

- Python syntax and command construction are checked.
- The HandMDM inference and pose-conversion stages are wired to their actual
  command-line interfaces.
- Full GPU inference is not included in repository validation because it
  requires external checkpoints and model assets.

## Attribution

The orchestration and conversion bridge are project integration work by
Weijian Zhang. HandMDM and EchoMimic remain separate upstream projects and
retain their original authorship and licenses.
