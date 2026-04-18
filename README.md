video links
mid sem - https://drive.google.com/file/d/1gpXIkqs69U7Mq8UtF-UCV7wuIeUR4y59/view?usp=sharing
end sem - https://drive.google.com/file/d/13Yv1wFQZxecAaM303JxRiqsYtcMQHd5Y/view?usp=sharing

## What We Changed

- Added a lightweight `DistortionStatsBranch` in TOPIQ-NR to better capture synthetic distortion artifacts.
- Added an `AdaptiveFusionGate` to fuse semantic score and distortion score dynamically (learned per image, no fixed alpha).
- Kept the original CFANet + ResNet50 semantic backbone intact.
- Used frozen Gaussian high-frequency residual extraction in the distortion branch.
- Added robust training checkpointing for Kaggle sessions (timed saves, interrupt/error save, auto-resume support).
- Fixed KonIQ processing script header-row bug and standardized TID2013 test workflow.

## What Improved

- Cross-dataset performance on **TID2013** improved over baseline:
  - SRCC: `0.4452 -> 0.4542`
  - PLCC: `0.5625 -> 0.5650`
  - KRCC: `0.3143 -> 0.3197`
- KonIQ validation remained near baseline (no meaningful regression).
