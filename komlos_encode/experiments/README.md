# Experiments

This folder holds the classification pipelines used to benchmark KomlosEncode,
plus the preprocessing script that turns sequences into the images they train
on.

## Pipeline

1. **Generate images** from sequences using the installed package:

   ```bash
   python experiments/generate_images.py \
       --input data/sequences.jsonl \
       --outdir Zoom_Images/KomlosEncoding \
       --labels-out labels.csv \
       --mode dna --k 1
   ```

   This writes `Zoom_Komlos_<index>.png` files and a `labels.csv`. It is the
   step that was previously not committed: every image the training scripts
   consume is now reproducible from the encoder itself.

2. **Train / evaluate** with one of the pipelines:

   | Script | Purpose |
   | --- | --- |
   | `final_komlos.py` | Optuna hyperparameter search + InceptionV3 training (CPU) |
   | `final_komlos_gpu.py` | GPU version with per-epoch memory logging |
   | `final_komlos_kf.py` | Stratified k-fold cross-validation + stability analysis |

   Point the `image_path` / `data_path` variables at the outputs of step 1.

## Notes

- The training scripts require the heavier stack (`torch`, `torchvision`,
  `optuna`, `opencv-python`, `Pillow`, `psutil`); see the top-level README.
- If your published figures used the FocusFFT / "Zoom" preprocessing (Amer et
  al., 2025), insert that transform in `generate_images.py` between encoding and
  saving. The default output is the plain Hilbert layout.
