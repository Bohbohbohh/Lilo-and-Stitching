# Lilo and Stitching: Embedding Mapping and Ensembling

## Project Overview

This repository contains the solution for a machine learning competition. The main objective is to build a robust **adapter** that maps text embeddings from a pre-trained text encoder into a target vision latent space. The goal is for these mapped text embeddings to match the ground-truth image embeddings produced by a provided pre-trained VAE.

The model is trained on a given dataset and must generalise to a held-out test set.

---

The code in this repository implements three different model architectures (**MLP**, **RMLPA**, and **Stitcher**) and an **ensembling** strategy to achieve this goal.

## Repository Guide

Below is a brief description of the main folders and files and their purpose.

### Main Folders

* `/checkpoints`:
    Contains the `.pth` files for the final trained models (the "best models") for each of the three architectures (`final_latent_mapper.pth`, `rmlpa_best_model.pth`, `stitcher_best_model.pth`). These are loaded for inference and submission generation.

* `/submission`:
    This crucial folder contains all the outputs required for ensembling and the final submissions.
    * **Final Submissions**: `submission_ensemble_2model.csv` and `submission_ensemble_3model.csv`.
    While running code on Kaggle you should be able to find also:
    * **Single Model Embeddings**: Contains the raw predictions from each model, saved in both `.csv` format (for direct submission) and `.npz` format (optimized for ensembling).
    * **Ensemble Support Files**: Includes intermediate `.npz` files (e.g., `val_mlp.npz`, `val_rmlpa.npz`, `val_stiticher.npz`, `gallery_data.npz`) that contain pre-calculated validation or gallery data, essential for building the final ensembles in the notebook.

### Main Files

* `utils_*.py`:
    These represent the "brain" of the entire pipeline. These scripts contain all the logic, model class definitions, training functions, validation utilities (like the MRR calculation), and functions to generate single-model submissions (e.g., `generate_dml_submission` in `utils_mlp.py`).

* `final-notebook-lilo-and-stitching.ipynb`:
    This is the Jupyter notebook that orchestrates the entire process. It loads the "best models" from the `/checkpoints` folder if the flags to re-run the training are false, runs training in the other case, then it generates predictions for the ensemble (using the logic defined in the `utils` files), and finally performs the "stitching" (ensembling) phase by combining the various `.npz` files to produce the final submissions.

* `Report_Lilo_and_Stitching`:
    The file (e.g., `.pdf` or `.md`) containing the final project report, which describes in detail the approach, model architectures, tuning process, and the results achieved.
