import os
import pathlib
import h5py
from pathlib import Path
import json
import itertools
import json
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset
from helpers import run_cross_validation

def main():
    # ==================================== Import Data ====================================
    clinical_data_dir = pathlib.Path("/mnt/lab-share/home/u5534904/cloud_workspace/Embeddings/clinical_data")
    histology_dir = pathlib.Path("/mnt/lab-share/home/u5534904/cloud_workspace/Embeddings/Histology/slide_features_prism_20x_256")

    patients = []
    patients = [int(os.path.splitext(filename)[0]) for filename in os.listdir(clinical_data_dir)]

    # BCR dictionary (key: patient_id, val: BCR)
    patient_BCR_dict = {}

    for patient in patients:
        file_path = clinical_data_dir / f"{patient}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        patient_BCR_dict[patient] = [float(data["time_to_follow-up/BCR"]), float(data["BCR"])]

    # Histology dictionary (key: patient_id, val: histology_embedding)
    h5_files = list(Path(histology_dir).glob("*.h5")) + list(Path(histology_dir).glob("*.hdf5"))
    histology_embedding_dict = {}
    histology_embedding_dim = 0
    patient_files = {}

    for h5_file in h5_files:
        patient_id = int(h5_file.stem[:4])

        # Store h5 files for each patient
        if patient_id in patient_files:
            patient_files[patient_id].append(h5_file)
        else:
            patient_files[patient_id] = [h5_file]


    # Extract a random histology embedding for each patient
    for patient_id, histology_embedding_list in patient_files.items():
        # Extract random histology embedding if patient has more than 1 wsi image
        h5_file = random.choice(histology_embedding_list)

        with h5py.File(h5_file, 'r') as f:
            histology_embedding = f["features"][:]
            histology_embedding_dict[patient_id] = histology_embedding

            # Store dimension of histology vector
            if histology_embedding_dim == 0:
                histology_embedding_dim = histology_embedding.shape[0]


    # Overall data dictionary (key: patient_id, val: dictionary containing histology embedding and BCR status)
    patient_data = {}
    for patient_id in patients:
        patient_data[patient_id] = {
            "histology_embedding": histology_embedding_dict[patient_id],
            "BCR": patient_BCR_dict[patient_id]
        }

    # Convert data to TensorDatasets
    histology_embeddings = []
    labels = []
    for patient_id, data in patient_data.items():
        histology_embeddings.append(data["histology_embedding"])
        labels.append(data["BCR"])

    histology_embeddings_tensor = torch.tensor(histology_embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create TensorDataset with histology embeddings, radiology embeddings and labels
    patient_dataset = TensorDataset(histology_embeddings_tensor, labels_tensor)

    # ==================================== Hyperparameter Tuning ====================================
    learning_rate = 0.0001

    # List of hyperparameter options to try
    batch_size_options = [4, 8]
    layer_sizes_options = [
        [256, 32],
        [512, 64],
        [256, 128, 32],
        [512, 256, 64]
    ]
    dropout_options = [0.1, 0.2]

    # Get list of all hyperparameter combinations
    param_grid = list(itertools.product(
        batch_size_options,
        layer_sizes_options,
        dropout_options
    ))

    # Hyperparameter grid search
    best_params = {}
    best_c_index = 0
    num_repeats = 1
    k_folds = 5

    # Iterate through all hyperparameter combinations
    for i, (batch_size, layer_sizes, dropout) in enumerate(param_grid):
        print(f"\n{'='*20} Hyperparameter Combination {i + 1}/{len(param_grid)} {'='*20}")
        print(f"Batch Size: {batch_size}, Layer Sizes: {layer_sizes}, Dropout: {dropout}")

        # Run repeated K-fold CV
        all_run_results = []
        for i in range(num_repeats):
            seed = 42 * (i + 1)
            print(f"\n{'='*20} Repetition {i + 1}/{num_repeats} (seed={seed}) {'='*20}")
            results_for_run = run_cross_validation(
                random_seed=seed,
                layer_sizes=layer_sizes,
                learning_rate=learning_rate,
                batch_size=batch_size,
                dropout=dropout,
                k_folds=k_folds,
                patient_dataset=patient_dataset,
                labels_tensor=labels_tensor,
                embedding_dim=histology_embedding_dim
            )
            all_run_results.extend(results_for_run)

        c_index_scores = [r['c_index'] for r in results_for_run]
        avg_c_index = np.mean(c_index_scores)
        print(f"Average C-Index for this combo: {avg_c_index :.4f}")

        if avg_c_index > best_c_index:
            best_c_index = avg_c_index
            best_params = {
                'batch_size': batch_size,
                'layer_sizes': layer_sizes,
                'dropout': dropout
            }


    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best C-Index: {best_c_index:.4f}")
    
    # ==================================== Repeated Cross Validation ====================================
    # Run repeated K-fold CV using best set of hyperparameters
    batch_size = best_params['batch_size']
    layer_sizes = best_params['layer_sizes']
    dropout = best_params['dropout']

    num_repeats = 10
    all_run_results = []
    for i in range(num_repeats):
        seed = 42 * (i + 1)
        print(f"\n{'='*20} Repetition {i + 1}/{num_repeats} (seed={seed}) {'='*20}")
        results_for_run = run_cross_validation(
                random_seed=seed,
                layer_sizes=layer_sizes,
                learning_rate=learning_rate,
                batch_size=batch_size,
                dropout=dropout,
                k_folds=k_folds,
                patient_dataset=patient_dataset,
                labels_tensor=labels_tensor,
                embedding_dim=histology_embedding_dim
            )
        all_run_results.extend(results_for_run)


    # ==================================== Final Results ====================================
    c_index_scores = [r['c_index'] for r in all_run_results]

    print("\n=============== Final Cross-Validation Summary ===============")
    print(f"(Based on {num_repeats} repetitions of {k_folds}-fold CV, total {len(all_run_results)} runs)\n")
    print(f"Avg C-Index: {np.mean(c_index_scores):.4f} ± {np.std(c_index_scores):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()