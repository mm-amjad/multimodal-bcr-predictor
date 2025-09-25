import os
import pathlib
import h5py
from pathlib import Path
import json
import itertools
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset
from helpers import run_cross_validation

def main():
    # ==================================== Import Data ====================================
    clinical_data_dir = pathlib.Path("/mnt/lab-share/home/u5534904/cloud_workspace/Embeddings/clinical_data")
    radiology_dir = pathlib.Path("/mnt/lab-share/home/u5534904/cloud_workspace/Embeddings/Radiology/MedicalNet")


    patients = []
    patients = [int(os.path.splitext(filename)[0]) for filename in os.listdir(clinical_data_dir)]

    # BCR dictionary (key: patient_id, val: BCR)
    patient_BCR_dict = {}

    for patient in patients:
        file_path = clinical_data_dir / f"{patient}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        patient_BCR_dict[patient] = [float(data["time_to_follow-up/BCR"]), float(data["BCR"])]

   # Radiology dictionary (key: patient_id, val: radiology_embedding)
    radiology_embedding_dim = 0
    npy_files = list(Path(radiology_dir).glob("*.npy"))
    radiology_embedding_dict = {}

    for npy_file in npy_files:
        radiology_embedding = np.load(npy_file)
        radiology_embedding_dict[patient_id] = radiology_embedding

        # Store dimension of histology vector
        if radiology_embedding_dim == 0:
            radiology_embedding_dim = radiology_embedding.shape[0]

    # Overall data dictionary (key: patient_id, val: dictionary containing histology embedding, radiology embedding and BCR status)
    patient_data = {}
    for patient_id in patients:
        patient_data[patient_id] = {
            "radiology_embedding": radiology_embedding_dict[patient_id],
            "BCR": patient_BCR_dict[patient_id]
        }

    # Convert data to TensorDatasets
    radiology_embeddings = []
    labels = []
    for patient_id, data in patient_data.items():
        radiology_embeddings.append(data["radiology_embedding"])
        labels.append(data["BCR"])

    radiology_embeddings_tensor = torch.tensor(radiology_embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create TensorDataset with histology embeddings, radiology embeddings and labels
    patient_dataset = TensorDataset(radiology_embeddings_tensor, labels_tensor)

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
                embedding_dim=radiology_embedding_dim
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
                embedding_dim=radiology_embedding_dim
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
