import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pycox.models import CoxPH
from lifelines.utils import concordance_index

def run_cross_validation(random_seed, 
                         batch_size, 
                         dropout, 
                         layer_sizes, 
                         learning_rate, 
                         embedding_dim,
                         patient_dataset, 
                         labels_tensor, 
                         k_folds):
    """Performs k-fold cross-validation. 
    
    Args:
        random_seed (int): Random seed for reproducibility
        batch_size (int): Training batch size
        dropout (float): Dropout probability for regularization
        layer_sizes (list): Hidden layer dimensions for MLP
        learning_rate (float): Learning rate for optimization
        histology_embedding_dim (int): Dimension of histology embeddings
        patient_dataset (TensorDataset): Complete tensor dataset with histology and time to BCR ground truth lavbels
        labels_tensor (torch.Tensor): Survival labels [time, event]
        k_folds (int): Number of cross-validation folds
        
    Returns:
        list: List of dictionaries containing fold results with c-index scores
        
    Example:
        >>> results = run_cross_validation(
        ...     random_seed=42, batch_size=8, dropout=0.1,
        ...     layer_sizes=[256, 64], learning_rate=0.001,
        ...     histology_embedding_dim=512, patient_dataset=dataset,
        ...     labels_tensor=labels, k_folds=5
        ... )
        >>> return fold_results = [{c_index: 0.7}, {c_index: 0.75}, ...]
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    fold_results = []

    print("Starting CV!")
    for fold, (train_ids, test_ids) in enumerate(skf.split(labels_tensor[:, 1], labels_tensor[:, 1])):
        print(f"FOLD {fold + 1}/{k_folds}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_hist = patient_dataset.tensors[0]  # Full hist embedding tensor
        all_labels = patient_dataset.tensors[1] # Full labels tensor

        # Extract train/test sets
        hist_train = all_hist[train_ids]
        hist_test = all_hist[test_ids]
        labels_train = all_labels[train_ids]
        labels_test = all_labels[test_ids]

        # Split train set into 80/20 train/val
        hist_train, hist_val, labels_train, labels_val = train_test_split(
            hist_train, labels_train, test_size=0.2)

        # Scale histology and radiology embeddings (train/val)
        hist_scaler = StandardScaler()
        hist_train_2d = hist_train.view(hist_train.size(0), -1).numpy()
        hist_scaler.fit(hist_train_2d)
        hist_train_scaled = hist_scaler.transform(hist_train_2d)
        hist_val_scaled = hist_scaler.transform(hist_val.view(hist_val.size(0), -1).numpy())

        # Reshape back to original dimensions and convert to tensors 
        hist_train_tensor = torch.from_numpy(hist_train_scaled).float().view(hist_train.shape).to(device)
        hist_val_tensor = torch.from_numpy(hist_val_scaled).float().view(hist_val.shape).to(device)
        labels_train = labels_train.to(device)
        labels_val = labels_val.to(device)

        # Create TensorDatasets and DataLoaders with train/val data
        train_dataset = TensorDataset(hist_train_tensor, labels_train)
        val_dataset = TensorDataset(hist_val_tensor, labels_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        # Initialise attention and evaluation models 
        attention_model = CrossAttentionFusion(embedding_dim=embedding_dim, dropout=dropout).to(device)
        optimizer_attention = torch.optim.Adam(attention_model.parameters(), lr=learning_rate)
        net = MLP(embedding_dim, layer_sizes, dropout).to(device)
        optimizer_eval = torch.optim.Adam(net.parameters(), lr=learning_rate)
        model_fold = CoxPH(net)
        model_fold.net = model_fold.net.to(device)
        criterion = model_fold.loss

        # Initialise copy of attention and regression models for early stopping
        attention_model_copy = CrossAttentionFusion(embedding_dim=embedding_dim, dropout=dropout).to(device)
        optimizer_attention_copy = torch.optim.Adam(attention_model_copy.parameters(), lr=learning_rate)
        net_copy = MLP(embedding_dim, layer_sizes, dropout).to(device)
        optimizer_copy = torch.optim.Adam(net_copy.parameters(), lr=learning_rate)
        model_copy = CoxPH(net_copy)
        model_copy.net = model_copy.net.to(device)
        criterion_copy = model_copy.loss

        # Calculate optimal num_epochs
        num_epochs = train_with_divergence_stopping(
            attention_model=attention_model_copy,
            evaluation_model=net_copy,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion_copy,
            optimizer_evaluation=optimizer_copy,
            optimizer_attention=optimizer_attention_copy,
            device=device
        )

        print(f"Optimal Epochs: {num_epochs}")

        # Extract train/test sets
        hist_train = all_hist[train_ids]
        hist_test = all_hist[test_ids]
        labels_train = all_labels[train_ids]
        labels_test = all_labels[test_ids]

        # Scale histology and radiology embeddings (train/test)
        hist_scaler = StandardScaler()
        hist_train_2d = hist_train.view(hist_train.size(0), -1).numpy()
        hist_scaler.fit(hist_train_2d)
        hist_train_scaled = hist_scaler.transform(hist_train_2d)
        hist_test_scaled = hist_scaler.transform(hist_test.view(hist_test.size(0), -1).numpy())

        # Reshape back to original dimensions and convert to tensors 
        hist_train_tensor = torch.from_numpy(hist_train_scaled).float().view(hist_train.shape).to(device)
        hist_test_tensor = torch.from_numpy(hist_test_scaled).float().view(hist_test.shape).to(device)

        labels_train = labels_train.to(device)
        labels_test = labels_test.to(device)

        # Create TensorDatasets and DataLoaders with train/test data
        train_dataset = TensorDataset(hist_train_tensor, labels_train)
        test_dataset = TensorDataset(hist_test_tensor, labels_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        

        # Training loop
        print("Starting training")
        net.train()
        for epoch in range(num_epochs):
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")

            for hist_embedding, labels in train_loader:
                hist_embedding, labels = hist_embedding.to(device), labels.to(device)
                optimizer_eval.zero_grad()
                durations = labels[:, 0]
                events = labels[:, 1]

                if events.sum() == 0:
                    continue 

                log_hazards = net(hist_embedding)
                loss = criterion(log_hazards, durations, events)

                if torch.isnan(loss):
                    print("Loss became NaN. Stopping training.")
                    break

                loss.backward()
                optimizer_attention.step()
                optimizer_eval.step()

        # Validation loop
        net.eval()
        all_durations, all_events, all_scores = [], [], []
        with torch.no_grad():
            for hist_embedding, labels in test_loader:
                hist_embedding, labels = hist_embedding.to(device), labels.to(device)
                log_hazards = net(hist_embedding)
                scores = -log_hazards.cpu().numpy()
                all_durations.append(labels[:, 0].cpu().numpy())
                all_events.append(labels[:, 1].cpu().numpy())
                all_scores.append(scores)

        # Store metrics for the fold
        all_durations = np.concatenate(all_durations)
        all_events = np.concatenate(all_events)
        all_scores = np.concatenate(all_scores)

        # Calculate the Concordance Index
        c_index = concordance_index(all_durations, all_scores, all_events)
        fold_results.append({'c_index': c_index})

    return fold_results



def train_with_divergence_stopping(model, 
                                   train_loader, 
                                   val_loader, 
                                   criterion, 
                                   optimizer, 
                                   device,
                                   max_epochs=512, 
                                   patience=10, 
                                   divergence_threshold=0.02): 
    """This function trains two models simultaneously: an attention model for feature fusion and 
    an evaluation model for survival analysis. Training stops early if validation loss increases
    beyond a threshold for consecutive epochs, indicating potential overfitting.
    
    Args:
        model: Neural network model for survival analysis prediction
        train_loader (DataLoader): Training data loader containing (hist_embedding, labels)
        val_loader (DataLoader): Validation data loader with same structure as train_loader
        criterion: Loss function for survival analysis (e.g., Cox proportional hazards loss)
        optimizer: Optimizer for the evaluation model parameters
        device (torch.device): Device to run training on (CPU or CUDA)
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 512.
        patience (int, optional): Number of epochs to wait before early stopping when 
                                 validation loss diverges. Defaults to 10.
        divergence_threshold (float, optional): Threshold for validation loss increase 
                                              to trigger patience counter. Defaults to 0.02.
    
    Returns:
        int: Optimal number of epochs to train networks on. If early stopping occurred, returns 
             the epoch number minus the patience value. Otherwise returns max_epochs
    
    Notes:
        - Both models are moved to the specified device at the start of training
        - Gradient clipping is applied to evaluation model with max_norm=1.0
        - Batches with no events (events.sum() == 0) are skipped
        - Training stops immediately if loss becomes NaN
        - Validation loss is calculated only on batches containing events
        - Early stopping is triggered when validation loss increases beyond threshold
          for 'patience' consecutive epochs
    
    Example:
        >>> epochs_trained = train_with_divergence_stopping(
        ...     model=survival_net,
        ...     train_loader=train_dl,
        ...     val_loader=val_dl,
        ...     criterion=cox_loss,
        ...     optimizer=optim_eval,
        ...     device=torch.device('cuda'),
        ...     max_epochs=100,
        ...     patience=5,
        ...     divergence_threshold=0.01
        ... )
        >>> return epoch = 30
    """
    prev_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        batches_with_events = 0
        
        # Training loop 
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            
            durations = labels[:, 0]
            events = labels[:, 1]

            if events.sum() == 0:
                continue # Move to the next batch

            optimizer.zero_grad()
            log_hazards = model(inputs)
            loss = criterion(log_hazards, durations, events)

            if torch.isnan(loss):
                print("Loss became NaN. Stopping training.")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batches_with_events += 1

        if batches_with_events > 0:
            train_loss /= batches_with_events

        # Validation loop
        model.eval()
        val_loss = 0
        batches_with_events = 0
        all_durations, all_events, all_scores = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device) 
                log_hazards = model(inputs)
                durations = labels[:, 0]
                events = labels[:, 1]

                if events.sum() == 0:
                    continue # Move to the next batch

                loss = criterion(log_hazards, durations, events)
                val_loss += loss.item()
                scores = -log_hazards.cpu().numpy()  
                all_durations.append(labels[:, 0].cpu().numpy())  
                all_events.append(labels[:, 1].cpu().numpy())  
                all_scores.append(scores)
                batches_with_events += 1

        if batches_with_events > 0:
            val_loss /= batches_with_events
            
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Note if validation loss is increasing
        if val_loss - prev_val_loss > divergence_threshold:
            patience_counter += 1
        else:
            prev_val_loss = val_loss
            patience_counter = 0

        if patience_counter >= patience:
            print(f'Early stopping due to train/val divergence at epoch {epoch+1}')
            return (epoch + 1 - patience)

    return epoch + 1  