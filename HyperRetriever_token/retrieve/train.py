import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path

from retrieve_dataset import RetrievalDataset, create_and_save_dataset
from model.mlp import MLP

if __name__ == "__main__":
    # --- Environment Setup ---
    parser = argparse.ArgumentParser(description="Train a retrieval model.")
    parser.add_argument("domain", type=str, help="Domain to train the model on.")
    args = parser.parse_args()

    # --- Configuration ---
    domain = args.domain
    dataset_dir = Path(f"../expr/wikitopics/{domain}/train")
    json_path = dataset_dir / "retrieval_samples.json"
    dataset_save_path = dataset_dir / "retrieval_dataset.pt"
    model_save_path = dataset_dir / "retrieval_model.pth"
    training_log_path = dataset_dir / "training_log.txt"
    plot_save_path = dataset_dir / "loss_trend.png"
    best_model_save_path = dataset_dir / "best_retrieval_model.pth" # Added for best model saving
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    emb_size = 256 # Hidden layer size

    # Early Stopping Parameters
    patience = 10 # Number of epochs to wait if no improvement
    min_delta = 1e-5 # Minimum change to be considered an improvement

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Dataset Loading ---
    if os.path.exists(dataset_save_path):
        saved_data = torch.load(dataset_save_path)
        dataset = RetrievalDataset(saved_data)
    else:
        dataset = create_and_save_dataset(
            json_file_path=json_path,
            save_path=dataset_save_path
        )
    
    # --- Dataset Splitting ---
    # Get labels for stratified splitting
    labels = [sample['label'].item() for sample in dataset] # Assuming labels are tensors
    
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2, # 20% for validation
        stratify=labels,
        random_state=42 # for reproducibility
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, shuffle=False) # No need to shuffle validation data
    
    # --- Logging Setup ---
    os.makedirs(os.path.dirname(training_log_path), exist_ok=True)
    log_file = open(training_log_path, "a") # Open in append mode

    def log_print(message):
        print(message)
        log_file.write(message + "\n")

    log_print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Using device: {device}")
    log_print(f"Loading pre-processed dataset from {dataset_save_path}...")
    log_print("Dataset loaded successfully." if os.path.exists(dataset_save_path) else "Saved dataset not found. Creating a new one...")
    log_print(f"\nDataset contains {len(dataset)} samples.")
    log_print(f"Training set size: {len(train_dataset)} samples.")
    log_print(f"Validation set size: {len(val_dataset)} samples.")
    log_print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Number of epochs: {num_epochs}, Embedding size: {emb_size}")
    log_print(f"Early stopping patience: {patience}, Minimum delta: {min_delta}") # Log early stopping params
    
    # --- Model, Loss, and Optimizer Initialization ---
    # Get input feature size from the first batch of the training dataloader
    first_batch = next(iter(train_dataloader))
    pred_in_size = first_batch['features'].shape[1]
    
    model = MLP(pred_in_size=pred_in_size, emb_size=emb_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_print(f"Model initialized with input size {pred_in_size} and embedding size {emb_size}.")

    # --- Training Loop ---
    log_print("\nStarting training...")
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        # Training Phase
        model.train() # Set the model to training mode
        total_train_loss = 0
        # for i, batch in enumerate(train_dataloader): # This loop is removed to only print final epoch loss
        for batch in train_dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)     # Ensure labels are of shape [batch_size, 1]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval() # Set the model to evaluation mode
        total_val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            # for i, batch in enumerate(val_dataloader): # This loop is removed to only print final epoch loss
            for batch in val_dataloader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)

                outputs = model(features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # Combined log for train and validation loss for the epoch
        log_print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'pred_in_size': pred_in_size,
                'emb_size': emb_size
            }, best_model_save_path)
            log_print(f"Validation loss improved. Saving best model to {best_model_save_path}")
        else:
            epochs_no_improve += 1
            log_print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                log_print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {patience} consecutive epochs.")
                break # Stop training loop

    log_print("\nTraining finished.")

    # --- Plotting Loss Trends ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Trends')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    log_print(f"Loss trend plot saved to {plot_save_path}")

    # Close the log file
    log_file.close()