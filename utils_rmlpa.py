import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from challenge.src.common.utils import load_data, prepare_train_data, generate_submission

class ResidualBottleneckAdapter(nn.Module):
    """
    Maps input text embeddings (D_in) to the image embedding space (D_out)
    using a residual bottleneck architecture.
    """
    def __init__(self, D_in: int, D_out: int, 
                 D_bottle_ratio: int = 4, dropout_p: float = 0.1):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        # Calculate bottleneck dimension as a fraction of the input
        D_bottle = D_in // D_bottle_ratio

        self.block = nn.Sequential(
            nn.Linear(D_in, D_bottle),
            nn.GELU(),  
            nn.Dropout(dropout_p),
            nn.Linear(D_bottle, D_out)
        )

        # A shortcut connection is required for the residual sum.
        # If dimensions mismatch, use a linear projection.
        if D_in == D_out:
            self.shortcut = nn.Identity()
        else:
            # No bias is needed, as it's just for dimension matching.
            self.shortcut = nn.Linear(D_in, D_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_out = self.block(x)
        shortcut_out = self.shortcut(x)
        
        output = block_out + shortcut_out
        
        # Normalize the final output for stable cosine similarity calculation
        output_normalized = F.normalize(output, p=2, dim=1)
        
        return output_normalized

def normalize_l2(embeddings):
    """Helper function for L2 normalization."""
    return F.normalize(embeddings, p=2, dim=-1)

class PairedEmbeddingDataset(Dataset):
    """
    Simple Dataset to hold (text, image) embedding pairs.
    """
    def __init__(self, z_text, z_img):
        self.z_text = z_text
        self.z_img = z_img
        assert len(self.z_text) == len(self.z_img), "Dataset size mismatch"

    def __len__(self):
        return len(self.z_text)

    def __getitem__(self, idx):
        return self.z_text[idx], self.z_img[idx]
    
def create_dataset_from_indices(indices, z_text_all, z_img_unique, text_to_image_map):
    """
    Factory function to build a PairedEmbeddingDataset from specific indices.
    Normalizes embeddings upon creation.
    """
    if len(indices) == 0:
        return None 

    z_text_np = z_text_all[indices]
    img_indices_map = text_to_image_map[indices]
    z_img_np = z_img_unique[img_indices_map]
    
    # Normalize embeddings before storing to save computation during training
    z_text_tensor = normalize_l2(torch.from_numpy(z_text_np).float())
    z_img_tensor  = normalize_l2(torch.from_numpy(z_img_np).float())
    
    return PairedEmbeddingDataset(z_text_tensor, z_img_tensor)

def setup_environment(seed, device_str="cuda"):
    """Sets random seeds and device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and device_str == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- Using device: {device} ---")
    return device

def load_and_prepare_data(train_data_path, submission_dir, device):
    """
    Loads main training data and creates/saves a global gallery of
    unique image embeddings for validation.
    """
    print("Loading Training Data...")
    train_data_dict = load_data(train_data_path)
    z_text_raw, z_img_raw, label_matrix = prepare_train_data(train_data_dict)

    # Pre-load all data to VRAM, assuming it fits.
    X_FINAL = z_text_raw.float().to(device) 
    Y_FINAL = z_img_raw.float().to(device) 
    print("Training data loaded.")

    # Convert one-hot label_matrix to dense group IDs
    groups = np.argmax(label_matrix, axis=1)

    print("Creating Unique Global Gallery...")
    Y_FINAL_np = normalize_l2(Y_FINAL).cpu().numpy()
    # Get the first occurrence index for each group to build a unique gallery
    all_unique_group_ids, all_unique_indices = np.unique(groups, return_index=True)

    Y_gallery_unique_ALL = Y_FINAL_np[all_unique_indices] 
    groups_gallery_unique_ALL = all_unique_group_ids
    print(f"Global Gallery created.")

    # Save the gallery for later use in ensemble tuning
    gallery_npz_path = Path(submission_dir) / "gallery_data.npz"
    np.savez(
        gallery_npz_path, 
        embeddings=Y_gallery_unique_ALL, 
        groups=groups_gallery_unique_ALL
    )
    print(f"Gallery saved for tuning.")
    
    return X_FINAL, Y_FINAL, groups, Y_gallery_unique_ALL, groups_gallery_unique_ALL

def create_train_val_test_splits(X_FINAL, groups, temp_split_ratio, test_split_ratio_of_temp, seed):
    """
    Splits data into train/val/test sets using GroupShuffleSplit to ensure
    no group leakage between sets.
    """
    print("\nPreparing data...")
    # Split 1: (Train) 90% vs (Temp) 10%
    print("Split 1: Creating Training set (90%) and Temporary set (10%) set...")
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=temp_split_ratio, random_state=seed)
    train_indices, temp_indices = next(gss_train_temp.split(X_FINAL.cpu().numpy(), y=None, groups=groups))

    # Split 2: (Temp) 10% -> (Val) 5% vs (Test) 5%
    print("Split 2: Splitting Temporary set into Validation set (5%) and Test set (5%)...")
    groups_temp = groups[temp_indices]
    X_temp_dummy = np.empty((len(groups_temp), 1)) 
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=test_split_ratio_of_temp, random_state=seed)
    val_indices_rel, test_indices_rel = next(gss_val_test.split(X_temp_dummy, y=None, groups=groups_temp))

    # Convert relative indices from temp back to absolute indices
    val_indices = temp_indices[val_indices_rel]
    test_indices = temp_indices[test_indices_rel]
    groups_val = groups[val_indices] 
    groups_test = groups[test_indices]
    
    print(f"Split completed:")
    print(f"    Training set: {len(train_indices)}")
    print(f"    Validation set: {len(val_indices)}")
    print(f"    Test set: {len(test_indices)}")
    return train_indices, val_indices, test_indices, groups_val, groups_test

def create_dataloaders(X_FINAL, Y_FINAL, train_indices, val_indices, test_indices, batch_size, num_workers):
    print("Creating Datasets and DataLoaders...")
    train_dataset = PairedEmbeddingDataset(X_FINAL[train_indices], Y_FINAL[train_indices])
    val_dataset = PairedEmbeddingDataset(X_FINAL[val_indices], Y_FINAL[val_indices])
    test_dataset = PairedEmbeddingDataset(X_FINAL[test_indices], Y_FINAL[test_indices])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=False, drop_last=True
    )
    # Use larger batch size for inference as no gradients are stored
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    print("DataLoaders created and ready.")
    return train_loader, val_loader, test_loader

def setup_model_and_optimizer(dim_roberta, dim_dino, d_bottle_ratio, dropout_p, init_temperature, learning_rate, weight_decay, num_epochs, train_loader_len, device):
    print("Initializing model, loss, and optimizer...")
    adapter = ResidualBottleneckAdapter(
        D_in=dim_roberta, 
        D_out=dim_dino,
        D_bottle_ratio=d_bottle_ratio,
        dropout_p=dropout_p
    ).to(device)
    # Temperature is a learnable parameter for the contrastive loss
    temperature = nn.Parameter(torch.tensor(init_temperature)).to(device)
    # Ensure temperature is also optimized
    all_parameters = list(adapter.parameters()) + [temperature]
    optimizer = optim.AdamW(
        all_parameters, 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * train_loader_len, 
        eta_min=1e-6 # Drive learning rate to near-zero
    )
    print("Setup complete. \nReady for training.")
    return adapter, temperature, optimizer, scheduler

def calculate_mrr_validation_sampled(X_queries_proj, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL, n_samples=99):
    """
    Calculates Mean Reciprocal Rank (MRR) using a sampled gallery
    (1 positive + 99 random negatives) for efficiency.
    """
    if X_queries_proj.shape[0] == 0:
        return 0.0

    N_queries = X_queries_proj.shape[0]
    N_gallery = Y_gallery_unique_ALL.shape[0]

    # Pre-normalize all vectors for efficient dot product (cosine similarity)
    Y_gallery_unique_ALL_norm = normalize(Y_gallery_unique_ALL, axis=1)
    X_queries_proj_norm = normalize(X_queries_proj, axis=1)

    all_gallery_indices = np.arange(N_gallery)
    
    mrr_sum = 0
    pbar_val = tqdm(range(N_queries), desc="[Validation] MRR calculation (1+99)", leave=False, disable=True)

    for i in pbar_val:
        query_vec = X_queries_proj_norm[i:i+1] 
        true_query_group_id = groups_val[i]
        
        # Find the index of the one correct gallery item
        correct_gallery_index = np.where(groups_gallery_unique_ALL == true_query_group_id)[0][0]
        
        # Create a pool of all *negative* indices
        is_negative = (all_gallery_indices != correct_gallery_index)
        negative_indices_pool = all_gallery_indices[is_negative]
        
        # Randomly sample 'n_samples' (99) from the negative pool
        sampled_negative_indices = np.random.choice(
            negative_indices_pool, 
            n_samples, 
            replace=False
        )
        
        # Create the final 1+99 gallery for this query
        candidate_indices = np.concatenate(
            ([correct_gallery_index], sampled_negative_indices)
        )
        
        candidate_gallery_vecs = Y_gallery_unique_ALL_norm[candidate_indices] 
        
        # Calculate similarity scores
        similarity_scores = np.dot(query_vec, candidate_gallery_vecs.T)[0]
        
        # Get the rank of the correct item
        ranked_indices = np.argsort(similarity_scores)[::-1]

        # Find where our correct item (at index 0) ended up in the ranks
        rank_zero_based = np.where(ranked_indices == 0)[0][0]
        
        # Add reciprocal rank (1-based)
        mrr_sum += 1.0 / (rank_zero_based + 1)
            
    return mrr_sum / N_queries

@torch.no_grad()
def run_inference(data_loader, adapter, device):
    """
    Helper function to run the adapter model over a dataloader.
    """
    adapter.eval() # Set model to evaluation mode
    all_translated_embeddings = []
    
    for batch in tqdm(data_loader, desc="[Validation] Execution R-MLP-A", leave=False):
        z_text = batch[0].to(device)
        z_final_norm = adapter(z_text)
        all_translated_embeddings.append(z_final_norm.cpu())
        
    return torch.cat(all_translated_embeddings, dim=0).numpy()

def train_loop(num_epochs, adapter, train_loader, val_loader, optimizer, scheduler, temperature, device, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL, patience, checkpoint_path):
    """
    Main training and validation loop with InfoNCE loss and early stopping.
    """
    best_mrr = -1.0
    epochs_no_improve = 0 # Counter for early stopping

    print(f"    Starting Training (InfoNCE Loss)")
    for epoch in range(num_epochs):
        
        adapter.train() # Set model back to training mode
        running_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for z_text, z_img in pbar_train:
            z_text, z_img = z_text.to(device), z_img.to(device)

            # Create diagonal labels
            labels = torch.arange(z_text.shape[0], device=device)
            mapped_text = adapter(z_text) 
            norm_vae = F.normalize(z_img, p=2, dim=1) 

            # Calculate cosine similarity
            logits = torch.matmul(mapped_text, norm_vae.T)

            # Scale logits by the learnable temperature
            logits = logits * temperature.exp()

            # Symmetric InfoNCE loss
            loss_i = F.cross_entropy(logits, labels) # Text-to-Image loss
            loss_t = F.cross_entropy(logits.T, labels) # Image-to-Text loss
            loss = (loss_i + loss_t) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step() 
            running_loss += loss.item()
            pbar_train.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        
        X_queries_proj_val = run_inference(val_loader, adapter, device)
        X_queries_proj_val = np.nan_to_num(X_queries_proj_val, nan=0.0, posinf=1e6, neginf=-1e6)
        
        current_mrr = calculate_mrr_validation_sampled(
            X_queries_proj=X_queries_proj_val,
            groups_val=groups_val,
            Y_gallery_unique_ALL=Y_gallery_unique_ALL,
            groups_gallery_unique_ALL=groups_gallery_unique_ALL
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Mode: DML (InfoNCE) | Avg Loss: {avg_train_loss:.4f} | Val MRR: {current_mrr:.4f} | LR: {current_lr:1.1e}")

        if current_mrr > best_mrr:
            best_mrr = current_mrr
            epochs_no_improve = 0
            print(f"  -> New best MRR! Saving model to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'adapter_state_dict': adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'temperature': temperature,
                'best_mrr': best_mrr
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            print(f"  -> MRR did not improve ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"Early Stopping: MRR has not improved for {patience} epochs.")
            break

    print(f"Training completed. Best MRR: {best_mrr:.4f}")
    return best_mrr

def run_verification(adapter, checkpoint_path, test_loader, val_loader, device, groups_test, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL, submission_dir):
    """
    Verifies the best model on the internal test set and saves
    validation embeddings for later ensemble tuning.
    """
    print("\nStarting Verification on Internal Test Set...")

    if not Path(checkpoint_path).exists():
        print("Error: Checkpoint not found. Cannot verify.")
        return

    print(f"Loading best model from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except:
         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    adapter.load_state_dict(checkpoint['adapter_state_dict'])

    # Run on Internal Test Set
    X_queries_proj_test = run_inference(test_loader, adapter, device)
    X_queries_proj_test = np.nan_to_num(X_queries_proj_test, nan=0.0, posinf=1e6, neginf=-1e6)

    test_mrr = calculate_mrr_validation_sampled(
        X_queries_proj=X_queries_proj_test,
        groups_val=groups_test, 
        Y_gallery_unique_ALL=Y_gallery_unique_ALL,
        groups_gallery_unique_ALL=groups_gallery_unique_ALL
    )
    print(f"MRR on Internal Test Set: {test_mrr:.4f}")

    # Generate Validation Embeddings for Tuning
    print(f"\nGenerating validation embeddings for alpha tuning...")
    
    val_embeddings = run_inference(val_loader, adapter, device)
    val_embeddings_clean = np.nan_to_num(val_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    
    val_npz_path = Path(submission_dir) / "val_rmlpa.npz" 
    np.savez(
        val_npz_path, 
        embeddings=val_embeddings_clean, 
        groups=groups_val 
    )
    print(f"Validation embeddings generated.")

@torch.no_grad()
def run_submission_inference(z_text_tensor, adapter, device, batch_size=512):
    """
    Inference loop for the final submission data, using manual batching
    to handle potentially large tensors that don't fit in VRAM.
    """
    adapter.eval()
    
    all_translated = []
    num_samples = z_text_tensor.shape[0]
    
    desc = "[Submission] Execution R-MLP-A"
    for i in tqdm(range(0, num_samples, batch_size), desc=desc):
        
        batch_z_text = z_text_tensor[i:i+batch_size].to(device)
        
        z_final_norm = adapter(batch_z_text)
        all_translated.append(z_final_norm.cpu())
        
    return torch.cat(all_translated, dim=0).numpy()

def generate_submission_files(adapter, checkpoint_path, test_data_path, submission_dir, device, batch_size):
    """
    Loads the official test data, runs inference, and generates
    both an .npz (for ensembling) and a final .csv submission.
    """
    print("\nStarting Submission File Generation...")
    
    if not Path(checkpoint_path).exists():
        print("Error: Checkpoint not found. Cannot generate submission.")
        return

    # Only load the checkpoint if the model isn't already on the GPU
    if not next(adapter.parameters()).is_cuda: 
        try:
            print(f"Loading model...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except:
             checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        adapter.load_state_dict(checkpoint['adapter_state_dict'])

    print("Model loaded.")
    print(f"Loading test data...")
    test_data_clean = load_data(test_data_path)

    # Extract data using the keys from the .npz file
    z_text_test_raw = test_data_clean['captions/embeddings'] 
    sample_ids = test_data_clean['captions/ids']
    
    z_text_test_tensor = torch.from_numpy(z_text_test_raw).float()
    print(f"Test data loaded.")
    
    print("Running inference on submission data...")
    translated_embeddings = run_submission_inference(
        z_text_test_tensor, 
        adapter,
        device,
        batch_size=batch_size
    )
    
    # Ensure robustness of final output
    translated_embeddings_clean = np.nan_to_num(translated_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Save .npz for Ensembling
    submission_npz_path = Path(submission_dir) / "submission_rmlpa.npz" 
    np.savez(
        submission_npz_path, 
        embeddings=translated_embeddings_clean, 
        ids=sample_ids
    )
    print(f"Submission embeddings for ensemble saved.")

    # Save final .csv submission
    submission_path = Path(submission_dir) / "submission_rmlpa.csv" 
    print(f"Calculating similarity and saving submission...")
    
    generate_submission(
        sample_ids,                     
        translated_embeddings_clean,    
        output_file=str(submission_path)  
    )

    print("Submission created successfully!")