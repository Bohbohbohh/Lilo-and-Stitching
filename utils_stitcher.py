import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

from challenge.src.common.utils import generate_submission, load_data, prepare_train_data

class Stitcher(nn.Module):
    """
    The "Stitcher" model maps text embeddings (input_dim) to the
    image embedding space (output_dim) using a parallel architecture.
    It combines a simple linear transformation with a non-linear MLP path,
    acting as a form of residual block.
    """
    def __init__(self, input_dim=1024, output_dim=1536, hidden_dim=2048, dropout_p=0.5):
        super().__init__()
        # A direct linear projection
        self.linear_map = nn.Linear(input_dim, output_dim, bias=False)

        # A non-linear mapping via a deep MLP
        self.mlp_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        # Combine the linear and non-linear paths
        prediction = self.linear_map(x) + self.mlp_map(x)

        # Normalize the final output for cosine similarity-based loss
        return F.normalize(prediction, p=2, dim=1)

def train_model_stitcher(model, train_loader, val_loader, DEVICE, EPOCHS, LR, MARGIN, MODEL_PATH, PATIENCE, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL):
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS * len(train_loader), 
        eta_min=1e-7
    )
    
    best_mrr = -1.0
    epochs_no_improve = 0

    print(f"Starting Stitcher training for {EPOCHS} epochs...")
    print(f"Saving checkpoint in: {MODEL_PATH}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for batch_X_norm, batch_Y_norm in pbar_train:
            batch_X_norm = batch_X_norm.to(DEVICE)
            batch_Y_norm = batch_Y_norm.to(DEVICE)

            anchor = model(batch_X_norm) # anchor: Mapped text embedding 
            positive = batch_Y_norm # positive: Ground truth image embedding

            # Calculate all-to-all similarity
            sim_matrix = torch.matmul(anchor, positive.T)
            positive_sim = torch.diag(sim_matrix)
            
            # Clone the matrix
            sim_matrix_clone = sim_matrix.clone()

            # Mask out the diagonal similarities
            sim_matrix_clone.fill_diagonal_(float('-inf'))

            # Find the max similarity for each anchor to all negatives
            hard_negative_sim = torch.max(sim_matrix_clone, dim=1).values

            # Triplet Loss
            loss_components = MARGIN - positive_sim + hard_negative_sim
            loss_components = torch.clamp(loss_components, min=0.0)
            loss = loss_components.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar_train.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval() 
        all_X_queries_proj = []
        with torch.no_grad():
            for batch_X_norm, _ in val_loader: 
                batch_X_norm = batch_X_norm.to(DEVICE)
                anchor_norm = model(batch_X_norm)
                all_X_queries_proj.append(anchor_norm.cpu().numpy())
        
        X_queries_proj_val = np.concatenate(all_X_queries_proj, axis=0)
        X_queries_proj_val = np.nan_to_num(X_queries_proj_val, nan=0.0, posinf=1e6, neginf=-1e6)

        current_mrr = calculate_mrr_validation_sampled(
            X_queries_proj=X_queries_proj_val,
            groups_val=groups_val,
            Y_gallery_unique_ALL=Y_gallery_unique_ALL,
            groups_gallery_unique_ALL=groups_gallery_unique_ALL
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Train Loss (Triplet): {avg_train_loss:.6f} | Val MRR: {current_mrr:.6f} | LR: {current_lr:1.1e}")

        if current_mrr > best_mrr:
            best_mrr = current_mrr
            epochs_no_improve = 0
            print(f" -> New, best MRR! Saving the model to {MODEL_PATH}")
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            print(f"  -> MRR not improved ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"--- Early Stopping: MRR hasn't improved in {PATIENCE} epochs. ---")
            break

    print(f"\nTraining completed. The best model has been saved at {MODEL_PATH} with Val MRR: {best_mrr:.6f}")
    return model

@torch.no_grad()
def run_submission_inference_stitcher(model, test_embds_np, DEVICE, BATCH_SIZE_TEST=1024):
    """Runs inference on the final test data with manual batching."""
    model.eval()

    # Ensure input embeddings are L2 normalized, as expected by the model
    test_embds_norm_np = normalize(test_embds_np, norm='l2', axis=1)
    test_embds_norm = torch.tensor(test_embds_norm_np, dtype=torch.float32)

    pred_embds_list = []

    # Manual batching loop
    for i in tqdm(range(0, len(test_embds_norm), BATCH_SIZE_TEST), desc="[Submission] Stitcher execution"):
        batch_X_norm = test_embds_norm[i:i+BATCH_SIZE_TEST].to(DEVICE)
        batch_final = model(batch_X_norm)
        pred_embds_list.append(batch_final.cpu())

    pred_embds_final = torch.cat(pred_embds_list, dim=0)
    return pred_embds_final.numpy()

@torch.no_grad()
def run_validation_inference_stitcher(model, val_loader, DEVICE):
    """Helper function to run inference on a validation dataloader."""
    model.eval()
    all_translated_embeddings = []
    
    pbar = tqdm(val_loader, desc="[Validation] Stitcher execution")
    for batch_X_norm, _ in pbar: 
        batch_X_norm = batch_X_norm.to(DEVICE)
        anchor_norm = model(batch_X_norm)
        all_translated_embeddings.append(anchor_norm.cpu())
        
    return torch.cat(all_translated_embeddings, dim=0).numpy()

def calculate_mrr_validation_sampled(X_queries_proj, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL, n_samples=99):
    """
    Calculates Mean Reciprocal Rank (MRR) using a sampled gallery
    (1 positive + 99 random negatives) for efficiency.
    """
    if X_queries_proj.shape[0] == 0:
        return 0.0

    N_queries = X_queries_proj.shape[0]
    N_gallery = Y_gallery_unique_ALL.shape[0]

    # Pre-normalize all vectors for efficient dot product
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
        
        # Create a pool of all negative indices
        is_negative = (all_gallery_indices != correct_gallery_index)
        negative_indices_pool = all_gallery_indices[is_negative]
        
        # Randomly sample 'n_samples' (99) from the negative pool
        sampled_negative_indices = np.random.choice(
            negative_indices_pool, 
            n_samples, 
            replace=False
        )
        
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
        
        mrr_sum += 1.0 / (rank_zero_based + 1)
            
    return mrr_sum / N_queries

def setup_environment_stitcher(seed, device_str="cuda"):
    """Sets random seeds for reproducibility and selects device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and device_str == "cuda":
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")
    print(f"--- Using device: {device} ---")
    return device

def load_and_prepare_data_stitcher(train_data_path, device):
    print("Loading training data...")
    train_data_dict = load_data(train_data_path)
    z_text_raw, z_img_raw, label_matrix = prepare_train_data(train_data_dict)
    print(f"Training data loaded.")

    X_FINAL = F.normalize(z_text_raw.float(), p=2, dim=1).to(device)
    Y_FINAL = F.normalize(z_img_raw.float(), p=2, dim=1).to(device)

    groups = torch.argmax(label_matrix.float(), axis=1).cpu().numpy()
    return X_FINAL, Y_FINAL, groups

def create_splits_stitcher(X_FINAL, groups, temp_split_ratio, test_split_ratio_of_temp, seed):
    """
    Performs a 2-stage grouped split to create train, validation, and test sets.
    This ensures that samples from the same group do not
    appear in different splits, preventing data leakage.
    """
    print("\nPreparing data...")
    # Split 1: 100% -> 90% Train vs 10% Temp
    print("Split 1: Creating Training set (90%) and Temporary set (10%) set...")
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=temp_split_ratio, random_state=seed)
    train_indices, temp_indices = next(gss_train_temp.split(X_FINAL.cpu().numpy(), y=None, groups=groups))

    # Split 2: 10% Temp -> 5% Validation vs 5% Test
    print("Split 2: Splitting Temporary set into Validation set (5%) and Test set (5%)...")
    groups_temp = groups[temp_indices]
    X_temp_dummy = np.empty((len(groups_temp), 1)) 
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=test_split_ratio_of_temp, random_state=seed)
    val_indices_rel, test_indices_rel = next(gss_val_test.split(X_temp_dummy, y=None, groups=groups_temp))

    # Convert relative indices back to absolute indices
    val_indices = temp_indices[val_indices_rel]
    test_indices = temp_indices[test_indices_rel]
    groups_val = groups[val_indices] 
    groups_test = groups[test_indices]
    
    print(f"Split completed:")
    print(f"    Training set: {len(train_indices)}")
    print(f"    Validation set: {len(val_indices)}")
    print(f"    Test set: {len(test_indices)}")
    return train_indices, val_indices, test_indices, groups_val, groups_test

def load_global_gallery(submission_dir, gallery_file="gallery_data.npz"):
    """
    Loads the pre-computed global gallery .npz file.
    """
    gallery_npz_path = Path(submission_dir) / gallery_file
    if not gallery_npz_path.exists():
        print(f"ERROR: Global gallery '{gallery_file}' not found.")
        exit()
        
    gallery_data = np.load(gallery_npz_path)
    Y_gallery_unique_ALL = gallery_data['embeddings']
    groups_gallery_unique_ALL = gallery_data['groups']
    print(f"Global gallery loaded.")
    return Y_gallery_unique_ALL, groups_gallery_unique_ALL

def create_dataloaders_stitcher(X_FINAL, Y_FINAL, train_indices, val_indices, groups, batch_size, num_workers):
    print("Creating Datasets and DataLoaders...")
    
    # Training Loader
    train_dataset = TensorDataset(X_FINAL[train_indices], Y_FINAL[train_indices])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=False, drop_last=True
    )

    # Validation Loader
    groups_val = groups[val_indices]
    groups_val_unique, val_unique_indices_relative = np.unique(groups_val, return_index=True)
    
    groups_gallery_val_unique = groups_val_unique 
    
    group_to_gallery_idx = {group_id: idx for idx, group_id in enumerate(groups_gallery_val_unique)}
    val_label_indices = [group_to_gallery_idx[gid] for gid in groups_val]
    val_label_indices_t = torch.tensor(val_label_indices, dtype=torch.long)

    val_dataset_train = TensorDataset(X_FINAL[val_indices], val_label_indices_t)
    val_loader_train = DataLoader(
        val_dataset_train, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    # Validation Loader
    val_dataset_inf = TensorDataset(X_FINAL[val_indices], Y_FINAL[val_indices])
    val_loader_inf = DataLoader(
        val_dataset_inf, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    print("DataLoaders loaded and ready.")
    return train_loader, val_loader_train, val_loader_inf

def setup_model_stitcher(input_dim, output_dim, hidden_dim, dropout_p, device):
    """Initializes the Stitcher model and moves it to the device."""
    print("Initializing model...")
    model = Stitcher(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=hidden_dim, 
        dropout_p=dropout_p
    ).to(device)
    print(f"    Stitcher parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def run_verification_stitcher(model, checkpoint_path, val_loader_inf, X_test_tensor, groups_test, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL, submission_dir, device):
    """
    Loads the best model to:
    1. Save validation embeddings for ensemble tuning.
    2. Report MRR on the internal test set.
    """
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}. Impossible to continue.")
        exit()
        
    print(f"Loading best model...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Run inference on validation set and save .npz for ensembling
    val_embeddings = run_validation_inference_stitcher(model, val_loader_inf, device)
    
    val_npz_path = Path(submission_dir) / "val_stitcher.npz" 
    np.savez(
        val_npz_path, 
        embeddings=val_embeddings, 
        groups=groups_val 
    )
    
    with torch.no_grad():
        test_internal_embeddings = model(X_test_tensor).cpu().numpy()

    test_mrr = calculate_mrr_validation_sampled(
        X_queries_proj=test_internal_embeddings,
        groups_val=groups_test, 
        Y_gallery_unique_ALL=Y_gallery_unique_ALL,
        groups_gallery_unique_ALL=groups_gallery_unique_ALL
    )
    print(f"MRR on Internal Test Set: {test_mrr:.6f}")

def generate_submission_files_stitcher(model, checkpoint_path, test_data_path, submission_dir, device, batch_size):    
    """
    Loads the best model and runs inference on the official test data.
    Saves both an .npz (for ensembling) and a .csv (for submission).
    """
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"Loading Test data...")
    test_data_clean = load_data(test_data_path)

    z_text_test_raw_np = test_data_clean['captions/embeddings'] 
    sample_ids = test_data_clean['captions/ids']
    print(f"Test data loaded.")
    
    translated_embeddings = run_submission_inference_stitcher(
        model, 
        z_text_test_raw_np, 
        device,
        BATCH_SIZE_TEST=batch_size
    )
    
    # Save .npz file for the ensemble
    submission_npz_path = Path(submission_dir) / "submission_stitcher.npz" 
    np.savez(
        submission_npz_path, 
        embeddings=translated_embeddings, 
        ids=sample_ids
    )
    print(f"Embeddings for ensemble saved.")

    # Save the final .csv submission file
    submission_path = Path(submission_dir) / "submission_stitcher.csv"
    
    generate_submission(
        sample_ids,                     
        translated_embeddings,    
        output_file=str(submission_path)  
    )

    print(f"Submission successfully created!")