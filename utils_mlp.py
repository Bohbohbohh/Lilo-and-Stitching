import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import normalize
import os 
from pathlib import Path 
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from challenge.src.common import generate_submission, load_data, prepare_train_data

class LatentMapper(nn.Module):
    def __init__(self, D_in=1536, D_out=1536, R_init=None, bias_init=None, mu_x=None, DEVICE=torch.device("cpu"), dropout_rate=0.25):
        super(LatentMapper, self).__init__()
        
        self.mu_x = torch.from_numpy(mu_x).float().to(DEVICE)
        
        self.fc1 = nn.Linear(D_in, D_out, bias=True)
        
        if R_init is not None and bias_init is not None:
            self.fc1.weight.data.copy_(torch.from_numpy(R_init.T).float())
            self.fc1.bias.data.copy_(torch.from_numpy(bias_init).float())
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_centered = x - self.mu_x
        
        x = self.fc1(x_centered) 
        x = self.dropout(x)

        x = F.normalize(x, p=2, dim=1)
        return x
    
class TripletDataset(Dataset):
    def __init__(self, X_padded_tensor, Y_target_tensor):
        self.X_padded = X_padded_tensor
        self.Y_target = Y_target_tensor

    def __len__(self):
        return len(self.X_padded)

    def __getitem__(self, idx):
        A = self.X_padded[idx] 
        P = self.Y_target[idx] 
        return A, P

class CustomTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CustomTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = lambda x, y: 1.0 - F.cosine_similarity(x, y)

    def forward(self, anchor, positive, negative):
        dist_pos = self.distance_metric(anchor, positive) 
        dist_neg = self.distance_metric(anchor, negative)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()

def setup_paths_and_device(base_dir_path, checkpoint_dir_name, submission_dir_name, model_file_name):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f"--- Using device: {DEVICE} ---")
    
    BASE_DIR = Path(base_dir_path)
    DATA_DIR = BASE_DIR / 'data' 
    SUBMISSION_DIR = BASE_DIR / submission_dir_name
    CHECKPOINT_DIR = BASE_DIR / checkpoint_dir_name
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    FINAL_MODEL_PATH = str(CHECKPOINT_DIR / model_file_name)
    
    return DEVICE, BASE_DIR, DATA_DIR, SUBMISSION_DIR, CHECKPOINT_DIR, FINAL_MODEL_PATH

def load_and_prepare_data_mlp(
    train_data_path, temp_split_ratio, 
    test_split_ratio_of_temp, seed, device
):
    print("Loading Training data...")
    train_data_dict = load_data(train_data_path)
    X_train_raw, y_train_raw, label_matrix = prepare_train_data(train_data_dict)

    N_samples = len(X_train_raw)
    D_text = X_train_raw.shape[1]
    D_vae = y_train_raw.shape[1]

    print("Training data loaded.")

    X_FINAL = F.normalize(X_train_raw.to(device), p=2, dim=1)
    Y_FINAL = F.normalize(y_train_raw.to(device), p=2, dim=1)

    PADDING_SIZE = D_vae - D_text
    groups = np.argmax(label_matrix, axis=1)

    print("\nPreparing data...")
    print("Split 1: Creating Training set (90%) and Temporary set (10%) set...")
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=temp_split_ratio, random_state=seed)
    train_indices, temp_indices = next(gss_train_temp.split(X_FINAL.cpu().numpy(), y=None, groups=groups))

    print("Split 2: Splitting Temporary set into Validation set (5%) and Test set (5%)...")
    groups_temp = groups[temp_indices]
    X_temp_dummy = np.empty((len(groups_temp), 1)) 
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=test_split_ratio_of_temp, random_state=seed)
    val_indices_rel, test_indices_rel = next(gss_val_test.split(X_temp_dummy, y=None, groups=groups_temp))

    val_indices = temp_indices[val_indices_rel]
    test_indices = temp_indices[test_indices_rel]

    train_idx = train_indices
    val_idx = val_indices
    groups_val = groups[val_indices]
    groups_test = groups[test_indices] 

    print(f"Split completed:")
    print(f"    Training set: {len(train_indices)}")
    print(f"    Validation set: {len(val_indices)}")
    print(f"    Test set: {len(test_indices)}")

    X_train_split = X_FINAL[train_idx]
    Y_train_split = Y_FINAL[train_idx]
    X_val_split = X_FINAL[val_idx]
    Y_val_split = Y_FINAL[val_idx]
    X_test_split = X_FINAL[test_indices]
    Y_test_split = Y_FINAL[test_indices]

    X_train_np = X_train_split.cpu().numpy()
    Y_train_np = Y_train_split.cpu().numpy()
    X_val_np = X_val_split.cpu().numpy()
    Y_val_np = Y_val_split.cpu().numpy()
    X_test_split_np = X_test_split.cpu().numpy()
    Y_test_split_np = Y_test_split.cpu().numpy()

    X_train_P = np.pad(X_train_np, ((0, 0), (0, PADDING_SIZE)), 'constant')
    X_val_P = np.pad(X_val_np, ((0, 0), (0, PADDING_SIZE)), 'constant') 
    X_test_P = np.pad(X_test_split_np, ((0, 0), (0, PADDING_SIZE)), 'constant')
    
    Y_train_P = Y_train_np      
    Y_val_P = Y_val_np
    
    mu_x_train = X_train_P.mean(axis=0)
    mu_y_train = Y_train_P.mean(axis=0)
    X_train_P_centered = X_train_P - mu_x_train
    Y_train_P_centered = Y_train_P - mu_y_train

    R_train, _ = orthogonal_procrustes(X_train_P_centered, Y_train_P_centered)
    bias_train = mu_y_train - (mu_x_train @ R_train)
    print(f"R, bias, mu_x calculated.")

    X_train_P_tensor = torch.from_numpy(X_train_P).float().to(device)
    Y_train_P_tensor = torch.from_numpy(Y_train_P).float().to(device) 
    X_val_tensor = torch.from_numpy(X_val_P).float().to(device)
    Y_val_tensor = torch.from_numpy(Y_val_P).float().to(device)
    X_test_P_tensor = torch.from_numpy(X_test_P).float().to(device)

    init_stats = {
        "R_init": R_train,
        "bias_init": bias_train,
        "mu_x": mu_x_train
    }
    
    data_pack = {
        "train": (X_train_P_tensor, Y_train_P_tensor),
        "val": (X_val_tensor, Y_val_tensor),
        "test": (X_test_P_tensor, Y_test_split_np),
        "groups_val": groups_val,
        "groups_test": groups_test,
        "init_stats": init_stats,
        "PADDING_SIZE": PADDING_SIZE,
        "D_vae": D_vae
    }
    return data_pack

def load_submission_test_data(test_data_path):
    print("\nLoading Test data...")
    test_data = load_data(test_data_path)
    X_test_np = test_data['captions/embeddings'] 
    test_data_ids = test_data['captions/ids']
    print("Test data loaded.")
    return X_test_np, test_data_ids

def setup_model_and_optimizer_mlp(
    D_vae, init_stats, n_epochs, train_loader_len, device,
    lr=1e-5, wd=1e-3, margin=0.2, eta_min=1e-7
):
    print("\nDML model initialisation...")
    model = LatentMapper(
        D_in=D_vae, 
        D_out=D_vae, 
        R_init=init_stats["R_init"],     
        bias_init=init_stats["bias_init"], 
        mu_x=init_stats["mu_x"],
        DEVICE=device
    ).to(device)
    
    triplet_loss = CustomTripletLoss(margin=margin)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    total_steps = n_epochs * train_loader_len
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    
    print(f"Loss: Triplet Loss (Margin={margin}) | Optimizer: AdamW (lr={lr})")
    return model, triplet_loss, optimizer, scheduler

def find_hardest_negative_in_batch(A_batch, P_batch):
    similarity_matrix = torch.matmul(A_batch, P_batch.T)

    hardest_negatives = []
    
    for i in range(A_batch.shape[0]):
        neg_sims = similarity_matrix[i, :].clone()
        neg_sims[i] = -float('inf') 
        
        hard_neg_idx_local = torch.argmax(neg_sims)
        
        hard_neg_vector = P_batch[hard_neg_idx_local]
        hardest_negatives.append(hard_neg_vector)
        
    return torch.stack(hardest_negatives)

def calculate_mrr_validation_sampled(
    X_queries_proj, 
    groups_val, 
    Y_gallery_unique_ALL, 
    groups_gallery_unique_ALL,
    n_samples=99 
):
    if X_queries_proj.shape[0] == 0:
        return 0.0

    N_queries = X_queries_proj.shape[0]
    N_gallery = Y_gallery_unique_ALL.shape[0]

    Y_gallery_unique_ALL_norm = normalize(Y_gallery_unique_ALL, axis=1)
    X_queries_proj_norm = normalize(X_queries_proj, axis=1)

    all_gallery_indices = np.arange(N_gallery)
    
    mrr_sum = 0
    disable_pbar = N_queries < 100 
    pbar_val = tqdm(range(N_queries), desc="[Validation] MRR calculation (1+99)", disable=disable_pbar, leave=False)

    for i in pbar_val:
        query_vec = X_queries_proj_norm[i:i+1] 
        true_query_group_id = groups_val[i]
        
        correct_gallery_index_arr = np.where(groups_gallery_unique_ALL == true_query_group_id)[0]
        
        if len(correct_gallery_index_arr) == 0:
            continue 
        correct_gallery_index = correct_gallery_index_arr[0]
        
        is_negative = (all_gallery_indices != correct_gallery_index)
        negative_indices_pool = all_gallery_indices[is_negative]
        
        n_samples_clamped = min(n_samples, len(negative_indices_pool))
        if n_samples_clamped == 0:
            continue 

        sampled_negative_indices = np.random.choice(
            negative_indices_pool, 
            n_samples_clamped, 
            replace=False
        )
        
        candidate_indices = np.concatenate(
            ([correct_gallery_index], sampled_negative_indices)
        )
        
        candidate_gallery_vecs = Y_gallery_unique_ALL_norm[candidate_indices] 
        
        similarity_scores = np.dot(query_vec, candidate_gallery_vecs.T)[0]
        
        ranked_indices = np.argsort(similarity_scores)[::-1]
        rank_zero_based = np.where(ranked_indices == 0)[0][0]
        
        mrr_sum += 1.0 / (rank_zero_based + 1)
            
    return mrr_sum / N_queries

def run_training_loop(model, train_loader, optimizer, scheduler, triplet_loss, N_EPOCHS, FINAL_MODEL_PATH, X_val, Y_val, groups_val, PATIENCE, DEVICE):
    best_val_mrr = -1.0
    patience_counter = 0
    Y_val_np = Y_val.cpu().numpy()
    
    print(f"\nStart Training for {N_EPOCHS} epochs with Early Stopping (Patience={PATIENCE})...")
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]")
        
        for A_batch_input, P_batch_target in pbar:

            sigma = 0.01 
            noise = torch.randn_like(A_batch_input) * sigma
            A_batch_input = A_batch_input + noise
            
            A_proj = model(A_batch_input)
            
            N_hardest = find_hardest_negative_in_batch(A_proj, P_batch_target)
            
            loss = triplet_loss(A_proj, P_batch_target, N_hardest)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix(Loss=loss.item())

        avg_loss = total_loss / len(train_loader)
                
        model.eval()
        X_val_proj_np = None
        with torch.no_grad():
            X_val_proj_np = model(X_val).cpu().numpy()
            
        current_val_mrr = calculate_mrr_validation_sampled(
            X_queries_proj=X_val_proj_np, 
            groups_val=groups_val, 
            Y_gallery_unique_ALL=Y_val_np, 
            groups_gallery_unique_ALL=groups_val,
            n_samples=99 
        )
        
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Train Loss: {avg_loss:.6f} | Val MRR: {current_val_mrr:.6f}")

        if current_val_mrr > best_val_mrr:
            best_val_mrr = current_val_mrr
            patience_counter = 0
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f"New Best Model Saved! Val MRR: {best_val_mrr:.6f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. MRR does not improve after {PATIENCE} epochs.")
            break
        
    print(f"\nTraining completed.")
    print(f"Best model saved  with MRR Val: {best_val_mrr:.6f}")

def save_validation_embeddings(model, X_val_tensor, groups_val, submission_dir):
    with torch.no_grad():
        val_embeddings = model(X_val_tensor).cpu().numpy()

    val_npz_path = submission_dir / "val_mlp.npz" 
    np.savez(
        val_npz_path, 
        embeddings=val_embeddings, 
        groups=groups_val 
    )

def run_internal_test(model, X_test_P_tensor, Y_test_split_np, groups_test, device):
    with torch.no_grad():
        test_internal_embeddings = model(X_test_P_tensor).cpu().numpy() 

    test_mrr = calculate_mrr_validation_sampled(
        X_queries_proj=test_internal_embeddings, 
        groups_val=groups_test,                  
        Y_gallery_unique_ALL=Y_test_split_np,    
        groups_gallery_unique_ALL=groups_test    
    )
    print(f"MRR on Internal Test Set: {test_mrr:.6f}")

def generate_dml_submission(model, FINAL_MODEL_PATH, X_test_np, test_data_ids, PADDING_SIZE, BASE_DIR, submission_suffix="mlp", DEVICE=torch.device("cpu")):
    print(f"\nGenerating Submission ({submission_suffix})...")
    
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    X_test_np_padded = np.pad(X_test_np, ((0, 0), (0, PADDING_SIZE)), 'constant')
    X_test_tensor_final = torch.from_numpy(X_test_np_padded).float().to(DEVICE)
    
    with torch.no_grad():
        Y_pred_dml_np = model(X_test_tensor_final).cpu().numpy()
    
    SUBMISSION_DIR = BASE_DIR / "submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    try:
        submission_mlp_path = SUBMISSION_DIR / "submission_mlp.npz" 
        np.savez(
            submission_mlp_path, 
            embeddings=Y_pred_dml_np, 
            ids=test_data_ids
        ) 
        print(f"Embedding for ensembles saved.")
    except Exception as e:
        print(f"WARNING: Error saving .npz file for ensemble: {e}")

    submission_path_dml = SUBMISSION_DIR / f'submission_{submission_suffix}.csv'
    generate_submission(test_data_ids, Y_pred_dml_np, str(submission_path_dml))
    
    print(f"DML submission saved.")

