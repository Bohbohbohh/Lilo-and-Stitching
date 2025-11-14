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
    def __init__(self, D_in: int, D_out: int, 
                 D_bottle_ratio: int = 4, dropout_p: float = 0.1):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        D_bottle = D_in // D_bottle_ratio

        self.block = nn.Sequential(
            nn.Linear(D_in, D_bottle),
            nn.GELU(),  
            nn.Dropout(dropout_p),
            nn.Linear(D_bottle, D_out)
        )

        if D_in == D_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(D_in, D_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_out = self.block(x)
        shortcut_out = self.shortcut(x)
        
        output = block_out + shortcut_out
        
        output_normalized = F.normalize(output, p=2, dim=1)
        
        return output_normalized

def normalize_l2(embeddings):
    return F.normalize(embeddings, p=2, dim=-1)

class PairedEmbeddingDataset(Dataset):
    def __init__(self, z_text, z_img):
        self.z_text = z_text
        self.z_img = z_img
        assert len(self.z_text) == len(self.z_img), "Dataset size mismatch"

    def __len__(self):
        return len(self.z_text)

    def __getitem__(self, idx):
        return self.z_text[idx], self.z_img[idx]
    
def create_dataset_from_indices(
    indices, 
    z_text_all, 
    z_img_unique, 
    text_to_image_map
):
    if len(indices) == 0:
        return None 

    z_text_np = z_text_all[indices]
    img_indices_map = text_to_image_map[indices]
    z_img_np = z_img_unique[img_indices_map]
    
    z_text_tensor = normalize_l2(torch.from_numpy(z_text_np).float())
    z_img_tensor  = normalize_l2(torch.from_numpy(z_img_np).float())
    
    return PairedEmbeddingDataset(z_text_tensor, z_img_tensor)

def setup_environment(seed, device_str="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and device_str == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- Using device: {device} ---")
    return device

def load_and_prepare_data(train_data_path, submission_dir, device):
    print("Loading Training Data (with prepare_train_data)...")
    train_data_dict = load_data(train_data_path)
    z_text_raw, z_img_raw, label_matrix = prepare_train_data(train_data_dict)

    print(f"Total samples: {len(z_text_raw)}")
    print(f"Text Dimension: {z_text_raw.shape[1]} | Image Dimension: {z_img_raw.shape[1]}")

    X_FINAL = z_text_raw.float().to(device) 
    Y_FINAL = z_img_raw.float().to(device) 
    print("Data moved to device.")

    groups = np.argmax(label_matrix, axis=1)

    print("\nCreating Unique Global Gallery...")
    Y_FINAL_np = normalize_l2(Y_FINAL).cpu().numpy()
    all_unique_group_ids, all_unique_indices = np.unique(groups, return_index=True)

    Y_gallery_unique_ALL = Y_FINAL_np[all_unique_indices] 
    groups_gallery_unique_ALL = all_unique_group_ids
    print(f"Global Gallery created with {len(Y_gallery_unique_ALL)} unique images.")

    gallery_npz_path = Path(submission_dir) / "gallery_data.npz"
    np.savez(
        gallery_npz_path, 
        embeddings=Y_gallery_unique_ALL, 
        groups=groups_gallery_unique_ALL
    )
    print(f"Gallery saved for tuning in: {gallery_npz_path}")
    
    return X_FINAL, Y_FINAL, groups, Y_gallery_unique_ALL, groups_gallery_unique_ALL

def create_train_val_test_splits(X_FINAL, groups, temp_split_ratio, test_split_ratio_of_temp, seed):
    print("Split 1: Creating Train (90%) and Temp (10%) set...")
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=temp_split_ratio, random_state=seed)
    train_indices, temp_indices = next(gss_train_temp.split(X_FINAL.cpu().numpy(), y=None, groups=groups))

    print("Split 2: Splitting Temp set into Validation (5%) and Test (5%)...")
    groups_temp = groups[temp_indices]
    X_temp_dummy = np.empty((len(groups_temp), 1)) 
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=test_split_ratio_of_temp, random_state=seed)
    val_indices_rel, test_indices_rel = next(gss_val_test.split(X_temp_dummy, y=None, groups=groups_temp))

    val_indices = temp_indices[val_indices_rel]
    test_indices = temp_indices[test_indices_rel]
    groups_val = groups[val_indices] 
    groups_test = groups[test_indices]
    
    print(f"Split completed: Train ({len(train_indices)}), Val ({len(val_indices)}), Test ({len(test_indices)})")
    
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
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    print("DataLoaders (train, val, test) ready.")
    return train_loader, val_loader, test_loader

def setup_model_and_optimizer(
    dim_roberta, dim_dino, d_bottle_ratio, dropout_p, 
    init_temperature, learning_rate, weight_decay, 
    num_epochs, train_loader_len, device
):
    print("Initializing model, loss, and optimizer...")
    adapter = ResidualBottleneckAdapter(
        D_in=dim_roberta, 
        D_out=dim_dino,
        D_bottle_ratio=d_bottle_ratio,
        dropout_p=dropout_p
    ).to(device)
    temperature = nn.Parameter(torch.tensor(init_temperature)).to(device)
    all_parameters = list(adapter.parameters()) + [temperature]
    optimizer = optim.AdamW(
        all_parameters, 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * train_loader_len, 
        eta_min=1e-6
    )
    print(" Setup complete. Ready for training.")
    return adapter, temperature, optimizer, scheduler

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
    pbar_val = tqdm(range(N_queries), desc="[Validation] MRR calculation (1+99)", leave=False, disable=True)

    for i in pbar_val:
        query_vec = X_queries_proj_norm[i:i+1] 
        true_query_group_id = groups_val[i]
        
        correct_gallery_index = np.where(groups_gallery_unique_ALL == true_query_group_id)[0][0]
        
        is_negative = (all_gallery_indices != correct_gallery_index)
        negative_indices_pool = all_gallery_indices[is_negative]
        
        sampled_negative_indices = np.random.choice(
            negative_indices_pool, 
            n_samples, 
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

@torch.no_grad()
def run_inference(data_loader, adapter, device):
    adapter.eval()
    all_translated_embeddings = []
    
    for batch in tqdm(data_loader, desc="[Validation] Execution R-MLP-A", leave=False):
        z_text = batch[0].to(device)
        z_final_norm = adapter(z_text)
        all_translated_embeddings.append(z_final_norm.cpu())
        
    return torch.cat(all_translated_embeddings, dim=0).numpy()

def train_loop(
    num_epochs, adapter, train_loader, val_loader, 
    optimizer, scheduler, temperature, device,
    groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL,
    patience, checkpoint_path
):
    best_mrr = -1.0
    epochs_no_improve = 0

    print(f"    Starting Training (InfoNCE Loss)")
    for epoch in range(num_epochs):
        
        adapter.train()
        running_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for z_text, z_img in pbar_train:
            z_text, z_img = z_text.to(device), z_img.to(device)
            labels = torch.arange(z_text.shape[0], device=device)
            mapped_text = adapter(z_text) 
            norm_vae = F.normalize(z_img, p=2, dim=1) 
            logits = torch.matmul(mapped_text, norm_vae.T)
            logits = logits * temperature.exp()
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
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
            print(f"--- Early Stopping: MRR has not improved for {patience} epochs. ---")
            break

    print(f"Training completed. Best MRR: {best_mrr:.4f} ---")
    return best_mrr

def run_verification(
    adapter, checkpoint_path, test_loader, val_loader, device,
    groups_test, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL,
    submission_dir
):
    print("\n--- Starting Verification on Internal Test Set ---")

    if not Path(checkpoint_path).exists():
        print("Error: Checkpoint not found. Cannot verify.")
        return

    print(f"Loading best model from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except:
         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    adapter.load_state_dict(checkpoint['adapter_state_dict'])

    X_queries_proj_test = run_inference(test_loader, adapter, device)
    X_queries_proj_test = np.nan_to_num(X_queries_proj_test, nan=0.0, posinf=1e6, neginf=-1e6)

    test_mrr = calculate_mrr_validation_sampled(
        X_queries_proj=X_queries_proj_test,
        groups_val=groups_test, 
        Y_gallery_unique_ALL=Y_gallery_unique_ALL,
        groups_gallery_unique_ALL=groups_gallery_unique_ALL
    )
    print(f"--- MRR on Internal Test Set: {test_mrr:.4f} ---")

    print(f"\nGenerating VALIDATION embeddings for alpha tuning...")
    
    val_embeddings = run_inference(val_loader, adapter, device)
    val_embeddings_clean = np.nan_to_num(val_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    
    val_npz_path = Path(submission_dir) / "val_rmlpa.npz" 
    np.savez(
        val_npz_path, 
        embeddings=val_embeddings_clean, 
        groups=groups_val 
    )
    print(f"Validation embeddings saved to: {val_npz_path}")

@torch.no_grad()
def run_submission_inference(z_text_tensor, adapter, device, batch_size=512):
    adapter.eval()
    
    all_translated = []
    num_samples = z_text_tensor.shape[0]
    
    desc = "[Submission] Execution R-MLP-A"
    for i in tqdm(range(0, num_samples, batch_size), desc=desc):
        
        batch_z_text = z_text_tensor[i:i+batch_size].to(device)
        
        z_final_norm = adapter(batch_z_text)
        all_translated.append(z_final_norm.cpu())
        
    return torch.cat(all_translated, dim=0).numpy()

def generate_submission_files(
    adapter, checkpoint_path, test_data_path, 
    submission_dir, device, batch_size
):
    print("\n--- Starting Submission File Generation ---")
    
    if not Path(checkpoint_path).exists():
        print("Error: Checkpoint not found. Cannot generate submission.")
        return

    if not next(adapter.parameters()).is_cuda: 
        try:
            print(f"Loading model from {checkpoint_path} for submission...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except:
             checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        adapter.load_state_dict(checkpoint['adapter_state_dict'])

    print(f"Loading test data from {test_data_path}...")
    test_data_clean = load_data(test_data_path)

    z_text_test_raw = test_data_clean['captions/embeddings'] 
    sample_ids = test_data_clean['captions/ids']
    
    z_text_test_tensor = torch.from_numpy(z_text_test_raw).float()
    print(f"Test data loaded: {z_text_test_tensor.shape}")
    
    print("Running inference on submission data (test.clean.npz)...")
    translated_embeddings = run_submission_inference(
        z_text_test_tensor, 
        adapter,
        device,
        batch_size=batch_size
    )
    
    translated_embeddings_clean = np.nan_to_num(translated_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    
    submission_npz_path = Path(submission_dir) / "submission_rmlpa.npz" 
    np.savez(
        submission_npz_path, 
        embeddings=translated_embeddings_clean, 
        ids=sample_ids
    )
    print(f"Submission embeddings for ensemble saved to: {submission_npz_path}")

    submission_path = Path(submission_dir) / "submission_rmlpa.csv" 
    print(f"Calculating similarity and saving submission to {submission_path}...")
    
    generate_submission(
        sample_ids,                     
        translated_embeddings_clean,    
        output_file=str(submission_path)  
    )

    print("    Submission created successfully!")