import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

# Import dalle utility del challenge (necessarie per le funzioni spostate qui)
from challenge.src.common.utils import generate_submission, load_data, prepare_train_data

# ----------------------------------------------------------------------------
# DEFINIZIONI CORE (Modello, Training, Inferenza)
# ----------------------------------------------------------------------------

class Stitcher(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1536, hidden_dim=2048, dropout_p=0.5):
        super().__init__()
        self.linear_map = nn.Linear(input_dim, output_dim, bias=False)
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
        prediction = self.linear_map(x) + self.mlp_map(x)
        return F.normalize(prediction, p=2, dim=1)

def train_model_stitcher(model, train_loader, val_loader, DEVICE, EPOCHS, LR, MARGIN, MODEL_PATH, PATIENCE,
                         groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL):
    """ Funzione di training (già in utils) """
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS * len(train_loader), 
        eta_min=1e-7
    )
    
    best_mrr = -1.0
    epochs_no_improve = 0

    print(f"Inizio training Stitcher E2E (obiettivo: MRR Globale) per {EPOCHS} epoche...")
    print(f"Salvataggio checkpoint in: {MODEL_PATH}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for batch_X_norm, batch_Y_norm in pbar_train:
            batch_X_norm = batch_X_norm.to(DEVICE)
            batch_Y_norm = batch_Y_norm.to(DEVICE)

            anchor = model(batch_X_norm) 
            positive = batch_Y_norm
            sim_matrix = torch.matmul(anchor, positive.T)
            positive_sim = torch.diag(sim_matrix)
            
            sim_matrix_clone = sim_matrix.clone()
            sim_matrix_clone.fill_diagonal_(float('-inf'))
            hard_negative_sim = torch.max(sim_matrix_clone, dim=1).values

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
        print(f"Epoch {epoch+1} | Train Loss (Triplet): {avg_train_loss:.6f} | Val MRR (Global): {current_mrr:.6f} | LR: {current_lr:1.1e}")

        if current_mrr > best_mrr:
            best_mrr = current_mrr
            epochs_no_improve = 0
            print(f"  -> Nuovo MRR migliore! Salvo il modello in {MODEL_PATH}")
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            print(f"  -> MRR non migliorato ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"--- Early Stopping: MRR non migliora da {PATIENCE} epoche. ---")
            break

    print(f"\nTraining completato. Il miglior modello è stato salvato in {MODEL_PATH} con Val MRR: {best_mrr:.6f}")
    return model

@torch.no_grad()
def run_submission_inference_stitcher(model, test_embds_np, DEVICE, BATCH_SIZE_TEST=1024):
    """ Funzione di inferenza submission (già in utils) """
    model.eval()
    test_embds_norm_np = normalize(test_embds_np, norm='l2', axis=1)
    test_embds_norm = torch.tensor(test_embds_norm_np, dtype=torch.float32)

    pred_embds_list = []

    for i in tqdm(range(0, len(test_embds_norm), BATCH_SIZE_TEST), desc="[Submission] Esecuzione Stitcher"):
        batch_X_norm = test_embds_norm[i:i+BATCH_SIZE_TEST].to(DEVICE)
        batch_final = model(batch_X_norm)
        pred_embds_list.append(batch_final.cpu())

    pred_embds_final = torch.cat(pred_embds_list, dim=0)
    return pred_embds_final.numpy()

@torch.no_grad()
def run_validation_inference_stitcher(model, val_loader, DEVICE):
    """ Funzione di inferenza validazione (già in utils) """
    model.eval()
    all_translated_embeddings = []
    
    pbar = tqdm(val_loader, desc="[Validazione] Esecuzione Stitcher")
    for batch_X_norm, _ in pbar: 
        batch_X_norm = batch_X_norm.to(DEVICE)
        anchor_norm = model(batch_X_norm)
        all_translated_embeddings.append(anchor_norm.cpu())
        
    return torch.cat(all_translated_embeddings, dim=0).numpy()

def calculate_mrr_validation_sampled(
    X_queries_proj, 
    groups_val, 
    Y_gallery_unique_ALL, 
    groups_gallery_unique_ALL,
    n_samples=99 
):
    """ Calcolo MRR (già in utils) """
    if X_queries_proj.shape[0] == 0:
        return 0.0

    N_queries = X_queries_proj.shape[0]
    N_gallery = Y_gallery_unique_ALL.shape[0]

    Y_gallery_unique_ALL_norm = normalize(Y_gallery_unique_ALL, axis=1)
    X_queries_proj_norm = normalize(X_queries_proj, axis=1)

    all_gallery_indices = np.arange(N_gallery)
    
    mrr_sum = 0
    pbar_val = tqdm(range(N_queries), desc="[Validazione] Calcolo MRR (1+99)", leave=False, disable=True)

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

# ----------------------------------------------------------------------------
# FUNZIONI DI SETUP E PREPARAZIONE (Spostate da 03_stitcher.py)
# ----------------------------------------------------------------------------

def setup_environment_stitcher(seed, device_str="cuda"):
    """Imposta seed e device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and device_str == "cuda":
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")
    print(f"--- Utilizzo del device: {device} ---")
    return device

def load_and_prepare_data_stitcher(train_data_path, device):
    """Carica i dati, li normalizza, li sposta sul device e crea i gruppi."""
    print("Caricamento Dati di Training...")
    train_data_dict = load_data(train_data_path)
    z_text_raw, z_img_raw, label_matrix = prepare_train_data(train_data_dict)
    print(f"Campioni totali: {len(z_text_raw)}")

    X_FINAL = F.normalize(z_text_raw.float(), p=2, dim=1).to(device)
    Y_FINAL = F.normalize(z_img_raw.float(), p=2, dim=1).to(device)
    print("Dati normalizzati e spostati su device.")

    groups = torch.argmax(label_matrix.float(), axis=1).cpu().numpy()
    return X_FINAL, Y_FINAL, groups

def create_splits_stitcher(X_FINAL, groups, temp_split_ratio, test_split_ratio_of_temp, seed):
    """Esegue il doppio GroupShuffleSplit."""
    print("Split 1: Creazione set Train (90%) e Temp (10%)...")
    gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=temp_split_ratio, random_state=seed)
    train_indices, temp_indices = next(gss_train_temp.split(X_FINAL.cpu().numpy(), y=None, groups=groups))

    print("Split 2: Divisione set Temp in Validation (5%) e Test (5%)...")
    groups_temp = groups[temp_indices]
    X_temp_dummy = np.empty((len(groups_temp), 1)) 
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=test_split_ratio_of_temp, random_state=seed)
    val_indices_rel, test_indices_rel = next(gss_val_test.split(X_temp_dummy, y=None, groups=groups_temp))

    val_indices = temp_indices[val_indices_rel]
    test_indices = temp_indices[test_indices_rel]
    groups_val = groups[val_indices] 
    groups_test = groups[test_indices]
    
    print(f"Split completato: Train ({len(train_indices)}), Val ({len(val_indices)}), Test ({len(test_indices)})")
    return train_indices, val_indices, test_indices, groups_val, groups_test

def load_global_gallery(submission_dir, gallery_file="gallery_data.npz"):
    """Carica la galleria globale creata da 01_sota.py."""
    gallery_npz_path = Path(submission_dir) / gallery_file
    if not gallery_npz_path.exists():
        print(f"ERRORE: Galleria globale '{gallery_file}' non trovata.")
        print("       Assicurati di aver eseguito '01_sota.py' prima di questo script.")
        exit()
        
    gallery_data = np.load(gallery_npz_path)
    Y_gallery_unique_ALL = gallery_data['embeddings']
    groups_gallery_unique_ALL = gallery_data['groups']
    print(f"Galleria Globale caricata per il test ({len(Y_gallery_unique_ALL)} campioni).")
    return Y_gallery_unique_ALL, groups_gallery_unique_ALL

def create_dataloaders_stitcher(
    X_FINAL, Y_FINAL, train_indices, val_indices, groups, 
    batch_size, num_workers
):
    """
    Crea i 3 DataLoaders necessari:
    1. train_loader: (X_train, Y_train) per la triplet loss.
    2. val_loader_train: (X_val, mapped_indices_locali) per il loop di validazione.
    3. val_loader_inf: (X_val, Y_val) per l'inferenza post-training.
    """
    print("Creazione Datasets e DataLoaders...")
    
    # 1. Train Loader (per Triplet Loss)
    train_dataset = TensorDataset(X_FINAL[train_indices], Y_FINAL[train_indices])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=False, drop_last=True
    )

    # 2. Validation Loader (per MRR Interno durante il training)
    groups_val = groups[val_indices]
    groups_val_unique, val_unique_indices_relative = np.unique(groups_val, return_index=True)
    val_unique_indices_absolute = val_indices[val_unique_indices_relative]
    
    groups_gallery_val_unique = groups_val_unique # numpy
    print(f"Galleria di Validazione (interna) creata con {len(groups_gallery_val_unique)} immagini uniche.")
    
    group_to_gallery_idx = {group_id: idx for idx, group_id in enumerate(groups_gallery_val_unique)}
    val_label_indices = [group_to_gallery_idx[gid] for gid in groups_val]
    val_label_indices_t = torch.tensor(val_label_indices, dtype=torch.long)

    val_dataset_train = TensorDataset(X_FINAL[val_indices], val_label_indices_t)
    val_loader_train = DataLoader(
        val_dataset_train, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    # 3. Validation Loader (per inferenza post-training)
    val_dataset_inf = TensorDataset(X_FINAL[val_indices], Y_FINAL[val_indices])
    val_loader_inf = DataLoader(
        val_dataset_inf, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    print("DataLoaders (train, val_for_train, val_for_inf) pronti.")
    return train_loader, val_loader_train, val_loader_inf

def setup_model_stitcher(input_dim, output_dim, hidden_dim, dropout_p, device):
    """Inizializza il modello Stitcher."""
    print("Inizializzazione modello...")
    model = Stitcher(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=hidden_dim, 
        dropout_p=dropout_p
    ).to(device)
    print(f"   Parametri dello Stitcher: {sum(p.numel() for p in model.parameters()):,}")
    return model

# ----------------------------------------------------------------------------
# FUNZIONI DI VERIFICA E SUBMISSION (Spostate da 03_stitcher.py)
# ----------------------------------------------------------------------------

def run_verification_stitcher(
    model, checkpoint_path, val_loader_inf, X_test_tensor,
    groups_test, groups_val, Y_gallery_unique_ALL, groups_gallery_unique_ALL,
    submission_dir, device
):
    """
    Esegue la verifica post-training:
    1. Carica il modello migliore.
    2. Salva gli embedding di validazione per l'ensemble.
    3. Calcola l'MRR sul test set interno.
    """
    print("\n--- Inizio Verifica Post-Training ---")

    if not Path(checkpoint_path).exists():
        print(f"Errore: Checkpoint non trovato in {checkpoint_path}. Impossibile continuare.")
        print("Esegui lo script con PERFORM_TRAINING = True.")
        exit()
        
    print(f"Caricamento del modello migliore da {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 1. Salva Embedding di Validazione (per tuning alpha)
    print(f"Generazione embedding di VALIDAZIONE (per tuning alpha)...")
    val_embeddings = run_validation_inference_stitcher(model, val_loader_inf, device)
    
    val_npz_path = Path(submission_dir) / "val_stitcher.npz" 
    np.savez(
        val_npz_path, 
        embeddings=val_embeddings, 
        groups=groups_val 
    )
    print(f"Embedding di validazione salvati in: {val_npz_path}")
    
    # 2. Esegui Test su Test Set Interno (5%)
    print(f"Valutazione performance su TEST SET INTERNO (5%)...")
    with torch.no_grad():
        test_internal_embeddings = model(X_test_tensor).cpu().numpy()

    test_mrr = calculate_mrr_validation_sampled(
        X_queries_proj=test_internal_embeddings,
        groups_val=groups_test, # Etichette query test
        Y_gallery_unique_ALL=Y_gallery_unique_ALL,
        groups_gallery_unique_ALL=groups_gallery_unique_ALL
    )
    print(f"--- MRR sul Test Set Interno (vs Galleria Globale): {test_mrr:.6f} ---")

def generate_submission_files_stitcher(
    model, checkpoint_path, test_data_path, 
    submission_dir, device, batch_size
):
    """Genera i file .npz e .csv finali per la submission."""
    print("\n--- Inizio Generazione Submission File ---")
    
    if not Path(checkpoint_path).exists():
        print(f"Errore: Checkpoint non trovato. (Già controllato, ma per sicurezza)")
        return

    # Assicurati che il modello sia caricato (già fatto in run_verification, 
    # ma lo rifacciamo qui se lo script viene eseguito solo per l'inferenza)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"Caricamento dati di test da {test_data_path}...")
    test_data_clean = load_data(test_data_path)

    z_text_test_raw_np = test_data_clean['captions/embeddings'] 
    sample_ids = test_data_clean['captions/ids']
    print(f"Dati di test caricati: {z_text_test_raw_np.shape}")
    
    print("Esecuzione inferenza sui dati di submission...")
    translated_embeddings = run_submission_inference_stitcher(
        model, 
        z_text_test_raw_np, 
        device,
        BATCH_SIZE_TEST=batch_size
    )
    
    # Salva .npz per l'ensemble
    submission_npz_path = Path(submission_dir) / "submission_stitcher.npz" 
    np.savez(
        submission_npz_path, 
        embeddings=translated_embeddings, 
        ids=sample_ids
    )
    print(f"Embedding di submission per ensemble salvati in: {submission_npz_path}")

    # Salva .csv per la submission
    submission_path = Path(submission_dir) / "submission_stitcher.csv"
    print(f"Calcolo similarità e salvataggio submission in {submission_path}...")
    
    generate_submission(
        sample_ids,                     
        translated_embeddings,    
        output_file=str(submission_path)  
    )

    print(f"--- Submission (Stitcher) creata con successo! ---")