import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

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

    np.random.seed(42) 
    
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
        # Questa chiamata ora è stabile grazie al seed impostato prima del loop
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