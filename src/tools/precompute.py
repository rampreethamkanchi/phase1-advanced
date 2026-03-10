import os
import numpy as np
import json

DATASET_DIR = "/raid/manoranjan/rampreetham/CholecT45"
MAPS_PATH = os.path.join(DATASET_DIR, "dict/maps.txt")

# CholecT45 standard split (first 35 videos for training)
def get_train_vids():
    vids = []
    # Check for existing triplet files to verify availability
    for i in range(1, 100): # Check up to 100 just in case
        vname = f"VID{i:02d}"
        if os.path.exists(os.path.join(DATASET_DIR, "triplet", f"{vname}.txt")):
            vids.append(vname)
    # Per our implementation in dataset.py, we take first 35 sorted
    return sorted(vids)[:35]

def precompute_all():
    train_vids = get_train_vids()
    print(f"Precomputing stats for {len(train_vids)} training videos: {train_vids[:5]}...")
    
    # 1. Load Triple Mapping
    # format: IVT_ID, I_ID, V_ID, T_ID, IV_ID, IT_ID
    if not os.path.exists(MAPS_PATH):
        print(f"Error: Maps file not found at {MAPS_PATH}")
        return
        
    maps = np.loadtxt(MAPS_PATH, delimiter=",", dtype=int)
    # Filter out -1 or comments (row 0 is header usually, handled by np.loadtxt skip or dtype)
    # Actually row 0 in my view_file was # IVT, I, V, T, IV, IT, so we skip first 1
    maps = np.loadtxt(MAPS_PATH, delimiter=",", dtype=int, skiprows=1)
    maps = maps[maps[:, 0] >= 0]
    
    id_to_ivt = {int(row[0]): (int(row[1]), int(row[2]), int(row[3])) for row in maps}
    
    # 2. Count frequencies
    num_I, num_V, num_T = 6, 10, 15
    num_triplets = 100
    
    triplet_counts = np.zeros(num_triplets)
    knowledge_counts = np.zeros((num_I, num_T, num_V))
    
    for vid in train_vids:
        trip_path = os.path.join(DATASET_DIR, "triplet", f"{vid}.txt")
        data = np.loadtxt(trip_path, delimiter=",", dtype=int)
        # Skip frame ID (col 0)
        trips = data[:, 1:] 
        
        # Binary summing across frames
        triplet_counts += trips.sum(axis=0)
        
        # For each frame, update knowledge counts
        for frame_idx in range(len(trips)):
            active_triplets = np.where(trips[frame_idx] == 1)[0]
            for tid in active_triplets:
                if tid in id_to_ivt:
                    i, v, t = id_to_ivt[tid]
                    knowledge_counts[i, t, v] += 1

    # 3. Identify Tail Classes (Bottom 15% frequency in Training Set)
    # Using 1e-6 to avoid zero sum issues
    sorted_trips = np.argsort(triplet_counts)
    num_tail = int(0.15 * num_triplets)
    tail_classes = sorted_trips[:num_tail].tolist()
    print(f"Identified {len(tail_classes)} tail classes: {tail_classes}")
    
    # 4. Calculate P(V | I, T)
    # P(v | i, t) = count(i,t,v) / count(i,t)
    prob_matrix = np.zeros_like(knowledge_counts)
    for i in range(num_I):
        for t in range(num_T):
            total = knowledge_counts[i, t, :].sum()
            if total > 0:
                prob_matrix[i, t, :] = knowledge_counts[i, t, :] / total
            else:
                # If this pair (I, T) never appears in training,
                # use valid triplets from the global map as uniform prior
                valid_vs = [vv for tid, (ii, vv, tt) in id_to_ivt.items() if ii == i and tt == t]
                if valid_vs:
                    for v in valid_vs:
                        prob_matrix[i, t, v] = 1.0 / len(valid_vs)
                else:
                    # Genuinely invalid clinical combo
                    pass

    # 5. Save results
    results = {
        'tail_classes': tail_classes,
        'triplet_counts': triplet_counts.tolist(),
        'conditional_prior': prob_matrix.tolist(),
        'id_to_ivt': id_to_ivt
    }
    
    os.makedirs("data/cache", exist_ok=True)
    stats_path = "data/cache/stats.json"
    with open(stats_path, "w") as f:
        json.dump(results, f)
    print(f"Successfully saved training stats and priors to {stats_path}")

if __name__ == "__main__":
    precompute_all()
