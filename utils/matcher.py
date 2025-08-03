import numpy as np

def match_gait(query_vector, db_path):
    database = np.load(db_path, allow_pickle=True).item()
    results = []
    for name, db_vector in database.items():
        dist = np.linalg.norm(query_vector - db_vector)
        sim = 100 / (1 + dist)  # similarity %
        results.append({
            "name": name,
            "distance": round(dist, 3),
            "probability": round(sim, 3)
        })
    results.sort(key=lambda x: -x['probability'])
    return results

