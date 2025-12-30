import torch


def kmeans(X, k, max_iters=10000):
    # X shape: (n_samples, n_features)
    X = X.reshape(-1, X.size(-1))
    # 1. Choose random initial centroids
    idx = torch.randperm(X.size(0))[:k]
    centroids = X[idx]

    for _ in range(max_iters):
        # 2. Compute distances (PyTorch built-in)
        distances = torch.cdist(X, centroids)

        # 3. Assign each sample to closest centroid
        labels = torch.argmin(distances, dim=1)

        # 4. Recompute centroids
        new_centroids = torch.stack([
            X[labels == c].mean(dim=0) if (labels == c).sum() > 0 else centroids[c]
            for c in range(k)
        ])

        # 5. Stop if no significant change
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    return centroids, labels