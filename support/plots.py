import numpy as np
import matplotlib.pyplot as plt

def show_k(labels,shape,title = "K-means segmentation"):

    label_map = labels.reshape(shape)


    def colorize_labels(label_map):
        K = int(label_map.max().item() + 1)
        colors = np.random.rand(K, 3)  # צבע רנדומלי לכל אובייקט
        rgb = colors[label_map.cpu().numpy()]
        return rgb


    rgb_map = colorize_labels(label_map)

    plt.imshow(rgb_map)
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_cluster_means(X, labels):
    clusters = np.unique(labels)
    means = []

    for c in clusters:
        means.append(X[labels == c].mean(axis=0))  # ממוצע לכל ממד

    means = np.array(means)

    plt.figure(figsize=(10, 6))

    for i, c in enumerate(clusters):
        plt.plot(means[i], marker='o', label=f"Cluster {c}")

    plt.title("Cluster Mean Profile")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(True)
    plt.show()