import matplotlib
matplotlib.use('TkAgg')

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

K, SEED, MAX_ITER = 3, 0, 100

folder = "data"
if not os.path.exists(folder):
    folder = "data"

files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
names = [f.split("_")[1].split(".")[0] for f in files]

imgs = []
for f in files:
    img_path = os.path.join(folder, f)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        imgs.append(cv2.resize(img, (100, 100)))

X = np.array([img.flatten() for img in imgs], dtype=float)

np.random.seed(SEED)
C = X[np.random.permutation(len(X))[:K]]
for it in range(1, MAX_ITER + 1):
    D = np.array([np.sum(np.abs(X - C[k]), axis=1) for k in range(K)]).T
    idx = np.argmin(D, axis=1)
    cold = C.copy()
    C = np.array([np.mean(X[idx == k], axis=0) if np.any(idx == k) else C[k] for k in range(K)])
    if np.sum(np.abs(cold - C)) == 0:
        break

for k in range(K):
    members = [names[j] for j in range(len(X)) if idx[j] == k]
    print(f"Cluster {k + 1}: {', '.join(members)}")

cols = int(max(np.bincount(idx, minlength=K).max(), 1))
fig, axes = plt.subplots(K, cols, figsize=(cols * 1.4 + 1, K * 2), squeeze=False)

for ax in axes.flatten():
    ax.axis("off")

for k in range(K):
    axes[k, 0].text(-0.15, 0.5, f"Cluster {k + 1}", transform=axes[k, 0].transAxes, ha="right", va="center")
    cluster_indices = np.where(idx == k)[0]
    for c, j in enumerate(cluster_indices):
        axes[k, c].imshow(imgs[j], cmap="gray", vmin=0, vmax=255)
        axes[k, c].set_title(names[j], fontsize=8)

plt.tight_layout()
plt.show()