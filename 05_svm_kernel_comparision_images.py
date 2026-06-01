# %% [markdown]
# ## Rohdaten

# %%
# Daten erzeugen
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=500,
    #centers=3,
    centers=[[-5, -5], [0, 10], [6, 0]],
    cluster_std=3.5,   # mehr Überlappung
    random_state=42
)

plt.figure(figsize=(8,6))

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap="viridis",
    s=50
)

#plt.title("Testdaten mit 3 Klassen")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

# %% [markdown]
# ## Linear Kernel

# %%
#Lineare SVM
from sklearn.svm import SVC

linear_model = SVC(
    kernel="linear",
    C=1.0
)

linear_model.fit(X, y)

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

Z = linear_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
).reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25,
    cmap="viridis"
)

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap="viridis",
    s=50
)

plt.title("Lineare SVM (3 Klassen)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

# %% [markdown]
# ## Poly Kernel

# %%
# Poly Kernel
poly_model = SVC(kernel="poly", degree=3, C=10, gamma=10)

poly_model.fit(X, y)

Z = poly_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
).reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25,
    cmap="viridis"
)

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap="viridis",
    s=50
)

#plt.title("Polynomial Kernel (3 Klassen)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

# %% [markdown]
# ## RBF Kernel

# %%
#RBF Kernel
rbf_model = SVC(kernel="rbf", C=1, gamma=10)

rbf_model.fit(X, y)

Z = rbf_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
).reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25,
    cmap="viridis"
)

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap="viridis",
    s=50
)

#plt.title("RBF Kernel (3 Klassen)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

# %% [markdown]
# # Alle Kernel im Vergleich

# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# ==========================================
# 1. Daten erzeugen
# ==========================================

X, y = make_blobs(
    n_samples=500,
    #centers=3,
    centers=[[-5, -5], [0, 10], [6, 0]],
    cluster_std=3.5,   # mehr Überlappung
    random_state=42
)

# ==========================================
# Hilfsfunktion zum Plotten
# ==========================================

def plot_decision_boundary(model, X, y, title, ax):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.25,
        cmap="coolwarm"
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        s=40
    )

    ax.set_title(title)

# ==========================================
# 2. Modelle trainieren
# ==========================================

linear_model = SVC(
    kernel="linear",
    C=1.0
)

poly_model = SVC(
    kernel="poly",
    degree=3,
    C=0.10,
    gamma=0.1
    )

rbf_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

linear_model.fit(X, y)
poly_model.fit(X, y)
rbf_model.fit(X, y)

# ==========================================
# 3. Ergebnisse darstellen
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Originaldaten
axes[0, 0].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    s=40
)
axes[0, 0].set_title("1. Originaldaten")

# Lineare SVM
plot_decision_boundary(
    linear_model,
    X,
    y,
    "2. Linearer Kernel",
    axes[0, 1]
)

# Polynomial Kernel
plot_decision_boundary(
    poly_model,
    X,
    y,
    "3. Polynomial Kernel (Grad 4)",
    axes[1, 0]
)

# RBF Kernel
plot_decision_boundary(
    rbf_model,
    X,
    y,
    "4. RBF Kernel",
    axes[1, 1]
)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3D Transformation der Rohdaten

# %%
#3D transformiert zum besseren trennen
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Deine Daten
# -----------------------------
X, y = make_blobs(
    n_samples=500,
    centers=[[-5, -5], [0, 10], [6, 0]],
    cluster_std=3.5,
    random_state=42
)

# -----------------------------
# Feature-Transformation (Intuition!)
# -----------------------------
x1 = X[:, 0]
x2 = X[:, 1]

# künstliche 3. Dimension
z = x1**2 + x2**2   # radialer Abstand (sehr anschaulich!)

X_3d = np.column_stack([x1, x2, z])

# -----------------------------
# 3D Plot
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X_3d[:,0],
    X_3d[:,1],
    X_3d[:,2],
    c=y,
    cmap="viridis",
    s=30
)

ax.set_title("2D → 3D Feature Expansion (Kernel-Intuition)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3 (nichtlinear)")

plt.show()
