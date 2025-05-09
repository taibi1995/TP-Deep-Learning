import torch
import torch.nn as nn
import torch.nn.functional as F


def my_conv2d(X, K):
    h, w = X.shape
    k, _ = K.shape  # Taille du filtre (carré, donc k x k)
    
    h_out = h - k + 1  # Hauteur de la sortie
    w_out = w - k + 1  # Largeur de la sortie
    
    Y = torch.zeros((h_out, w_out))

    for x in range(h_out):  
        for y in range(w_out):
            # Produit élément par élément entre le patch de X et le filtre K
            patch = X[x:x+k, y:y+k]  # Extraire la fenêtre de taille k x k
            Y[x, y] = torch.sum(patch * K)  # Appliquer la convolution et stocker le résultat

    return Y
   
X = torch.randn(100,100)
K = torch.randn(3,3)

my_Y = my_conv2d(X, K)
print(f"Résultat my_conv2d : {my_Y.shape}")

X_reshape = X.reshape(1, 1, 100, 100)
K_reshape = K.reshape(1, 1, 3, 3)

Y = F.conv2d(X_reshape, K_reshape)

Y_prime = torch.linalg.norm(my_Y)
Y_2d = torch.linalg.norm(Y)
print(f"Resultat F.conv2d : {Y.shape}")

is_close = torch.allclose(Y_prime, Y_2d)
print(f"Are they close? {is_close}")

Kh = torch.tensor([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
], dtype=torch.float32)

Kv = torch.transpose(Kh, dim0=0, dim1= 1)

Kd = torch.tensor([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2]
], dtype=torch.float32)

Kd2 = torch.rot90(Kd)

Kb = torch.tensor([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=torch.float32)

Ki = torch.tensor([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=torch.float32)

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()

mnist = torchvision.datasets.MNIST(root="./datasets", train=True, transform=transform, download=True)
fashion_mnist = torchvision.datasets.FashionMNIST(root="./datasets", train=True, transform=transform, download=True)

def show_images(dataset, title, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(num_images):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap="gray")  # Supprimer la dimension du canal (1,28,28) -> (28,28)
        axes[i].axis("off")
    plt.suptitle(title)
    plt.show()

# show_images(mnist, "Exemples du dataset MNIST")
# show_images(fashion_mnist, "Exemples du dataset FashionMNIST")

def apply_filter(X, K):
    X_reshape = X.reshape(1, 1, X.shape[1], X.shape[2])
    K_reshape = K.reshape(1, 1, 3, 3)

    Y = F.conv2d(X_reshape, K_reshape, padding=1)

    return Y.squeeze()

mnist_image, _ = mnist[0]  # Première image du dataset MNIST
fashion_image, _ = fashion_mnist[0]  # Première image du dataset FashionMNIST

mnist_filtered = apply_filter(mnist_image, Ki)
fashion_filtered = apply_filter(fashion_image, Ki)

# fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# # MNIST
# axes[0, 0].imshow(mnist_image.squeeze(), cmap="gray")
# axes[0, 0].set_title("Image MNIST originale")
# axes[0, 0].axis("off")

# axes[0, 1].imshow(mnist_filtered, cmap="gray")
# axes[0, 1].set_title("MNIST après filtre")
# axes[0, 1].axis("off")

# # FashionMNIST
# axes[1, 0].imshow(fashion_image.squeeze(), cmap="gray")
# axes[1, 0].set_title("Image FashionMNIST originale")
# axes[1, 0].axis("off")

# axes[1, 1].imshow(fashion_filtered, cmap="gray")
# axes[1, 1].set_title("FashionMNIST après filtre")
# axes[1, 1].axis("off")

# plt.show()

def my_conv2d_batch(X_batches, K_filters):
    b, h, w = X_batches.shape
    f, k, _ = K_filters.shape

    h_prime = h - k + 1
    w_prime = w - k + 1

    Y = torch.zeros(b, f, h_prime, w_prime, dtype= torch.float32)

    for i in range(b):
        for j in range(f):
            Y[i, j] = my_conv2d(X_batches[i], K_filters[j])

    return Y

import time

# Définition des dimensions pour le test
# batch_size = 32
# image_size = 100  # Images de 100x100
# num_filters = 8
# kernel_size = 3

# # Générer un batch de 32 images aléatoires (100x100)
# X_batch = torch.randn(batch_size, image_size, image_size, dtype=torch.float32)

# # Générer 8 filtres aléatoires (3x3)
# K_filters = torch.randn(num_filters, kernel_size, kernel_size, dtype=torch.float32)

# # ➤ 1️⃣ Mesurer le temps d'exécution de `my_conv2d_batch`
# start_time = time.time()
# Y_my = my_conv2d_batch(X_batch, K_filters)
# end_time = time.time()
# time_my_conv2d = end_time - start_time
# print(f"Temps d'exécution de `my_conv2d_batch` : {time_my_conv2d:.4f} s")

# # ➤ 2️⃣ Adapter les entrées pour `F.conv2d`
# X_reshape = X_batch.reshape(batch_size, 1, image_size, image_size)  # (batch, channels, h, w)
# K_reshape = K_filters.reshape(num_filters, 1, kernel_size, kernel_size)  # (out_channels, in_channels, k, k)

# # ➤ 3️⃣ Mesurer le temps d'exécution de `F.conv2d`
# start_time = time.time()
# Y_torch = F.conv2d(X_reshape, K_reshape)  # Pas de padding, stride=1
# end_time = time.time()
# time_torch_conv2d = end_time - start_time
# print(f"Temps d'exécution de `F.conv2d` : {time_torch_conv2d:.4f} s")

# # ➤ 4️⃣ Comparaison des résultats
# is_close = torch.allclose(Y_my, Y_torch.squeeze(), atol=1e-5)
# print(f"Les résultats sont-ils proches ? {is_close}")

def my_conv2d_multi_channel(X_batch, K_filters):
    b, c_i, h, w = X_batch.shape
    f, _, k, _ = K_filters.shape
    
    h_out = h - k + 1
    w_out = w - k + 1
    
    Y = torch.zeros((b, f, h_out, w_out), dtype=torch.float32)

    for i in range(b):
        for j in range(f):
            Y[i, j] = sum(my_conv2d(X_batch[i, t], K_filters[j, t]) for t in range(c_i))  # Somme sur les canaux
    
    return Y

import torch.nn as nn
import time

# Création d'un batch de 8 images RGB (64x64)
batch_size = 8
image_size = 64
num_channels = 3
num_filters = 4
kernel_size = 3

X_batch = torch.randn(batch_size, num_channels, image_size, image_size, dtype=torch.float32)

# Générer des filtres 3x3 pour 3 canaux
K_filters = torch.randn(num_filters, num_channels, kernel_size, kernel_size, dtype=torch.float32)

# ➤ 1️⃣ Temps d'exécution de `my_conv2d_multi_channel`
start_time = time.time()
Y_my = my_conv2d_multi_channel(X_batch, K_filters)
end_time = time.time()
time_my_conv = end_time - start_time
print(f"Temps d'exécution de `my_conv2d_multi_channel` : {time_my_conv:.4f} s")

# ➤ 2️⃣ Comparaison avec `nn.Conv2d`
conv_layer = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=kernel_size, bias=False)
conv_layer.weight.data = K_filters  # Utiliser les mêmes poids

start_time = time.time()
Y_torch = conv_layer(X_batch)
end_time = time.time()
time_torch_conv = end_time - start_time
print(f"Temps d'exécution de `nn.Conv2d` : {time_torch_conv:.4f} s")

# ➤ 3️⃣ Comparaison des résultats
is_close = torch.allclose(Y_my, Y_torch, atol=1e-5)
print(f"Les résultats sont-ils proches ? {is_close}")


import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Charger une image couleur
image = Image.open("chat.jpg")  # Remplace par ton image
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])  # Redimensionner à 64x64
X_rgb = transform(image).unsqueeze(0)  # Ajouter la dimension batch (1, 3, 64, 64)

# Définition des filtres de Sobel multi-canaux (RGB)
Kh = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
Kv = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)

K_rgb = torch.stack([Kh, Kv, Kh])  # Appliquer le même filtre sur les 3 canaux
K_filters_rgb = K_rgb.unsqueeze(0).expand(2, -1, -1, -1)  # Créer 2 filtres appliqués sur 3 canaux

# Appliquer la convolution multi-canaux
Y_rgb = my_conv2d_multi_channel(X_rgb, K_filters_rgb)

# Afficher les résultats
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(X_rgb.squeeze().permute(1, 2, 0))  # Image originale
axes[0].set_title("Image RGB Originale")
axes[0].axis("off")

axes[1].imshow(Y_rgb[0, 0], cmap="gray")  # Sobel horizontal
axes[1].set_title("Sobel Horizontal")
axes[1].axis("off")

axes[2].imshow(Y_rgb[0, 1], cmap="gray")  # Sobel vertical
axes[2].set_title("Sobel Vertical")
axes[2].axis("off")

plt.show()
