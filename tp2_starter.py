import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import gzip
import os

DATA_DIR = "./datasets/fashion-mnist"
TRAIN_IMAGES_FILE = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")
TEST_IMAGES_FILE = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS_FILE = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")

TEXT_LABELS = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

def load_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32, count=4)
        num_images, rows, cols = num_images.byteswap(), rows.byteswap(), cols.byteswap()
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols) / 255.0

def load_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.uint32, count=2)
        num_labels = num_labels.byteswap()
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

train_imgs = load_images(TRAIN_IMAGES_FILE)
train_labels = load_labels(TRAIN_LABELS_FILE)

test_imgs = load_images(TEST_IMAGES_FILE)
test_labels = load_labels(TEST_LABELS_FILE)

# print("Shape des images d'entraînement :", train_imgs.shape)
# print("Shape des labels d'entraînement :", train_labels.shape)
# print("Shape des images de test :", test_imgs.shape)
# print("Shape des labels de test :", test_labels.shape)

def show_image(index):
    label_text = TEXT_LABELS[train_labels[index]]
    fig = px.imshow(train_imgs[index], color_continuous_scale="gray", title=f"Image {index} - Label: {label_text}")
    fig.update_coloraxes(showscale=False)
    fig.show()

# show_image(50050)

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

train_imgs_flat = flatten_images(train_imgs)
test_imgs_flat = flatten_images(test_imgs)

print("Shape des images aplaties d'entraînement :", train_imgs_flat.shape)
print("Shape des images aplaties de test :", test_imgs_flat.shape)

def softmax(o):
    o_stable = o - np.max(o)
    exp_o = np.exp(o_stable)
    return exp_o / np.sum(exp_o)

def cross_entropy_loss(p, c):
    return -np.log(p[c])

def grad(x, p, c):
    p_prime = p.copy()
    p_prime[c] -= 1

    dL_dW = np.outer(p_prime, x)
    dL_db = p_prime

    return dL_dW, dL_db

def init_weights_biases(input_size, output_size):
    W = np.random.normal(0, 0.01, (input_size, output_size))
    b = np.zeros(output_size)

    return W, b

def train_on_batch(X, Y, W, b, learning_rate=0.1, iterations=500):

    n_samples = X.shape[0]
    loss_history = []
    
    for it in range(iterations):
        # 1. Calcul des logits et des probabilités
        logits = np.dot(X, W) + b  # Logits : x · W + b
        probabilities = np.apply_along_axis(softmax, 1, logits)  # Softmax sur chaque exemple
        
        # 2. Calcul de la perte moyenne d'entropie croisée
        losses = [-np.log(probabilities[i, Y[i]]) for i in range(n_samples)]  # Perte pour chaque exemple
        mean_loss = np.mean(losses)  # Perte moyenne sur le batch
        loss_history.append(mean_loss)
        
        # 3. Calcul des gradients
        dW = np.zeros_like(W)  # Gradient de W
        db = np.zeros_like(b)  # Gradient de b
        
        for i in range(n_samples):
            x = X[i]  # Exemple unique (784,)
            p = probabilities[i]  # Probabilités pour cet exemple
            c = Y[i]  # Classe correcte
            
            # Calcul des gradients pour cet exemple
            p_prime = p.copy()
            p_prime[c] -= 1  # Modification de p pour la classe correcte
            dW += np.outer(x, p_prime)  # Produit extérieur pour W
            db += p_prime  # Gradient pour b
            
        dW /= n_samples  # Moyenne des gradients pour W
        db /= n_samples  # Moyenne des gradients pour b
        
        # 4. Mise à jour des poids et biais
        W -= learning_rate * dW
        b -= learning_rate * db
        
        # Affichage de la perte toutes les 50 itérations
        if (it + 1) % 50 == 0:
            print(f"Iteration {it + 1}/{iterations}, Perte moyenne : {mean_loss:.5f}")
    
    return W, b, loss_history

# Préparation des données
batch_size = 64
X_batch = train_imgs_flat[:batch_size]  # Les 64 premières images aplaties
Y_batch = train_labels[:batch_size]  # Les 64 premiers labels

# Initialisation des poids et biais
W, b = init_weights_biases(input_size=784, output_size=10)

# Entraînement sur le batch unique
learning_rate = 0.1
iterations = 500
W, b, loss_history = train_on_batch(X_batch, Y_batch, W, b, learning_rate, iterations)

# Affichage de la perte finale
# print(f"Perte finale après {iterations} itérations : {loss_history[-1]:.5f}")

# # Visualisation de l'évolution de la perte

# plt.plot(range(iterations), loss_history)
# plt.title("Évolution de la perte moyenne d'entropie croisée")
# plt.xlabel("Itérations")
# plt.ylabel("Perte")
# plt.grid(True)
# plt.show()

def compute_accuracy(X, Y, W, b):

    # 1. Calcul des logits
    logits = np.dot(X, W) + b
    
    # 2. Calcul des probabilités (softmax)
    probabilities = np.apply_along_axis(softmax, 1, logits)
    
    # 3. Prédictions : choisir la classe avec la probabilité maximale
    predictions = np.argmax(probabilities, axis=1)
    
    # 4. Comparaison avec les labels réels
    correct_predictions = np.sum(predictions == Y)
    
    # 5. Calcul de la précision
    accuracy = (correct_predictions / len(Y)) * 100
    return accuracy

# Exemple d'évaluation
# (a) Précision sur le batch d'entraînement
train_accuracy = compute_accuracy(X_batch, Y_batch, W, b)
print(f"Précision sur le batch d'entraînement : {train_accuracy:.2f}%")

# (b) Précision sur un batch du jeu de validation
X_val_batch = train_imgs_flat[batch_size:2*batch_size]  # Un autre batch de validation
Y_val_batch = train_labels[batch_size:2*batch_size]

val_accuracy = compute_accuracy(X_val_batch, Y_val_batch, W, b)
print(f"Précision sur le batch de validation : {val_accuracy:.2f}%")


import time

def train_with_minibatches(X_train, Y_train, X_val, Y_val, W, b, batch_size=64, epochs=10, learning_rate=0.1):
  
    n_train = X_train.shape[0]
    n_batches = n_train // batch_size
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        # Shuffle des données d'entraînement au début de chaque epoch
        indices = np.arange(n_train)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        
        epoch_start_time = time.time()
        epoch_loss = 0
        
        for batch_idx in range(n_batches):
            # Sélection du batch
            start = batch_idx * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            Y_batch = Y_train[start:end]
            
            # 1. Calcul des logits et des probabilités
            logits = np.dot(X_batch, W) + b
            probabilities = np.apply_along_axis(softmax, 1, logits)
            
            # 2. Calcul de la perte moyenne d'entropie croisée
            losses = [-np.log(probabilities[i, Y_batch[i]]) for i in range(batch_size)]
            batch_loss = np.mean(losses)
            epoch_loss += batch_loss
            
            # 3. Calcul des gradients
            p_prime = probabilities.copy()
            for i in range(batch_size):
                p_prime[i, Y_batch[i]] -= 1
            dW = np.dot(X_batch.T, p_prime) / batch_size
            db = np.sum(p_prime, axis=0) / batch_size
            
            # 4. Mise à jour des paramètres
            W -= learning_rate * dW
            b -= learning_rate * db
        
        # Moyenne de la perte pour l'epoch
        epoch_loss /= n_batches
        train_loss_history.append(epoch_loss)
        
        # Calcul de la perte sur le dataset de validation
        val_logits = np.dot(X_val, W) + b
        val_probabilities = np.apply_along_axis(softmax, 1, val_logits)
        val_losses = [-np.log(val_probabilities[i, Y_val[i]]) for i in range(len(Y_val))]
        val_loss = np.mean(val_losses)
        val_loss_history.append(val_loss)
        
        # Temps d'exécution pour l'epoch
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs}: Perte d'entraînement = {epoch_loss:.4f}, Perte de validation = {val_loss:.4f}, Temps = {epoch_time:.2f}s")
    
    return W, b, train_loss_history, val_loss_history

# Préparation des données
X_train = train_imgs_flat[:50000]  # 50 000 exemples d'entraînement
Y_train = train_labels[:50000]
X_val = train_imgs_flat[50000:]  # 10 000 exemples de validation
Y_val = train_labels[50000:]

# Initialisation des poids et biais
W, b = init_weights_biases(input_size=784, output_size=10)

# Entraînement avec descente de gradient par mini-batchs
batch_size = 64
epochs = 10
learning_rate = 0.1
W, b, train_loss_history, val_loss_history = train_with_minibatches(X_train, Y_train, X_val, Y_val, W, b, batch_size, epochs, learning_rate)


plt.plot(range(epochs), train_loss_history, label="Perte d'entraînement")
plt.plot(range(epochs), val_loss_history, label="Perte de validation")
plt.title("Évolution des pertes au fil des epochs")
plt.xlabel("Epochs")
plt.ylabel("Perte")
plt.legend()
plt.grid(True)
plt.show()

def evaluate_model(X_test, Y_test, W, b):
    # 1. Calcul des logits pour le dataset de test
    logits = np.dot(X_test, W) + b
    
    # 2. Calcul des probabilités (softmax)
    logits -= np.max(logits, axis=1, keepdims=True)  # Stabilisation pour éviter les débordements
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 3. Prédictions : choisir la classe avec la probabilité maximale
    predictions = np.argmax(probabilities, axis=1)
    
    # 4. Calcul de la précision
    correct_predictions = np.sum(predictions == Y_test)
    accuracy = (correct_predictions / len(Y_test)) * 100
    
    return accuracy

# Chargement du dataset de test
X_test = test_imgs_flat  # 10 000 exemples de test aplatis
Y_test = test_labels  # Labels des exemples de test

# Évaluation de la précision sur le dataset de test
test_accuracy = evaluate_model(X_test, Y_test, W, b)
print(f"Précision sur le dataset de test : {test_accuracy:.2f}%")
