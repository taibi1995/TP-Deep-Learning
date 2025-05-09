import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
    X = data[:, 0]
    y = data[:, 1]
    return X, y

def visualize_data(x, y):
    fig = px.scatter(x=x, y=y, labels={'x': 'x', 'y': 'y'})
    fig.show()

x, y = load_data('tp1-data.csv')
#visualize_data(x,y)

def predict(xs, phis):
    return phis[0] + phis[1] * xs

def loss(y_pred, y_true):
    squared_errors = (y_pred - y_true) ** 2
    return np.sum(squared_errors)

def plot_data_and_prediction(x, y, phi0, phi1):
    y_pred = predict(x, [phi0, phi1])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="blue", label="Données réelles (x, y)")
    plt.plot(x, y_pred, color="red", label=f"Prédiction (y = {phi0} + {phi1} * x)")
    plt.title("Nuage de points et droite prédite")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    loss_value = loss(y_pred, y)

    print(f"Somme des pertes: {loss_value:.2f}")

phi0 = 800
phi1 = 20
#plot_data_and_prediction(x, y, phi0, phi1)

def loss_gradient(xs, ys, phis):
    y_pred = predict(xs, phis)

    errors = y_pred - ys

    dL_dphi0 = 2 * np.sum(errors)
    dL_dphi1 = 2 * np.sum(errors * xs)

    return dL_dphi0, dL_dphi1

def step(xs, ys, phis, alpha=1e-6):
    dL_dphi0, dL_dphi1 = loss_gradient(xs, ys, phis)

    phis[0] = phis[0] - alpha * dL_dphi0
    phis[1] = phis[1] - alpha * dL_dphi1

    return phis

def train(xs, ys, phis, alpha=1e-6, num_steps=10):
    losses = []

    for step_num in range (num_steps):
        y_pred = predict(xs, phis)
        current_loss = loss(y_pred, ys)
        losses.append(current_loss)

        # plt.figure(figsize=(8, 6))
        # plt.scatter(xs, ys, color="blue", label="Données réelles (x, y)")
        # plt.plot(xs, y_pred, color="red", label=f"Modèle après étape {step_num+1}")
        # plt.title(f"Étape {step_num+1} - Perte : {current_loss:.2f}")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        phis = step(xs, ys, phis, alpha)

    return losses, phis

phis = np.array([phi0, phi1])

losses, final_phis = train(x, y, phis)
steps = 10
# Afficher la fonction de perte au cours des étapes
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, steps+1), losses, marker="o")
# plt.title("Évolution de la perte au cours des étapes")
# plt.xlabel("Étape")
# plt.ylabel("Perte")
# plt.grid(True)
# plt.show()

# # Résultats finaux
# print("Paramètres finaux après l'entraînement :")
# print(f"φ0 : {final_phis[0]}")
# print(f"φ1 : {final_phis[1]}")


def analytical_solution(xs, ys):
    X = np.vstack([np.ones_like(xs), xs]).T
    
    phis, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
    
    return phis

optimal_phis = analytical_solution(x, y)

y_pred_optimal = predict(x, optimal_phis)
optimal_loss = loss(y_pred_optimal, y)

# print("Paramètres optimaux (solution analytique) :")
# print(f"φ0 : {optimal_phis[0]}")
# print(f"φ1 : {optimal_phis[1]}")
# print(f"Perte (solution analytique) : {optimal_loss:.2f}")

# print("\nComparaison avec les résultats par apprentissage :")
# print(f"φ0 (apprentissage) : {final_phis[0]}")
# print(f"φ1 (apprentissage) : {final_phis[1]}")
# print(f"Perte (apprentissage) : {loss(predict(x, final_phis), y):.2f}")

def check_gradient(xs, ys, phis, epsilon=1e-7):
    dphi0_analytic, dphi1_analytic = loss_gradient(xs, ys, phis)
    
    phis_plus = phis.copy()
    phis_minus = phis.copy()
    phis_plus[0] += epsilon
    phis_minus[0] -= epsilon
    loss_plus = loss(predict(xs, phis_plus), ys)
    loss_minus = loss(predict(xs, phis_minus), ys)
    dphi0_approx = (loss_plus - loss_minus) / (2 * epsilon)
    
    phis_plus = phis.copy()
    phis_minus = phis.copy()
    phis_plus[1] += epsilon
    phis_minus[1] -= epsilon
    loss_plus = loss(predict(xs, phis_plus), ys)
    loss_minus = loss(predict(xs, phis_minus), ys)
    dphi1_approx = (loss_plus - loss_minus) / (2 * epsilon)
    
    return {
        "dphi0_analytic": dphi0_analytic,
        "dphi0_approx": dphi0_approx,
        "dphi1_analytic": dphi1_analytic,
        "dphi1_approx": dphi1_approx
    }

phis = np.array([600, 10])
gradient_check = check_gradient(x, y, phis)

print("Comparaison des gradients :")
print(f"dL/dφ0 (analytique) : {gradient_check['dphi0_analytic']}")
print(f"dL/dφ0 (approximé)  : {gradient_check['dphi0_approx']}")
print(f"dL/dφ1 (analytique) : {gradient_check['dphi1_analytic']}")
print(f"dL/dφ1 (approximé)  : {gradient_check['dphi1_approx']}")
