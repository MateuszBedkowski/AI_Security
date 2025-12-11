import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate synthetic network traffic data
# Features represent packet attributes: e.g., source IP, dest IP, ports, payload length, etc.
# Labels: 0 for normal traffic, 1 for attack traffic
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the intrusion detection model (linear classifier for simplicity)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Select malicious samples (attack traffic)
attack_samples = X_test[y_test == 1][:5]

# Implement simple Fast Gradient Sign Method (FGSM) for evasion
# For linear model: adversarial example x' = x - eps * sign(w)
w = model.coef_[0]
eps = 1.0  # perturbation strength

print("Scenario #2: Manipulation of network traffic to evade intrusion detection systems")
print("Demonstrating adversarial evasion on a linear IDS model")
print()

for i, x in enumerate(attack_samples):
    original_pred = model.predict([x])[0]
    # Generate adversarial by perturbing
    perturbation = eps * np.sign(w)
    x_adv = x - perturbation
    adversarial_pred = model.predict([x_adv])[0]
    print(f"Sample {i+1}: Original prediction: {original_pred} (1=attack), Adversarial prediction: {adversarial_pred} (0=evaded)")

print()
print("Evasion successful if adversarial prediction is 0.")

# print("Perturbations correspond to altering packet features to fool the linear classifier.")

