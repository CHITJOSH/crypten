from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from phe import paillier
import numpy as np

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset equally among parties
X_party1, X_party2, y_party1, y_party2 = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to avoid convergence warning

# Train the model on Party 1's data
model.fit(X_party1, y_party1)

# Encrypt the model parameters
public_key, private_key = paillier.generate_paillier_keypair()

# Flatten the model coefficients into a 1D array
coef_flat = model.coef_.flatten()

# Encrypt each coefficient separately
encrypted_weights = [public_key.encrypt(coef) for coef in coef_flat]

# Flatten the intercept into a scalar
intercept_scalar = model.intercept_[0]

# Encrypt the intercept
encrypted_intercept = public_key.encrypt(intercept_scalar)

# Securely communicate the encrypted model parameters

# Party 2 performs local computation
# Encrypt the model parameters
encrypted_weights_party2 = [public_key.encrypt(coef) for coef in model.coef_.flatten()]
encrypted_intercept_party2 = public_key.encrypt(model.intercept_[0])

# Aggregate the updated model parameters using homomorphic addition
aggregated_weights = [enc1 + enc2 for enc1, enc2 in zip(encrypted_weights, encrypted_weights_party2)]
aggregated_intercept = encrypted_intercept + encrypted_intercept_party2

# Decrypt the aggregated model parameters to obtain the final model
final_weights = np.array([private_key.decrypt(enc) for enc in aggregated_weights])
final_intercept = private_key.decrypt(aggregated_intercept)

# Update the model with the aggregated parameters
model.coef_ = final_weights.reshape(model.coef_.shape)
model.intercept_ = np.array([final_intercept])

# Evaluate the final model
accuracy = model.score(X, y)
print("Final Model Accuracy:", accuracy)
