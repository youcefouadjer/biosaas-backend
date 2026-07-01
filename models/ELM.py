import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ELM:
    def __init__(self, input_dim, hidden_dim=300):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = self.b = self.beta = self.encoder = None
    def _sigmoid(self, X): return 1/(1+np.exp(-X))
    def fit(self, X, y):
        encoder = OneHotEncoder(sparse_output=False)
        T = encoder.fit_transform(y.reshape(-1,1))
        self.encoder = encoder
        self.W = np.random.randn(self.input_dim, self.hidden_dim)
        self.b = np.random.randn(self.hidden_dim)
        H = self._sigmoid(X @ self.W + self.b)
        self.beta = np.linalg.pinv(H) @ T
    def predict(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        return np.argmax(H @ self.beta, axis=1)
    
    def predict_proba(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        output = H @ self.beta
        exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict_probas(self, x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

    

    