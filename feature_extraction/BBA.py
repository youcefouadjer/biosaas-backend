import numpy as np
from sklearn.metrics import confusion_matrix
import ELM


class BBAFeatureSelector:
    """Bat Algorithm Feature Selector - compatible avec le .pkl sauvegardé"""
    def __init__(self, n_bats=20, n_iter=20, fmin=0, fmax=2):
        self.n_bats = n_bats
        self.n_iter = n_iter
        self.fmin = fmin
        self.fmax = fmax
        self.best_mask = None

    def compute_eer(self, y_true, y_pred):
        
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        FAR = np.mean(FP / (FP + TN + 1e-8))
        FRR = np.mean(FN / (FN + TP + 1e-8))
        return (FAR + FRR) / 2

    def fitness(self, X, y, mask):
        if mask.sum() == 0:
            return 1.0
        X_sel = X[:, mask == 1]
        indices = np.random.permutation(len(X_sel))
        X_sel = X_sel[indices]
        y_shuffled = y[indices]
        split = int(0.8 * len(X_sel))
        Xtr, Xte = X_sel[:split], X_sel[split:]
        ytr, yte = y_shuffled[:split], y_shuffled[split:]
        elm = ELM(input_dim=Xtr.shape[1], hidden_dim=5000)
        elm.fit(Xtr, ytr)
        pred = elm.predict(Xte)
        return self.compute_eer(yte, pred)

    def select(self, X, y):
        n_features = X.shape[1]
        Xb = np.random.randint(0, 2, (self.n_bats, n_features))
        V = np.zeros_like(Xb, dtype=float)
        fitness = np.array([self.fitness(X, y, Xb[i]) for i in range(self.n_bats)])
        best = Xb[np.argmin(fitness)].copy()
        for t in range(self.n_iter):
            for i in range(self.n_bats):
                beta = np.random.rand()
                f = self.fmin + (self.fmax - self.fmin) * beta
                V[i] = V[i] + (Xb[i] - best) * f
                S = 1 / (1 + np.exp(-V[i]))
                Xb[i] = (np.random.rand(n_features) < S).astype(int)
                fit = self.fitness(X, y, Xb[i])
                if fit < fitness[i]:
                    fitness[i] = fit
            best = Xb[np.argmin(fitness)].copy()
        self.best_mask = best
        return best