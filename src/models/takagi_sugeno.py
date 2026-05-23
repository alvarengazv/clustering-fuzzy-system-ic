import numpy as np
from models.fuzzy_cmeans import FuzzyCMeans
import config

# Takagi-Sugeno classifier
class TakagiSugenoClassifier:

    # Constructor
    def __init__(self, n_rules: int = 4, m: float = 2.0,
                 max_iter_fcm: int = 300, random_state: int = None):
        self.n_rules = n_rules
        self.m = m
        self.max_iter_fcm = max_iter_fcm
        self.random_state = random_state

        self.fcm_ = None
        self.consequent_params_ = None  
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_ = 0

    # Fit method
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TakagiSugenoClassifier':
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Fuzzy C-Means (FCM)
        if config.PRINT_OPTION:
            print(f"\n  [FCM] Executando Fuzzy C-Means com {self.n_rules} clusters...")
        self.fcm_ = FuzzyCMeans(
            n_clusters=self.n_rules,
            m=self.m,
            max_iter=self.max_iter_fcm,
            random_state=self.random_state
        )
        self.fcm_.fit(X, y)
        if config.PRINT_OPTION:
            print(f"  [FCM] Convergiu em {self.fcm_.n_iter_} iterações.")
            print(self.fcm_.summary())

        # Calculate firing strengths (W)
        W = self.fcm_.get_firing_strengths(X)  # (N, R)

        # One-hot encoding
        Y_onehot = np.zeros((n_samples, self.n_classes_))
        for idx, cls in enumerate(self.classes_):
            Y_onehot[y == cls, idx] = 1.0

        # Augmented matrix: [1, x1, ..., xD]
        X_aug = np.hstack([np.ones((n_samples, 1)), X]) 
        D_aug = X_aug.shape[1]

        # Parameters: (n_rules, n_classes, D+1)
        self.consequent_params_ = np.zeros((self.n_rules, self.n_classes_, D_aug))

        if config.PRINT_OPTION:
            print(f"\n  [TS] Estimando parâmetros consequentes ({self.n_rules} regras × {self.n_classes_} classes)...")

        for i in range(self.n_rules):
            # Weight of rule i for each sample
            wi = W[:, i]  # (N,)

            # Weighted least squares: (X^T W X) p = X^T W y
            sqrt_wi = np.sqrt(wi)[:, np.newaxis]  # (N, 1)
            X_w = X_aug * sqrt_wi  # (N, D+1)
            Y_w = Y_onehot * sqrt_wi  # (N, K)

            try:
                params = np.linalg.lstsq(X_w, Y_w, rcond=None)[0]  # (D+1, K)
                self.consequent_params_[i] = params.T  # (K, D+1)
            except np.linalg.LinAlgError:
                if config.PRINT_OPTION:
                    print(f"    [AVISO] Regra {i}: lstsq falhou, usando zeros.")
                self.consequent_params_[i] = np.zeros((self.n_classes_, D_aug))

        if config.PRINT_OPTION:
            print("  [TS] Treinamento concluído.")

        return self

    # Method to get the model outputs for each class
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        # Get firing strengths (W)
        W = self.fcm_.get_firing_strengths(X)  # (N, R)

        # Normalize weights
        W_sum = W.sum(axis=1, keepdims=True)  # (N, 1)
        W_norm = W / np.maximum(W_sum, 1e-300)  # (N, R)

        # Augmented matrix
        X_aug = np.hstack([np.ones((n_samples, 1)), X])  # (N, D+1)

        # Output: y_k = sum_i( w_i_norm * (X_aug @ p_ik^T) )
        Y_pred = np.zeros((n_samples, self.n_classes_))

        for i in range(self.n_rules):
            # Consequent of rule i for each class: (N, K)
            local_output = X_aug @ self.consequent_params_[i].T 
            Y_pred += W_norm[:, i:i+1] * local_output

        return Y_pred

    # Method to predict the class for each sample
    def predict(self, X: np.ndarray) -> np.ndarray:
        Y_pred = self.predict_proba(X)
        indices = np.argmax(Y_pred, axis=1)
        return self.classes_[indices]

    # Method to return the accuracy
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # Method to print the rules
    def print_rules(self, feature_names: list = None):
        import textwrap
        if feature_names is None:
            feature_names = [f"x{d+1}" for d in range(self.n_features_)]

        print("\n" + "=" * 60)
        print("  BASE DE REGRAS FUZZY — TAKAGI-SUGENO (1ª ordem)")
        print("=" * 60)

        for i in range(self.n_rules):
            print(f"\n  REGRA {i+1}:")
            
            # Antecedent
            ante_parts = []
            for d, fname in enumerate(feature_names):
                c = self.fcm_.centers_[i, d]
                s = self.fcm_.sigmas_[i, d]
                ante_parts.append(f"{fname} ≈ {c:.4f} (σ={s:.4f})")
            
            ante_str = " E ".join(ante_parts)
            wrapped_ante = textwrap.fill(ante_str, width=100, 
                                         initial_indent="    SE ", 
                                         subsequent_indent="       ")
            print(wrapped_ante)

            # Consequent
            for k, cls in enumerate(self.classes_):
                params = self.consequent_params_[i, k]
                terms = [f"{params[0]:.4f}"]
                for d, fname in enumerate(feature_names):
                    terms.append(f"{params[d+1]:+.4f}·{fname}")
                eq = " ".join(terms)
                
                wrapped_cons = textwrap.fill(eq, width=100,
                                             initial_indent=f"    ENTÃO y_classe{cls} = ",
                                             subsequent_indent="                        ")
                print(wrapped_cons)

        print("\n" + "=" * 60)
