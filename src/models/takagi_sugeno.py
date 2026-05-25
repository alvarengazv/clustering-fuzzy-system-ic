import numpy as np
from models.fuzzy_cmeans import FuzzyCMeans
import config

class TakagiSugenoClassifier:

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TakagiSugenoClassifier':
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

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

        W = self.fcm_.get_firing_strengths(X)

        Y_onehot = np.zeros((n_samples, self.n_classes_))
        for idx, cls in enumerate(self.classes_):
            Y_onehot[y == cls, idx] = 1.0

        X_aug = np.hstack([np.ones((n_samples, 1)), X]) 
        D_aug = X_aug.shape[1]

        self.consequent_params_ = np.zeros((self.n_rules, self.n_classes_, D_aug))

        if config.PRINT_OPTION:
            print(f"\n  [TS] Estimando parâmetros consequentes ({self.n_rules} regras × {self.n_classes_} classes)...")

        for i in range(self.n_rules):
            wi = W[:, i]

            sqrt_wi = np.sqrt(wi)[:, np.newaxis]
            X_w = X_aug * sqrt_wi
            Y_w = Y_onehot * sqrt_wi

            try:
                params = np.linalg.lstsq(X_w, Y_w, rcond=None)[0]
                self.consequent_params_[i] = params.T
            except np.linalg.LinAlgError:
                if config.PRINT_OPTION:
                    print(f"    [AVISO] Regra {i}: lstsq falhou, usando zeros.")
                self.consequent_params_[i] = np.zeros((self.n_classes_, D_aug))

        if config.PRINT_OPTION:
            print("  [TS] Treinamento concluído.")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        W = self.fcm_.get_firing_strengths(X)

        W_sum = W.sum(axis=1, keepdims=True)
        W_norm = W / np.maximum(W_sum, 1e-300)
        X_aug = np.hstack([np.ones((n_samples, 1)), X])

        Y_pred = np.zeros((n_samples, self.n_classes_))

        for i in range(self.n_rules):
            local_output = X_aug @ self.consequent_params_[i].T 
            Y_pred += W_norm[:, i:i+1] * local_output

        return Y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y_pred = self.predict_proba(X)
        indices = np.argmax(Y_pred, axis=1)
        return self.classes_[indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def print_rules(self, feature_names: list = None):
        import textwrap
        if feature_names is None:
            feature_names = [f"x{d+1}" for d in range(self.n_features_)]

        print("\n" + "=" * 60)
        print("  BASE DE REGRAS FUZZY — TAKAGI-SUGENO (1ª ordem)")
        print("=" * 60)

        for i in range(self.n_rules):
            print(f"\n  REGRA {i+1}:")
            
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
