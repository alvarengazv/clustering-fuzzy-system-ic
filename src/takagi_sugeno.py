"""
Classificador Fuzzy Takagi-Sugeno

Usa os clusters do Fuzzy C-Means como base de regras.
Cada regra tem:
  - Antecedente: funções de pertinência Gaussianas (centros e sigmas do FCM)
  - Consequente: modelo linear de primeira ordem (por classe)

Para classificação multi-classe:
  - Cada regra i possui um vetor de parâmetros para cada classe k
  - y_ik(x) = a_ik0 + a_ik1*x1 + ... + a_ikD*xD
  - Saída final por classe: ŷ_k = Σ(w_i * y_ik) / Σ(w_i)
  - Classe predita = argmax_k(ŷ_k)
"""

import numpy as np
from fuzzy_cmeans import FuzzyCMeans


class TakagiSugenoClassifier:
    """
    Classificador Fuzzy Takagi-Sugeno de primeira ordem.

    Parâmetros
    ----------
    n_rules : int
        Número de regras fuzzy (= número de clusters FCM).
    m : float
        Expoente de fuzzificação do FCM.
    max_iter_fcm : int
        Iterações máximas do FCM.
    random_state : int ou None
        Semente para reprodutibilidade.
    """

    def __init__(self, n_rules: int = 4, m: float = 2.0,
                 max_iter_fcm: int = 300, random_state: int = None):
        self.n_rules = n_rules
        self.m = m
        self.max_iter_fcm = max_iter_fcm
        self.random_state = random_state

        self.fcm_ = None
        self.consequent_params_ = None  # (n_rules, n_classes, n_features+1)
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TakagiSugenoClassifier':
        """
        Treina o modelo Takagi-Sugeno.

        1. Aplica FCM nos dados de entrada para obter centros e sigmas.
        2. Estima os parâmetros consequentes via mínimos quadrados ponderados.

        Parâmetros
        ----------
        X : np.ndarray (n_samples, n_features)
            Atributos de entrada.
        y : np.ndarray (n_samples,)
            Rótulos das classes.
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # --- Passo 1: Fuzzy C-Means ---
        print(f"\n  [FCM] Executando Fuzzy C-Means com {self.n_rules} clusters...")
        self.fcm_ = FuzzyCMeans(
            n_clusters=self.n_rules,
            m=self.m,
            max_iter=self.max_iter_fcm,
            random_state=self.random_state
        )
        self.fcm_.fit(X, y)
        print(f"  [FCM] Convergiu em {self.fcm_.n_iter_} iterações.")
        print(self.fcm_.summary())

        # --- Passo 2: Calcular forças de disparo (firing strengths) ---
        W = self.fcm_.get_firing_strengths(X)  # (N, R)

        # --- Passo 3: Estimar parâmetros consequentes ---
        # Para cada regra i e classe k, resolver:
        #   y_ik = [1, x1, ..., xD] @ p_ik
        # usando mínimos quadrados ponderados com peso w_i(x)
        #
        # Codificação one-hot do alvo
        Y_onehot = np.zeros((n_samples, self.n_classes_))
        for idx, cls in enumerate(self.classes_):
            Y_onehot[y == cls, idx] = 1.0

        # Matriz aumentada: [1, x1, ..., xD]
        X_aug = np.hstack([np.ones((n_samples, 1)), X])  # (N, D+1)
        D_aug = X_aug.shape[1]

        # Parâmetros: (n_rules, n_classes, D+1)
        self.consequent_params_ = np.zeros((self.n_rules, self.n_classes_, D_aug))

        print(f"\n  [TS] Estimando parâmetros consequentes ({self.n_rules} regras × {self.n_classes_} classes)...")

        for i in range(self.n_rules):
            # Pesos da regra i para cada amostra
            wi = W[:, i]  # (N,)

            # Mínimos quadrados ponderados: (X^T W X) p = X^T W y
            sqrt_wi = np.sqrt(wi)[:, np.newaxis]  # (N, 1)
            X_w = X_aug * sqrt_wi  # (N, D+1)
            Y_w = Y_onehot * sqrt_wi  # (N, K)

            # Resolver para cada classe
            # X_w^T X_w p = X_w^T Y_w
            # Usar pseudo-inversa para estabilidade numérica
            try:
                params = np.linalg.lstsq(X_w, Y_w, rcond=None)[0]  # (D+1, K)
                self.consequent_params_[i] = params.T  # (K, D+1)
            except np.linalg.LinAlgError:
                print(f"    [AVISO] Regra {i}: lstsq falhou, usando zeros.")
                self.consequent_params_[i] = np.zeros((self.n_classes_, D_aug))

        print("  [TS] Treinamento concluído.")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula as saídas do modelo para cada classe (não normalizadas).

        Retorna
        -------
        Y_pred : np.ndarray (n_samples, n_classes)
        """
        n_samples = X.shape[0]

        # Forças de disparo
        W = self.fcm_.get_firing_strengths(X)  # (N, R)

        # Normalizar pesos
        W_sum = W.sum(axis=1, keepdims=True)  # (N, 1)
        W_norm = W / np.maximum(W_sum, 1e-300)  # (N, R)

        # Matriz aumentada
        X_aug = np.hstack([np.ones((n_samples, 1)), X])  # (N, D+1)

        # Saída: y_k = sum_i( w_i_norm * (X_aug @ p_ik^T) )
        Y_pred = np.zeros((n_samples, self.n_classes_))

        for i in range(self.n_rules):
            # Consequente da regra i para todas as classes: (N, K)
            local_output = X_aug @ self.consequent_params_[i].T  # (N, K)
            Y_pred += W_norm[:, i:i+1] * local_output

        return Y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz a classe para cada amostra.

        Retorna
        -------
        y_pred : np.ndarray (n_samples,) com rótulos originais das classes.
        """
        Y_pred = self.predict_proba(X)
        indices = np.argmax(Y_pred, axis=1)
        return self.classes_[indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Retorna a acurácia."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def print_rules(self, feature_names: list = None):
        """Imprime as regras fuzzy do modelo."""
        if feature_names is None:
            feature_names = [f"x{d+1}" for d in range(self.n_features_)]

        print("\n" + "=" * 60)
        print("  BASE DE REGRAS FUZZY — TAKAGI-SUGENO (1ª ordem)")
        print("=" * 60)

        for i in range(self.n_rules):
            print(f"\n  REGRA {i+1}:")
            
            # Antecedente
            ante_parts = []
            for d, fname in enumerate(feature_names):
                c = self.fcm_.centers_[i, d]
                s = self.fcm_.sigmas_[i, d]
                ante_parts.append(f"{fname} ≈ {c:.4f} (σ={s:.4f})")
            
            print(f"    SE {' E '.join(ante_parts)}")

            # Consequente
            for k, cls in enumerate(self.classes_):
                params = self.consequent_params_[i, k]
                terms = [f"{params[0]:.4f}"]
                for d, fname in enumerate(feature_names):
                    terms.append(f"{params[d+1]:+.4f}·{fname}")
                eq = " ".join(terms)
                print(f"    ENTÃO y_classe{cls} = {eq}")

        print("\n" + "=" * 60)
