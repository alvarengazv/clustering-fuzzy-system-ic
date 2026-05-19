"""
Fuzzy C-Means (FCM) Clustering

Implementação do algoritmo Fuzzy C-Means para formação da base de regras
do sistema fuzzy Takagi-Sugeno.
"""

import numpy as np


class FuzzyCMeans:
    """
    Implementação do Fuzzy C-Means.

    Parâmetros
    ----------
    n_clusters : int
        Número de clusters (regras fuzzy).
    m : float
        Expoente de fuzzificação (m > 1). Quanto maior, mais "fuzzy".
        Valor típico: 2.0.
    max_iter : int
        Número máximo de iterações.
    tol : float
        Tolerância para convergência (variação máxima na matriz U).
    random_state : int ou None
        Semente para reprodutibilidade.
    """

    def __init__(self, n_clusters: int = 4, m: float = 2.0,
                 max_iter: int = 300, tol: float = 1e-6,
                 random_state: int = None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Resultados após fit
        self.centers_ = None       # (n_clusters, n_features)
        self.U_ = None             # (n_clusters, n_samples) — graus de pertinência
        self.n_iter_ = 0
        self.sigmas_ = None        # (n_clusters, n_features) — desvios por cluster

    def _init_membership(self, n_samples: int) -> np.ndarray:
        """Inicializa a matriz de pertinência U aleatoriamente, normalizada."""
        rng = np.random.default_rng(self.random_state)
        U = rng.random((self.n_clusters, n_samples))
        # Normalizar colunas para que somem 1
        U = U / U.sum(axis=0, keepdims=True)
        return U

    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Calcula os centros dos clusters a partir de U."""
        Um = U ** self.m  # (C, N)
        # centers[i] = sum_j(u_ij^m * x_j) / sum_j(u_ij^m)
        centers = Um @ X / Um.sum(axis=1, keepdims=True)  # (C, D)
        return centers

    def _update_membership(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Atualiza a matriz de pertinência U com base nas distâncias."""
        n_samples = X.shape[0]
        C = self.n_clusters
        exp = 2.0 / (self.m - 1.0)

        # Distâncias euclidianas: dist[i, j] = ||x_j - c_i||
        # X: (N, D), centers: (C, D)
        dists = np.zeros((C, n_samples))
        for i in range(C):
            diff = X - centers[i]  # (N, D)
            dists[i] = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)

        # Evitar divisão por zero
        dists = np.maximum(dists, 1e-10)

        U = np.zeros((C, n_samples))
        for i in range(C):
            # u_ij = 1 / sum_k (d_ij / d_kj)^exp
            ratio_sum = np.zeros(n_samples)
            for k in range(C):
                ratio_sum += (dists[i] / dists[k]) ** exp
            U[i] = 1.0 / ratio_sum

        return U

    def _compute_sigmas(self, X: np.ndarray, U: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Calcula os desvios padrão fuzzy por cluster e por atributo.
        Usado para definir as funções de pertinência Gaussianas nas regras.
        
        sigma_i_d = sqrt( sum_j(u_ij^m * (x_jd - c_id)^2) / sum_j(u_ij^m) )
        """
        Um = U ** self.m  # (C, N)
        C, D = centers.shape
        sigmas = np.zeros((C, D))

        for i in range(C):
            diff = X - centers[i]  # (N, D)
            weighted_sq = Um[i, :, np.newaxis] * (diff ** 2)  # (N, D)
            sigmas[i] = np.sqrt(weighted_sq.sum(axis=0) / Um[i].sum())

        # Evitar sigma zero
        sigmas = np.maximum(sigmas, 1e-6)
        return sigmas

    def fit(self, X: np.ndarray) -> 'FuzzyCMeans':
        """
        Executa o algoritmo FCM.

        Parâmetros
        ----------
        X : np.ndarray de forma (n_samples, n_features)
            Dados de entrada (apenas atributos, sem classe).

        Retorna
        -------
        self
        """
        n_samples, n_features = X.shape

        # Inicializar U
        U = self._init_membership(n_samples)

        for iteration in range(1, self.max_iter + 1):
            # Atualizar centros
            centers = self._update_centers(X, U)

            # Atualizar pertinência
            U_new = self._update_membership(X, centers)

            # Verificar convergência
            delta = np.max(np.abs(U_new - U))
            U = U_new
            self.n_iter_ = iteration

            if delta < self.tol:
                break

        self.centers_ = centers
        self.U_ = U
        self.sigmas_ = self._compute_sigmas(X, U, centers)

        return self

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula os graus de pertinência para novos dados.

        Retorna
        -------
        U : np.ndarray de forma (n_clusters, n_samples)
        """
        return self._update_membership(X, self.centers_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna o cluster com maior pertinência para cada amostra."""
        U = self.predict_membership(X)
        return np.argmax(U, axis=0)

    def get_firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula a força de disparo (firing strength) de cada regra para
        cada amostra, usando funções de pertinência Gaussianas derivadas
        dos centros e sigmas do FCM.

        w_i(x) = prod_d exp(-(x_d - c_id)^2 / (2 * sigma_id^2))

        Retorna
        -------
        W : np.ndarray de forma (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        C = self.n_clusters
        W = np.ones((n_samples, C))

        for i in range(C):
            for d in range(X.shape[1]):
                exponent = -((X[:, d] - self.centers_[i, d]) ** 2) / \
                           (2.0 * self.sigmas_[i, d] ** 2)
                W[:, i] *= np.exp(exponent)

        # Evitar zeros
        W = np.maximum(W, 1e-300)
        return W

    def summary(self) -> str:
        """Retorna um resumo do resultado do FCM."""
        lines = []
        lines.append(f"Fuzzy C-Means — {self.n_clusters} clusters, m={self.m}")
        lines.append(f"  Iterações: {self.n_iter_}")
        lines.append(f"  Centros dos clusters:")
        for i, c in enumerate(self.centers_):
            lines.append(f"    Cluster {i}: {np.round(c, 4)}")
        lines.append(f"  Sigmas (desvios):")
        for i, s in enumerate(self.sigmas_):
            lines.append(f"    Cluster {i}: {np.round(s, 4)}")
        return "\n".join(lines)
