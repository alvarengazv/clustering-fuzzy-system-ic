import numpy as np

# Fuzzy C-Means (FCM)
class FuzzyCMeans:
    # Initialization
    def __init__(self, n_clusters: int = 4, m: float = 2.0,
                 max_iter: int = 300, tol: float = 1e-6,
                 random_state: int = None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None      
        self.U_ = None           
        self.n_iter_ = 0
        self.sigmas_ = None    

    # Initialization of membership matrix
    def _init_membership(self, n_samples: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        U = rng.random((self.n_clusters, n_samples))
        # Normalize columns to sum to 1
        U = U / U.sum(axis=0, keepdims=True)
        return U

    # Update centers
    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        Um = U ** self.m
        centers = Um @ X / Um.sum(axis=1, keepdims=True)
        return centers

    # Update membership
    def _update_membership(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        C = self.n_clusters
        exp = 2.0 / (self.m - 1.0)

        # Euclidean distances
        dists = np.zeros((C, n_samples))
        for i in range(C):
            diff = X - centers[i]  
            dists[i] = np.sqrt(np.sum(diff ** 2, axis=1)) 

        # Avoid division by zero
        dists = np.maximum(dists, 1e-10)

        U = np.zeros((C, n_samples))
        for i in range(C):
            ratio_sum = np.zeros(n_samples)
            for k in range(C):
                ratio_sum += (dists[i] / dists[k]) ** exp
            U[i] = 1.0 / ratio_sum

        return U

    # Compute sigmas
    def _compute_sigmas(self, X: np.ndarray, U: np.ndarray, centers: np.ndarray) -> np.ndarray:
        Um = U ** self.m
        C, D = centers.shape
        sigmas = np.zeros((C, D))

        for i in range(C):
            diff = X - centers[i]  
            weighted_sq = Um[i, :, np.newaxis] * (diff ** 2) 
            sigmas[i] = np.sqrt(weighted_sq.sum(axis=0) / Um[i].sum())

        # Avoid division by zero
        sigmas = np.maximum(sigmas, 1e-6)
        return sigmas

    # Fit method
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'FuzzyCMeans':
        n_samples, n_features = X.shape

        if y is not None:
            # Supervised FCM: calculates optimal initial centers as the class averages
            classes = np.unique(y)
            centers = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                # Cyclic mapping if the number of classes differs from n_clusters
                cls = classes[i % len(classes)]
                X_cls = X[y == cls]
                if len(X_cls) > 0:
                    centers[i] = X_cls.mean(axis=0)
                else:
                    centers[i] = X.mean(axis=0) + np.random.normal(0, 0.1, n_features)
            
            # Initialize the membership matrix U based on the initial centers
            U = self._update_membership(X, centers)

        else:
            # Unsupervised FCM: traditional random initialization
            U = self._init_membership(n_samples)

        # FCM algorithm iterations
        for iteration in range(1, self.max_iter + 1):
            # Update centers
            centers = self._update_centers(X, U)

            # Update membership
            U_new = self._update_membership(X, centers)

            # Check convergence
            delta = np.max(np.abs(U_new - U))
            U = U_new
            self.n_iter_ = iteration

            if delta < self.tol:
                break

        self.centers_ = centers
        self.U_ = U
        self.sigmas_ = self._compute_sigmas(X, U, centers)

        return self

    # Predict membership method
    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        return self._update_membership(X, self.centers_)

    # Method to predict the class for each sample
    def predict(self, X: np.ndarray) -> np.ndarray:
        U = self.predict_membership(X)
        return np.argmax(U, axis=0)

    # Method to get the firing strengths for each rule
    def get_firing_strengths(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        C = self.n_clusters
        W = np.ones((n_samples, C))

        for i in range(C):
            for d in range(X.shape[1]):
                exponent = -((X[:, d] - self.centers_[i, d]) ** 2) / \
                           (2.0 * self.sigmas_[i, d] ** 2)
                W[:, i] *= np.exp(exponent)

        # Avoid division by zero
        W = np.maximum(W, 1e-300)
        return W

    # Method to return the summary
    def summary(self) -> str:
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
