import numpy as np

class CustomNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.class_means = None
        self.class_variances = None
    
    def fit(self, X, y):

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_means = np.zeros((n_classes, n_features))
        self.class_variances = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = X_c.shape[0] / X.shape[0]
            self.class_means[i, :] = X_c.mean(axis=0)
            self.class_variances[i, :] = X_c.var(axis=0) + 1e-7
        
        return self
    
    def _gaussian_probability(self, x, mean, variance):
        exponent = np.exp(-((x - mean)**2 / (2 * variance)))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponent
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        probabilities = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            prior = np.log(self.class_priors[i])
            conditional = np.sum(np.log(
                self._gaussian_probability(
                    X, 
                    self.class_means[i, :], 
                    self.class_variances[i, :]
                ) + 1e-10 
            ), axis=1)
            
            
            probabilities[:, i] = prior + conditional
        
        probabilities = np.exp(probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
