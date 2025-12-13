import numpy as np

def get_features_from_id(self, id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the features from the features list.
    """
    return (self.features[f][id] for f in range(2))

def are_same(features1: tuple, features2: tuple) -> bool:
    """
    Check if two sets of features are the same.
    """
    for f1, f2 in zip(features1, features2):
        if not np.array_equal(f1, f2):
            return False
    return True
        
def get_state_id(self, features: tuple) -> tuple[int, bool]:
    """
    Check if the features have already been visited.
    """
    ids = np.arange(len(self.features[0]))
    mask = (features[0] == self.features[0]).all(axis=self.feature_axes[0])
    ids = ids[mask]
    for i in range(1,len(self.features)):
        mask = (self.features[i][ids] == features[i]).all(axis=self.feature_axes[i])
        ids = ids[mask]
    
    assert len(ids) <= 1

    if len(ids) > 0:
        return ids[0], True

    return len(self.features[0]), False
    

def get_distances(self, features: np.ndarray) -> int:
    """
    Returns the Manhattan distance between two sets of features.
    """
    distances = np.array([(np.abs(features[i] - self.features[i])).sum(axis=self.feature_axes[i]) for i in range(len(features))])

    return np.dot(self.weights, distances)


