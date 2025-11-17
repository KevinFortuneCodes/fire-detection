# models/knn_model.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from .utils import save_model

def train_knn(X_train, y_train, model_name='knn'):
    """Train KNN model with hyperparameter tuning"""
    print("Training KNN model...")
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # Train with grid search
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best KNN parameters: {grid_search.best_params_}")
    print(f"Best KNN cross-validation score: {grid_search.best_score_:.4f}")
    
    best_knn = grid_search.best_estimator_
    return best_knn

def create_knn():
    """Create KNN model with default parameters"""
    return KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')