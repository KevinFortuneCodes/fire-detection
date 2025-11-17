# models/random_forest_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .utils import save_model

def train_random_forest(X_train, y_train, model_name='random_forest'):
    """Train Random Forest model with hyperparameter tuning"""
    print("Training Random Forest model...")
    
    # Define parameter grid (limited for speed)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Train with grid search
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Random Forest parameters: {grid_search.best_params_}")
    print(f"Best Random Forest cross-validation score: {grid_search.best_score_:.4f}")
    
    best_rf = grid_search.best_estimator_
    return best_rf

def create_random_forest():
    """Create Random Forest model with default parameters"""
    return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)