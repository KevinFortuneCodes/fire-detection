# models/decision_tree_model.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from .utils import save_model

def train_decision_tree(X_train, y_train, model_name='decision_tree'):
    """Train Decision Tree model with hyperparameter tuning"""
    print("Training Decision Tree model...")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    # Train with grid search
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Decision Tree parameters: {grid_search.best_params_}")
    print(f"Best Decision Tree cross-validation score: {grid_search.best_score_:.4f}")
    
    best_dt = grid_search.best_estimator_
    return best_dt

def create_decision_tree():
    """Create Decision Tree model with default parameters"""
    return DecisionTreeClassifier(max_depth=10, random_state=42)