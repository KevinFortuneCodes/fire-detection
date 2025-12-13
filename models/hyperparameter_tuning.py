# models/hyperparameter_tuning.py

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

def tune_knn(X_train, y_train, cv=3, n_iter=20, random_state=42):
    """Hyperparameter tuning for KNN"""
    print("Tuning KNN hyperparameters...")
    
    param_dist = {
        'n_neighbors': list(range(3, 21, 2)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2, 3],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    
    knn = KNeighborsClassifier()
    
    random_search = RandomizedSearchCV(
        knn,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"  Best KNN parameters: {random_search.best_params_}")
    print(f"  Best KNN CV score: {random_search.best_score_:.4f}")
    print(f"  Tuning time: {tuning_time:.1f} seconds")
    
    return random_search.best_estimator_

def tune_decision_tree(X_train, y_train, cv=3, n_iter=30, random_state=42):
    """Hyperparameter tuning for Decision Tree"""
    print("Tuning Decision Tree hyperparameters...")
    
    param_dist = {
        'max_depth': [5, 10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9],
        'splitter': ['best', 'random']
    }
    
    dt = DecisionTreeClassifier(random_state=random_state)
    
    random_search = RandomizedSearchCV(
        dt,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"  Best Decision Tree parameters: {random_search.best_params_}")
    print(f"  Best Decision Tree CV score: {random_search.best_score_:.4f}")
    print(f"  Tuning time: {tuning_time:.1f} seconds")
    
    return random_search.best_estimator_

def tune_random_forest(X_train, y_train, cv=3, n_iter=30, random_state=42):
    """Hyperparameter tuning for Random Forest"""
    print("Tuning Random Forest hyperparameters...")
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"  Best Random Forest parameters: {random_search.best_params_}")
    print(f"  Best Random Forest CV score: {random_search.best_score_:.4f}")
    print(f"  Tuning time: {tuning_time:.1f} seconds")
    
    return random_search.best_estimator_