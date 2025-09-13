import pandas as pd
import numpy as np
import time
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split, KFold
import dimod
from minorminer import find_embedding
import networkx as nx
import sys
import matplotlib.pyplot as plt
import itertools
from tabulate import tabulate
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler



# Train weak classifiers using the entire dataset 
def create_weak_classifiers(X, y):
    classifiers = {}
    
    for feature in X.columns:
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X[[feature]], y)
        classifiers[feature] = stump
    
    return classifiers

# ---------------------------- Build Weak Classifier Matrix (H) ----------------------------
def build_H(classifiers, X):
    """Constructs the prediction matrix H from weak classifiers."""
    H = np.array([classifiers[feature].predict(X[[feature]]) for feature in classifiers], dtype=float).T
    N = len(classifiers)
    H /= N
    return H

# ---------------------------- Construct BQM ----------------------------
def build_bqm(H, y, lam):
    """Builds the Binary Quadratic Model (BQM) for feature selection in QBoost."""
    n_samples, n_classifiers = H.shape
    samples_factor = 1.0 / n_samples  # Normalization factor

    bqm = dimod.BQM('BINARY')

    # Add linear terms (self-interactions)
    for i in range(n_classifiers):
        bqm.add_variable(i, lam - 2.0 * samples_factor * np.dot(H[:, i], y) + samples_factor * np.dot(H[:, i], H[:, i]))

    # Add quadratic terms (pairwise interactions)
    for i in range(n_classifiers):
        for j in range(i + 1, n_classifiers):
            bqm.add_interaction(i, j, 2.0 * samples_factor * np.dot(H[:, i], H[:, j]))

    return bqm


def evaluate_model(X, y, selected_features, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train selected weak classifiers on the current fold
        selected_classifiers = {feature: DecisionTreeClassifier(max_depth=1).fit(X_train[[feature]], y_train) for feature in selected_features}
        
        # Assemble strong classifier and evaluate
        votes = np.zeros(len(X_test))
        for feature in selected_features:
            votes += selected_classifiers[feature].predict(X_test[[feature]])
        
        y_pred = np.sign(votes)
        accuracies.append(accuracy_score(y_test, y_pred))
        #f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    
    return np.mean(accuracies), np.mean(f1_scores)

def test_model(X_training, y_training, X_test, y_test, selected_features):
    
    selected_classifiers = {feature: DecisionTreeClassifier(max_depth=1).fit(X_training[[feature]], y_training) for feature in selected_features}
    
    votes = np.zeros(len(X_test))
    for feature in selected_features:
        votes += selected_classifiers[feature].predict(X_test[[feature]])
    
    y_pred = np.sign(votes)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    #f1 = f1_score(y_test, y_pred, average='macro')
    
    return accuracy, f1

# ---------------------------- Load and Process Data ----------------------------
database = input("What database to work on? Database.xlsx: ")
data = pd.read_excel(database)

# Extract features and labels
X_raw = data.drop(columns=["Composition", "Phase Category"])
y = data["Phase Category"]

# Handle missing values
X_raw = X_raw.fillna(X_raw.mean())


# Load Test Dataset
test_database = "Validation_HEAs.xlsx"
sheet_name = input("Enter the sheet name (Laves, Heusler, RB2): ")
test_data = pd.read_excel(test_database, sheet_name=sheet_name)


X_test = test_data.drop(columns=["Composition", "Phase Category"], errors='ignore')
y_test = test_data["Phase Category"]
X_test = X_test.fillna(X_test.mean())

#start program
#-------------------Grid Search for Lambda Values -------------------

n_rounds = 30
random_states = range(n_rounds)

# Store results across rounds
all_round_results = []

# Regularization parameter (controls complexity)
lambda_val = [0.01]  # Adjust as needed

start_total_time = time.time()

for seed in random_states:
    print(f"\n--- Undersampling Round {seed+1}/{n_rounds} ---")



     # Random Undersampling
    rus = RandomUnderSampler(random_state=seed)
    X_resampled, y_resampled = rus.fit_resample(X_raw, y)

    # Weak classifiers using resampled data
    weak_classifiers = create_weak_classifiers(X_resampled, y_resampled)
    H_train = build_H(weak_classifiers, X_resampled)

    # Store results for different lambda
    qa_results = []

    for lambda_ in lambda_val:
        bqm = build_bqm(H_train, y_resampled.values, lambda_)
        
        # Measure CPU and Wall-Clock Time
        start_cpu_time = time.process_time()
        start_wall_time = time.time()

        # Solve using dimod's Simulated Annealing Sampler (purely classical, no QPU)
        sampler = dimod.SimulatedAnnealingSampler()
        results_qa = sampler.sample(bqm, num_reads=100)

        end_cpu_time = time.process_time()
        end_wall_time = time.time()

    
        # Compute elapsed times
        cpu_time_taken = end_cpu_time - start_cpu_time  # Measures CPU computation time
        wall_time_taken = end_wall_time - start_wall_time  # Measures real-world elapsed time
        
        solutions = []
        for sample, energy in results_qa.data(['sample', 'energy']):
            selected_indices = [i for i, value in sample.items() if value > 0]
            selected_features = [list(weak_classifiers.keys())[i] for i in selected_indices]
            solutions.append((energy, selected_features))

        # Sort solutions by energy (ascending)
        solutions.sort(key=lambda x: x[0])

        # Retrieve the best (lowest energy) solution
        best_energy, best_features = solutions[0]

        accuracy, f1 = evaluate_model(X_resampled, y_resampled, best_features)
        accuracy_test, f1_test = test_model(X_resampled, y_resampled, X_test, y_test, best_features)

        qa_results.append({
            'Seed': seed,
            'Lambda': lambda_,
            'Selected Features': best_features,
            'Num Features': len(best_features),
            'Accuracy (train)': accuracy,
            'F1-score (train)': f1,
            'Accuracy (test)': accuracy_test,
            'F1-score (test)': f1_test,
            'CPU Time (s)' : cpu_time_taken,
            'Wall Time (s)' : wall_time_taken,
        })

    # Save results from this round
    all_round_results.extend(qa_results)


   



# ---------------------------- Save and Display Results ----------------------------
results_df = pd.DataFrame(all_round_results)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"QA_grid_search_results_{database}_{timestamp}.csv"
results_df.to_csv(filename, index=False)
print(f"Results saved to {filename}")
# Compute mean results by Lambda
summary = results_df.groupby('Lambda').agg({
    'Accuracy (train)': ['mean', 'std'],
    'F1-score (train)': ['mean', 'std'],
    'Accuracy (test)': ['mean', 'std'],
    'F1-score (test)': ['mean', 'std'],
    'Num Features': ['mean', 'std']
})

print("\nSummary of results after 30 undersampling rounds:")
print(summary)
