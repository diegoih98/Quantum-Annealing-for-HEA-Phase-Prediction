import pandas as pd
import numpy as np
import itertools
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import dimod
from dimod import BinaryQuadraticModel
from datetime import datetime



# Hyperparameter tuning using classical SVM (to find best gamma)
def tune_hyperparameters(X, y,C):
    #param_grid = {'C': [C], 'gamma': [0.125,0.25,0.5,1,2,4,8], 'kernel': ['rbf']}
    param_grid = {'C': [C], 'kernel': ['linear']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X, y)
    return 1#grid_search.best_params_['gamma']

# Gaussian Kernel Function
def gaussian_kernel(x, y, gamma):
    return np.dot(x, y) #np.exp(-gamma * np.linalg.norm(x - y)**2)

# Construct QUBO matrix
def construct_QUBO(X, y, B, K, xi, gamma, kernel_threshold=1e-5):
    N = len(X)
    size = N * K
    Q = np.zeros((size, size))

    for n in range(N):
        for m in range(N):
            kernel_value = gaussian_kernel(X[n], X[m], gamma)
            # Explicitly reduce couplers by ignoring negligible kernel values
            if kernel_value < kernel_threshold:
                continue
            
            for k in range(K):
                for j in range(K):
                    idx_n = n*K + k
                    idx_m = m*K + j
                    Q[idx_n, idx_m] += 0.5 * (B**(k+j)) * y[n] * y[m] * (kernel_value + xi)

    for n in range(N):
        for k in range(K):
            idx = n*K + k
            Q[idx, idx] -= B**k

    return Q

# Solve QUBO using Quantum Annealer
def solve_qubo(Q):
    bqm = BinaryQuadraticModel(Q, "BINARY")
    
    # Measure CPU and Wall-Clock Time
    start_cpu_time = time.process_time()
    start_wall_time = time.time()

    # Solve using dimod's Simulated Annealing Sampler (purely classical, no QPU)
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=1000)

    end_cpu_time = time.process_time()
    end_wall_time = time.time()

    
    # Compute elapsed times
    cpu_time_taken = end_cpu_time - start_cpu_time  # Measures CPU computation time
    wall_time_taken = end_wall_time - start_wall_time  # Measures real-world elapsed time
            
    return sampleset, cpu_time_taken, wall_time_taken

def build_hybrid(sampleset, X, B, K):
    # Retrieve the lowest-energy solution (first/best sample)
    sample = sampleset.first.sample

    alphas = np.zeros(len(X))
    for n in range(len(X)):
        alpha_n = sum(B**k * sample[n*K + k] for k in range(K))
        alphas[n] = alpha_n

    return alphas

def calculate_bias(final_alpha, X, y, gamma, C):
    support_vectors = (final_alpha > 1e-5)  # Identify support vectors
    numerator = sum(final_alpha[n] * (C - final_alpha[n]) * 
                    (y[n] - sum(final_alpha[m] * y[m] * gaussian_kernel(X[m], X[n], gamma)
                                for m in range(len(X))))
                    for n in range(len(X)) if support_vectors[n])
    
    denominator = sum(final_alpha[n] * (C - final_alpha[n]) for n in range(len(X)) if support_vectors[n])

    return numerator / denominator if denominator != 0 else 0  # Avoid division by zero

# Decision function
def decision_function(X_new, X, y, final_alpha, b, gamma):
    return np.sign(sum(final_alpha[n]*y[n]*gaussian_kernel(X[n], X_new, gamma)
                   for n in range(len(X))) + b)

def evaluate_model(X, y, final_alpha, b, gamma, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    # Ensure X is a NumPy array
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X  
    y = y.to_numpy() if isinstance(y, pd.Series) else y  

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]  # Direct indexing since y is now an array

        y_pred = [decision_function(X_test[i], X_train, y_train, final_alpha, b, gamma) for i in range(len(X_test))]
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))

    return np.mean(accuracies), np.mean(f1_scores)

# Test model function
def test_model(X_train, y_train, X_test, y_test, final_alpha, b, gamma):
    # Ensure inputs are NumPy arrays
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
    y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
    y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

    y_pred = [decision_function(X_test[i], X_train, y_train, final_alpha, b, gamma) for i in range(len(X_test))]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, C, gamma, cv_folds=5):

    print(f"Training SVM with C={C}, gamma={gamma}, using {cv_folds}-fold CV...")

    # Initialize SVM with RBF kernel
    #svm_model = SVC(C=C, gamma=gamma, kernel='rbf')
    svm_model = SVC(C=C, kernel='linear')

    # Perform cross-validation on training set
    cv_accuracy = cross_val_score(svm_model, X_train, y_train, cv=cv_folds, scoring='accuracy').mean()
    cv_f1 = cross_val_score(svm_model, X_train, y_train, cv=cv_folds, scoring='f1').mean()

    # Train the model on the entire training set
    svm_model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred_test = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)


    # Return results as a dictionary
    return {
        "C": C,
        "gamma": gamma,
        "CV Accuracy": cv_accuracy,
        "CV F1-Score": cv_f1,
        "Test Accuracy": test_accuracy,
        "Test F1-Score": test_f1
    }

database = input("What database to work on? Database.xlsx: ")
data = pd.read_excel(database)

# Extract features and labels
X_raw = data.drop(columns=["Composition", "Phase Category"])
# Explicitly select your desired features
selected_features = ['Smix', 'Hmix', 'Omega', 'Phi2', 'Eta', 'K1Cr', 'MismatchR', 'E2_E0', 'DeltaX', 'VEC', 'PFP_A1', 'PFP_B2', 'PFP_Laves', 'Radius', 'Ionic_E_2nd', 'Ionic_E_3rd', 'Hardness']  
X_raw = X_raw[selected_features]


y = data["Phase Category"]

# Handle missing values
X_raw = X_raw.fillna(X_raw.mean())
scaler = StandardScaler()
X_scaled_df = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)

X_scaled = scaler.fit_transform(X_raw)

# Load Test Dataset
test_database = "Validation_HEAs.xlsx"
sheet_name = input("Enter the sheet name (Laves, Heusler, RB2): ")
test_data = pd.read_excel(test_database, sheet_name=sheet_name)


X_test = test_data.drop(columns=["Composition", "Phase Category"], errors='ignore')
X_test = X_test[selected_features]

y_test = test_data["Phase Category"]
X_test = X_test.fillna(X_test.mean())
scaler = StandardScaler()
X_test_scaled_df = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)
X_test_scaled = scaler.fit_transform(X_test)


# Define parameter ranges
B_values = [2, 3, 5]
K_values = [2, 3]
xi_values = [0, 1, 5]

results = []

# Iterate over all combinations of (B, K, xi)
for B, K, xi in itertools.product(B_values, K_values, xi_values):

    print(f"Tuning hyperparameters for B={B}, K={K}, xi={xi}...")

    # Compute C based on B and K
    C = sum(B**k for k in range(K))

    # Tune hyperparameters
    gamma = tune_hyperparameters(X_scaled, y, C)

    
    SVC_results = train_and_evaluate_svm(X_scaled_df, y, X_test_scaled_df, y_test, C=C, gamma=gamma)
    
    print("Constructing QUBO...")
    Q = construct_QUBO(X_scaled, y, B, K, xi, gamma)

    print("Solving QUBO...")
    sampleset, cpu_time_taken, wall_time_taken= solve_qubo(Q)

    print("Ensemble alpha...")
    final_alpha = build_hybrid(sampleset, X_scaled, B, K) 

    print("Calculating bias...")
    b = calculate_bias(final_alpha, X_scaled, y, gamma, C)

    print("Evaluating training set 5-CV...")
    accuracy, f1 = evaluate_model(X_scaled_df, y, final_alpha, b, gamma)

    print("Evaluating test set...")
    accuracy_test, f1_test = test_model(X_scaled_df, y, X_test_scaled_df, y_test, final_alpha, b, gamma)

    print(f"Completed: B={B}, K={K}, xi={xi} -> Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

    # Store results
    results.append({
        'B': B,
        'K': K,
        'xi': xi,
        'gamma' : gamma,
        'final_alpha': final_alpha,
        'b': b,
        'C' : C,
        'Features' : selected_features,
        'Accuracy_training': accuracy,
        'F1-Score_training': f1,
        'Accuracy_test': accuracy_test,
        'F1-Score_test': f1_test,
        'SVC_Accuracy_training': SVC_results["CV Accuracy"],
        'SVC_F1-Score_training': SVC_results["CV F1-Score"],
        'SVC_Accuracy_test':SVC_results["Test Accuracy"],
        'SVC_F1-Score_test': SVC_results["Test F1-Score"],
        'CPU Time (s)' : cpu_time_taken,
        'Wall Time (s)' : wall_time_taken,
    })

results_df = pd.DataFrame(results)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"QA_grid_search_results_{database}_{timestamp}.csv"
results_df.to_csv(filename, index=False)
print(f"Results saved to {filename}")
