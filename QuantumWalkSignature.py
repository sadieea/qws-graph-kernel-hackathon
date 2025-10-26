import numpy as np
import networkx as nx
from time import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==============================================================================
# SECTION 1: Quantum Walk Signature (QWS)
# ==============================================================================

def get_quantum_walk_signature(G, time_points, normalize_time=True):
    """Computes Quantum Walk Signature feature vector for a single graph."""
    if G.number_of_nodes() == 0:
        return np.zeros(len(time_points))

    L = nx.normalized_laplacian_matrix(G).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    lambda_max = np.max(np.abs(eigvals)) if np.max(np.abs(eigvals)) != 0 else 1.0
    if normalize_time:
        time_points = time_points / lambda_max

    V_sq = eigvecs * eigvecs
    signature = []
    for t in time_points:
        phase = np.exp(-1j * eigvals * t)
        diag_U = V_sq @ phase
        p_uu = np.abs(diag_U) ** 2
        signature.append(np.mean(p_uu.real))
    return np.array(signature)

# ==============================================================================
# SECTION 2: Extended Feature Vector (EFV) Baseline
# ==============================================================================

def wiener_index(G):
    total = 0
    for u, lengths in nx.all_pairs_shortest_path_length(G):
        total += sum(lengths.values())
    return 0.5 * total

def randic_index(G):
    s = 0.0
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        if du > 0 and dv > 0:
            s += 1.0 / np.sqrt(du * dv)
    return s

def estrada_index(G):
    A = nx.to_numpy_array(G)
    eigvals = np.linalg.eigvalsh(A)
    return np.sum(np.exp(eigvals))

def get_efv_features(G):
    return np.array([wiener_index(G), randic_index(G), estrada_index(G)])

# ==============================================================================
# SECTION 3: Manual Weisfeiler–Lehman (WL) Kernel
# ==============================================================================

def weisfeiler_lehman_kernel(graphs, h=3):
    """
    Compute the Weisfeiler-Lehman subtree kernel matrix between all graphs.
    Based on iterative label refinement and feature histogram comparison.
    """
    # Step 1: initial labels = node degrees (as strings)
    labels = [{node: str(deg) for node, deg in G.degree()} for G in graphs]

    # Feature dictionaries for each graph
    feature_dicts = [dict() for _ in graphs]

    # Iterative WL relabeling
    for it in range(h):
        new_labels = []
        for i, G in enumerate(graphs):
            new_label = {}
            for node in G.nodes():
                neighborhood = sorted([labels[i][nbr] for nbr in G.neighbors(node)])
                new_label[node] = labels[i][node] + "_" + "_".join(neighborhood)
            # compress labels to unique integers
            unique = {lab: str(idx) for idx, lab in enumerate(sorted(set(new_label.values())))}
            for node in new_label:
                new_label[node] = unique[new_label[node]]
            labels[i] = new_label

            # Update feature counts
            for l in new_label.values():
                feature_dicts[i][l] = feature_dicts[i].get(l, 0) + 1
            new_labels.append(new_label)

    # Convert feature dicts to consistent vector form
    all_labels = sorted(set(l for d in feature_dicts for l in d.keys()))
    label_index = {l: idx for idx, l in enumerate(all_labels)}
    X = np.zeros((len(graphs), len(all_labels)))
    for i, d in enumerate(feature_dicts):
        for l, c in d.items():
            X[i, label_index[l]] = c

    # Linear kernel (dot product)
    K = X @ X.T
    # Normalize
    norms = np.sqrt(np.diag(K))
    K /= norms[:, None]
    K /= norms[None, :]
    return K

# ==============================================================================
# SECTION 4: Evaluation Functions
# ==============================================================================
def evaluate_features_svm(X, y, param_grid=None):
    if param_grid is None:
        param_grid = {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 0.1, 1]}
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf'))
    ])
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    acc = grid.best_score_
    f1_scores = cross_val_f1(X, y, grid.best_estimator_)
    return acc, f1_scores

def cross_val_f1(X, y, estimator):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1s = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
    return np.mean(f1s)

def evaluate_precomputed_kernel(K, y, C_values=[0.1, 1, 10]):
    """Evaluates a precomputed kernel SVM with manual CV."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(K, y):
        K_train = K[np.ix_(train_idx, train_idx)]
        K_test = K[np.ix_(test_idx, train_idx)]
        y_train, y_test = y[train_idx], y[test_idx]
        best_acc, best_C = -1, 1
        for C in C_values:
            clf = SVC(kernel='precomputed', C=C)
            clf.fit(K_train, y_train)
            y_pred_val = clf.predict(K_test)
            acc_val = accuracy_score(y_test, y_pred_val)
            if acc_val > best_acc:
                best_acc, best_C = acc_val, C
        # final train + test
        clf_final = SVC(kernel='precomputed', C=best_C)
        clf_final.fit(K_train, y_train)
        y_pred = clf_final.predict(K_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
    return np.mean(accs), np.mean(f1s)

# ==============================================================================
# SECTION 5: Dataset Loader (TUDataset local files)
# ==============================================================================
def load_tudataset_txt(dataset_path):
    """
    Load TU-Dortmund dataset (.txt files) with support for comma or space delimiters.
    """

    inner_folder = next((os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, f))), dataset_path)

    dataset_name = os.path.basename(inner_folder)

    def load_txt(filename):
        """Loads a text file and auto-detects comma or space delimiter."""
        with open(filename, 'r') as f:
            first_line = f.readline()
        delimiter = ',' if ',' in first_line else None
        return np.loadtxt(filename, dtype=int, delimiter=delimiter)

    # File paths
    edges_file = f"{dataset_path}/A.txt"
    indicator_file = f"{dataset_path}/graph_indicator.txt"
    labels_file = f"{dataset_path}/graph_labels.txt"

    # Load data
    edges = load_txt(edges_file)
    indicators = load_txt(indicator_file)
    labels = load_txt(labels_file)

    num_graphs = int(np.max(indicators))
    graphs = []

    for g_idx in range(1, num_graphs + 1):
        nodes = np.where(indicators == g_idx)[0] + 1
        edge_subset = [tuple(e - 1) for e in edges if e[0] in nodes and e[1] in nodes]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edge_subset)
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)

    print(f"✅ Loaded {num_graphs} graphs from {dataset_name}")
    return graphs, np.array(labels)

# ==============================================================================
# SECTION 6: Full Pipeline
# ==============================================================================
def run_pipeline(dataset_path, dataset_name):
    graphs, labels = load_tudataset_txt(dataset_path)
    y = np.array(labels)

    # 1. Quantum Walk Signature
    print(f"\n--- {dataset_name}: Quantum Walk Signature ---")
    times = np.linspace(0.1, 5.0, 20)
    X_qws = np.array([get_quantum_walk_signature(G, times) for G in graphs])
    qws_acc, qws_f1 = evaluate_features_svm(X_qws, y)
    print(f"QWS Accuracy: {qws_acc:.4f}, F1: {qws_f1:.4f}")

    # 2. Extended Feature Vector
    print(f"--- {dataset_name}: Extended Feature Vector ---")
    X_efv = np.array([get_efv_features(G) for G in graphs])
    efv_acc, efv_f1 = evaluate_features_svm(X_efv, y)
    print(f"EFV Accuracy: {efv_acc:.4f}, F1: {efv_f1:.4f}")

    # 3. Weisfeiler–Lehman Kernel
    print(f"--- {dataset_name}: WL Kernel ---")
    K_wl = weisfeiler_lehman_kernel(graphs, h=3)
    wl_acc, wl_f1 = evaluate_precomputed_kernel(K_wl, y)
    print(f"WL Accuracy: {wl_acc:.4f}, F1: {wl_f1:.4f}")

    return {
        "Dataset": dataset_name,
        "QWS_Acc": qws_acc, "QWS_F1": qws_f1,
        "EFV_Acc": efv_acc, "EFV_F1": efv_f1,
        "WL_Acc": wl_acc, "WL_F1": wl_f1
    }

# ==============================================================================
# SECTION 7: Main
# ==============================================================================
if __name__ == "__main__":
    datasets = {
        "MUTAG": "datasets/MUTAG",
        "PROTEINS": "datasets/PROTEINS",
        "PTC_MR": "datasets/PTC_MR",
        "NCI1": "datasets/NCI1",
        "AIDS": "datasets/AIDS"
    }

    all_results = []
    for name, path in datasets.items():
        try:
            res = run_pipeline(path, name)
            all_results.append(res)
        except Exception as e:
            print(f"Error on {name}: {e}")

    print("\n================ FINAL RESULTS ================")
    print(f"{'Dataset':<12} | {'Method':<8} | {'Accuracy':<9} | {'F1':<9}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['Dataset']:<12} | QWS     | {r['QWS_Acc']:<9.4f} | {r['QWS_F1']:<9.4f}")
        print(f"{'':<12} | EFV     | {r['EFV_Acc']:<9.4f} | {r['EFV_F1']:<9.4f}")
        print(f"{'':<12} | WL      | {r['WL_Acc']:<9.4f} | {r['WL_F1']:<9.4f}")
        print("-" * 50)
