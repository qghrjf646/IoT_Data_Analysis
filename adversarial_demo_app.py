"""
Adversarial Machine Learning Interactive Demo
==============================================

An interactive Streamlit application demonstrating exploratory and causative
attacks on machine learning classifiers with video simulations.

Run with: streamlit run adversarial_demo_app.py --server.address=localhost

Authors: CIC-IIoT-2025 Security Analysis Project
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import tempfile
import os
import subprocess
import shutil
import warnings
warnings.filterwarnings('ignore')

# Data paths for real CIC-IIoT dataset
ATTACK_PATH = 'data/attack_data/'
BENIGN_PATH = 'data/benign_data/'

# Page configuration
st.set_page_config(
    page_title="Adversarial ML Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a6fa5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .simulation-status {
        font-size: 1.5rem;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .phase-indicator {
        font-size: 1.2rem;
        color: #4a6fa5;
        text-align: center;
        margin: 1rem 0;
    }
    video {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_real_data(time_interval='1sec', n_samples=500, reduction_method='pca'):
    """Load real CIC-IIoT dataset and reduce to 2D for visualization.
    
    Parameters:
    -----------
    time_interval : str
        Time interval for data sampling (1sec, 2sec, etc.)
    n_samples : int
        Number of samples to use (balanced between attack and benign)
    reduction_method : str
        Method for dimensionality reduction ('pca' or 'feature_selection')
    
    Returns:
    --------
    X : ndarray (n_samples, 2)
        2D feature matrix for visualization
    y : ndarray (n_samples,)
        Binary labels (0=benign, 1=attack)
    scaler : StandardScaler
        Fitted scaler for transformations
    feature_names : list
        Names of the 2 selected features (for labels)
    """
    # Load data
    attack_file = f'{ATTACK_PATH}attack_samples_{time_interval}.csv'
    benign_file = f'{BENIGN_PATH}benign_samples_{time_interval}.csv'
    
    df_attack = pd.read_csv(attack_file)
    df_benign = pd.read_csv(benign_file)
    
    df_attack['is_attack'] = 1
    df_benign['is_attack'] = 0
    
    # Balance the dataset
    samples_per_class = min(n_samples // 2, len(df_attack), len(df_benign))
    df_attack_sampled = df_attack.sample(n=samples_per_class, random_state=42)
    df_benign_sampled = df_benign.sample(n=samples_per_class, random_state=42)
    
    df = pd.concat([df_attack_sampled, df_benign_sampled], ignore_index=True)
    
    # Identify metadata columns to exclude
    metadata_cols = ['device_name', 'device_mac', 'label_full', 'label1', 'label2', 'label3', 'label4',
                     'timestamp', 'timestamp_start', 'timestamp_end', 'is_attack']
    
    # Get feature columns (numerical only)
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Remove columns that are lists/arrays stored as strings
    def is_list_column(col):
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
        if isinstance(sample, str) and sample.startswith('['):
            return True
        return False
    
    list_cols = [col for col in feature_cols if is_list_column(col)]
    feature_cols = [col for col in feature_cols if col not in list_cols]
    
    # Prepare feature matrix and target
    X = df[feature_cols].copy()
    y = df['is_attack'].values
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce to 2D
    if reduction_method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        feature_names = ['PC1 (Principal Component 1)', 'PC2 (Principal Component 2)']
    else:
        # Feature selection: select top 2 most discriminative features
        selector = SelectKBest(f_classif, k=2)
        X_2d = selector.fit_transform(X_scaled, y)
        selected_indices = selector.get_support(indices=True)
        feature_names = [feature_cols[i].replace('network_', '').replace('_', ' ').title() 
                        for i in selected_indices]
    
    # Re-scale the 2D features
    scaler_2d = StandardScaler()
    X_2d = scaler_2d.fit_transform(X_2d)
    
    return X_2d, y, scaler_2d, feature_names


@st.cache_data
def generate_data(n_samples=500, dataset_type='classification'):
    """Generate synthetic data for demonstration."""
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            flip_y=0.05,
            class_sep=1.2,
            random_state=42
        )
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler, ['Feature 1', 'Feature 2']


def get_model(model_name, random_state=42):
    """Get a classifier by name."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Linear SVM': LinearSVC(random_state=random_state, max_iter=5000),
        'SVM (RBF)': SVC(kernel='rbf', random_state=random_state, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    }
    return models.get(model_name, LogisticRegression(random_state=random_state))


def compute_decision_boundary(model, X, resolution=80):
    """Compute decision boundary for plotting."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    return xx, yy, Z


def fgsm_attack(model, X, y, epsilon):
    """Fast Gradient Sign Method attack."""
    if hasattr(model, 'coef_'):
        weights = model.coef_.flatten()
    else:
        weights = np.random.randn(X.shape[1])
    
    perturbation = epsilon * np.sign(weights)
    X_adv = X - np.outer(y * 2 - 1, perturbation)
    return X_adv


def poison_data(X_train, y_train, poison_rate, shift_magnitude=1.0):
    """Poison training data by flipping labels and shifting features."""
    n_poison = int(len(y_train) * poison_rate)
    if n_poison == 0:
        return X_train.copy(), y_train.copy(), []
    
    class_1_idx = np.where(y_train == 1)[0]
    poison_idx = np.random.choice(class_1_idx, size=min(n_poison, len(class_1_idx)), replace=False)
    
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()
    
    y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]
    
    if shift_magnitude > 0:
        centroid_0 = X_train[y_train == 0].mean(axis=0)
        centroid_1 = X_train[y_train == 1].mean(axis=0)
        shift = (centroid_0 - centroid_1) * shift_magnitude * 0.3
        X_poisoned[poison_idx] += shift
    
    return X_poisoned, y_poisoned, list(poison_idx)


def apply_defense(X, defense_type, model=None):
    """Apply defense mechanism to input data."""
    if defense_type == 'None':
        return X, np.ones(len(X), dtype=bool)
    
    elif defense_type == 'Input Validation (Clipping)':
        X_defended = np.clip(X, -3, 3)
        valid_mask = np.ones(len(X), dtype=bool)
        return X_defended, valid_mask
    
    elif defense_type == 'Anomaly Filtering (LOF)':
        lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
        predictions = lof.fit_predict(X)
        valid_mask = predictions == 1
        return X, valid_mask
    
    elif defense_type == 'Feature Squeezing':
        X_defended = np.round(X * 10) / 10
        valid_mask = np.ones(len(X), dtype=bool)
        return X_defended, valid_mask
    
    elif defense_type == 'Gaussian Noise Injection':
        noise = np.random.normal(0, 0.1, X.shape)
        X_defended = X + noise
        valid_mask = np.ones(len(X), dtype=bool)
        return X_defended, valid_mask
    
    return X, np.ones(len(X), dtype=bool)


def create_fgsm_frame(model, X_train, X_test, y_test, X_current, y_pred_current, 
                      xx, yy, Z, epsilon, t, clean_acc, adv_acc, accuracies_so_far, 
                      epsilons_so_far, defense, feature_names=None):
    """Create a single frame for FGSM simulation."""
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Decision boundary with points
    ax1 = axes[0]
    ax1.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot points with colors based on true labels and markers based on predictions
    for label, color in [(0, 'green'), (1, 'red')]:
        mask = y_test == label
        correct = y_pred_current[mask] == y_test[mask]
        
        # Correct predictions
        ax1.scatter(X_current[mask & (y_pred_current == y_test), 0], 
                   X_current[mask & (y_pred_current == y_test), 1],
                   c=color, marker='o', s=60, edgecolors='black', linewidths=0.5,
                   label=f'Class {label} (correct)')
        
        # Incorrect predictions
        ax1.scatter(X_current[mask & (y_pred_current != y_test), 0],
                   X_current[mask & (y_pred_current != y_test), 1],
                   c=color, marker='x', s=80, linewidths=2,
                   label=f'Class {label} (misclassified)')
    
    current_acc = (y_pred_current == y_test).mean()
    ax1.set_title(f'FGSM Attack (ε={epsilon:.2f}) - {int(t*100)}% Applied\n'
                  f'Current Accuracy: {current_acc:.1%}', fontsize=12, fontweight='bold')
    ax1.set_xlabel(feature_names[0], fontsize=10)
    ax1.set_ylabel(feature_names[1], fontsize=10)
    
    # Defense indicator
    if defense != 'None':
        ax1.text(0.02, 0.98, f'Defense: {defense}', transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                color='white')
    
    # Right plot: Accuracy degradation
    ax2 = axes[1]
    ax2.fill_between(epsilons_so_far, accuracies_so_far, alpha=0.3, color='blue')
    ax2.plot(epsilons_so_far, accuracies_so_far, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=clean_acc, color='green', linestyle='--', linewidth=2, 
                label=f'Astute Accuracy: {clean_acc:.1%}')
    ax2.axhline(y=adv_acc, color='red', linestyle='--', linewidth=2,
                label=f'Robust Accuracy: {adv_acc:.1%}')
    
    ax2.set_xlim(0, epsilon * 1.1)
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Accuracy Degradation Over Attack', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Effective Epsilon', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_poisoning_frame(X_poisoned, y_poisoned, poison_idx, X_test, y_test,
                           xx, yy, Z, poison_rate, accuracies, rates_so_far, baseline_acc,
                           feature_names=None):
    """Create a single frame for poisoning simulation."""
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Decision boundary with poisoned points
    ax1 = axes[0]
    ax1.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Clean training points
    clean_mask = np.ones(len(X_poisoned), dtype=bool)
    clean_mask[poison_idx] = False
    
    for label, color in [(0, 'green'), (1, 'red')]:
        mask = clean_mask & (y_poisoned == label)
        ax1.scatter(X_poisoned[mask, 0], X_poisoned[mask, 1],
                   c=color, marker='o', s=40, edgecolors='black', linewidths=0.5, alpha=0.7)
    
    # Poisoned points (highlighted)
    if len(poison_idx) > 0:
        ax1.scatter(X_poisoned[poison_idx, 0], X_poisoned[poison_idx, 1],
                   c='purple', marker='X', s=120, edgecolors='yellow', linewidths=2,
                   label=f'Poisoned ({len(poison_idx)} samples)', zorder=10)
    
    current_acc = accuracies[-1] if accuracies else baseline_acc
    ax1.set_title(f'Causative Attack - Poison Rate: {poison_rate:.1%}\n'
                  f'Accuracy: {current_acc:.1%}', fontsize=12, fontweight='bold')
    ax1.set_xlabel(feature_names[0], fontsize=10)
    ax1.set_ylabel(feature_names[1], fontsize=10)
    if len(poison_idx) > 0:
        ax1.legend(loc='upper right', fontsize=9)
    
    # Right plot: Accuracy vs poison rate
    ax2 = axes[1]
    ax2.fill_between(rates_so_far, accuracies, alpha=0.3, color='red')
    ax2.plot(rates_so_far, accuracies, 'r-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=baseline_acc, color='green', linestyle='--', linewidth=2,
                label=f'Baseline: {baseline_acc:.1%}')
    
    ax2.set_xlim(0, max(poison_rate * 1.1, 0.01))
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Model Accuracy vs Poison Rate', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Poison Rate', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis as percentage
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.tight_layout()
    return fig


def generate_fgsm_video(model, X_train, y_train, X_test, y_test, epsilon, defense, 
                        n_frames, fps, progress_callback, feature_names=None):
    """Generate video frames for FGSM attack simulation."""
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    # Train model
    model.fit(X_train, y_train)
    
    # Compute decision boundary
    xx, yy, Z = compute_decision_boundary(model, X_train)
    
    # Generate adversarial examples
    X_adv = fgsm_attack(model, X_test, y_test, epsilon)
    
    # Apply defense
    if defense != 'None':
        X_defended, valid_mask = apply_defense(X_adv, defense, model)
    else:
        X_defended = X_adv
        valid_mask = np.ones(len(X_adv), dtype=bool)
    
    # Compute accuracies
    y_pred_clean = model.predict(X_test)
    y_pred_adv = model.predict(X_defended)
    
    clean_acc = (y_pred_clean == y_test).mean()
    adv_acc = (y_pred_adv[valid_mask] == y_test[valid_mask]).mean() if valid_mask.any() else 0
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    accuracies_so_far = []
    epsilons_so_far = []
    
    for frame in range(n_frames + 1):
        t = frame / n_frames
        
        # Interpolate between clean and adversarial
        X_current = X_test * (1 - t) + X_adv * t
        y_pred_current = model.predict(X_current)
        current_acc = (y_pred_current == y_test).mean()
        
        accuracies_so_far.append(current_acc)
        epsilons_so_far.append(t * epsilon)
        
        # Create frame
        fig = create_fgsm_frame(
            model, X_train, X_test, y_test, X_current, y_pred_current,
            xx, yy, Z, epsilon, t, clean_acc, adv_acc,
            accuracies_so_far.copy(), epsilons_so_far.copy(), defense, feature_names
        )
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{frame:04d}.png')
        fig.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        frame_paths.append(frame_path)
        
        # Update progress
        progress_callback((frame / n_frames) * 0.9, f"Generating frame {frame + 1}/{n_frames + 1}")
    
    # Compile video
    progress_callback(0.92, "Compiling video...")
    video_path = os.path.join(temp_dir, 'simulation.mp4')
    
    # Use ffmpeg to create video
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        video_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try with different codec or return None
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'mpeg4', '-q:v', '5',
                video_path
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        except:
            # Clean up frames and return None if ffmpeg fails
            for fp in frame_paths:
                os.remove(fp)
            os.rmdir(temp_dir)
            return None, clean_acc, adv_acc, temp_dir
    
    progress_callback(1.0, "Video ready!")
    
    return video_path, clean_acc, adv_acc, temp_dir


def generate_poisoning_video(model_name, X_train, y_train, X_test, y_test, poison_rate,
                             n_frames, fps, progress_callback, feature_names=None):
    """Generate video frames for poisoning attack simulation."""
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    poison_rates = np.linspace(0, poison_rate, n_frames + 1)
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    accuracies = []
    rates_so_far = []
    baseline_acc = None
    
    for frame, rate in enumerate(poison_rates):
        # Poison data
        X_poisoned, y_poisoned, poison_idx = poison_data(X_train, y_train, rate)
        
        # Train model
        model = get_model(model_name)
        model.fit(X_poisoned, y_poisoned)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        accuracies.append(acc)
        rates_so_far.append(rate)
        
        if baseline_acc is None:
            baseline_acc = acc
        
        # Compute decision boundary
        xx, yy, Z = compute_decision_boundary(model, X_train)
        
        # Create frame
        fig = create_poisoning_frame(
            X_poisoned, y_poisoned, poison_idx, X_test, y_test,
            xx, yy, Z, rate, accuracies.copy(), rates_so_far.copy(), baseline_acc,
            feature_names
        )
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{frame:04d}.png')
        fig.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        frame_paths.append(frame_path)
        
        # Update progress
        progress_callback((frame / n_frames) * 0.9, f"Generating frame {frame + 1}/{n_frames + 1}")
    
    # Compile video
    progress_callback(0.92, "Compiling video...")
    video_path = os.path.join(temp_dir, 'simulation.mp4')
    
    # Use ffmpeg to create video
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        video_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'mpeg4', '-q:v', '5',
                video_path
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        except:
            for fp in frame_paths:
                os.remove(fp)
            os.rmdir(temp_dir)
            return None, baseline_acc, accuracies[-1] if accuracies else 0, temp_dir
    
    progress_callback(1.0, "Video ready!")
    
    return video_path, baseline_acc, accuracies[-1], temp_dir


def cleanup_temp_dir(temp_dir):
    """Clean up temporary directory."""
    try:
        shutil.rmtree(temp_dir)
    except:
        pass


def main():
    # Header
    st.markdown('<p class="main-header">🛡️ Adversarial Machine Learning Demo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Simulation of Exploratory and Causative Attacks on IoT Security</p>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Dataset selection - NEW: Real data vs Synthetic
    st.sidebar.subheader("📊 Dataset")
    
    data_source = st.sidebar.radio(
        "Data Source",
        ["Real IoT Data (CIC-IIoT)", "Synthetic Data"],
        help="Choose between real CIC-IIoT-2025 dataset or synthetic data for demonstration."
    )
    
    if data_source == "Real IoT Data (CIC-IIoT)":
        time_interval = st.sidebar.selectbox(
            "Time Interval",
            ["1sec", "2sec", "3sec", "4sec", "5sec"],
            help="Sampling interval for network traffic aggregation."
        )
        
        reduction_method = st.sidebar.selectbox(
            "Feature Reduction",
            ["pca", "feature_selection"],
            format_func=lambda x: "PCA (Principal Components)" if x == "pca" else "Top 2 Discriminative Features",
            help="Method to reduce features to 2D for visualization."
        )
        
        n_samples = st.sidebar.slider(
            "Dataset Size",
            min_value=200, max_value=2000, value=500, step=100,
            help="Number of samples (balanced between attack and benign)"
        )
    else:
        dataset_type = st.sidebar.selectbox(
            "Synthetic Dataset Type",
            ["classification", "moons"],
            format_func=lambda x: "Linear Separable" if x == "classification" else "Non-Linear (Moons)"
        )
        
        n_samples = st.sidebar.slider(
            "Dataset Size",
            min_value=200, max_value=1000, value=500, step=100
        )
    
    st.sidebar.markdown("---")
    
    # Attack type selection
    attack_type = st.sidebar.selectbox(
        "⚔️ Attack Type",
        ["Exploratory (FGSM)", "Causative (Data Poisoning)"],
        help="Exploratory attacks perturb test samples. Causative attacks poison training data."
    )
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "🤖 Detection Model",
        ["Logistic Regression", "Linear SVM", "SVM (RBF)", "Random Forest", "Gradient Boosting"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Attack Parameters")
    
    if attack_type == "Exploratory (FGSM)":
        epsilon = st.sidebar.slider(
            "Epsilon (ε)",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            help="Perturbation magnitude. Higher values = stronger attack."
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("🛡️ Defense Mechanisms")
        
        defense = st.sidebar.selectbox(
            "Active Defense",
            ["None", "Input Validation (Clipping)", "Anomaly Filtering (LOF)", 
             "Feature Squeezing", "Gaussian Noise Injection"],
            help="Defense mechanism to mitigate the attack."
        )
    else:
        poison_rate = st.sidebar.slider(
            "Poison Rate",
            min_value=0.0, max_value=0.5, value=0.2, step=0.05,
            help="Fraction of training data to poison."
        )
        defense = "None"
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎬 Video Settings")
    
    n_frames = st.sidebar.slider(
        "Number of Frames",
        min_value=10, max_value=60, value=30, step=5,
        help="More frames = smoother video but longer generation time"
    )
    
    fps = st.sidebar.slider(
        "Playback Speed (FPS)",
        min_value=2, max_value=15, value=5, step=1,
        help="Frames per second for video playback"
    )
    
    # Load data based on selection
    try:
        if data_source == "Real IoT Data (CIC-IIoT)":
            X, y, scaler, feature_names = load_real_data(time_interval, n_samples, reduction_method)
            data_description = f"CIC-IIoT-2025 ({time_interval})"
        else:
            X, y, scaler, feature_names = generate_data(n_samples, dataset_type)
            data_description = f"Synthetic ({dataset_type})"
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Make sure the data files are in the correct location (data/attack_data/ and data/benign_data/)")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Main content area
    st.markdown("---")
    
    # Show a banner when using real data
    if data_source == "Real IoT Data (CIC-IIoT)":
        st.markdown("""
        <div style='padding: 1rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; border-radius: 10px; margin-bottom: 1rem; text-align: center;'>
            <strong>🌐 Using Real CIC-IIoT-2025 Dataset</strong><br>
            <small>Network traffic data from actual IoT devices - demonstrating real-world attack impact</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Info boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data_source == "Real IoT Data (CIC-IIoT)":
            st.info(f"""
            **Data:** Real CIC-IIoT-2025  
            **Model:** {model_name}  
            **Samples:** {len(X):,}  
            **Features:** {feature_names[0][:20]}...
            """)
        else:
            st.info(f"""
            **Data:** {data_description}  
            **Model:** {model_name}  
            **Samples:** {n_samples}
            """)
    
    with col2:
        if attack_type == "Exploratory (FGSM)":
            st.warning(f"""
            **Epsilon:** {epsilon}  
            **Defense:** {defense}  
            **Test Samples:** {len(X_test)}
            """)
        else:
            st.warning(f"""
            **Poison Rate:** {poison_rate:.0%}  
            **Training Samples:** {len(X_train)}  
            **Poisoned:** ~{int(len(X_train) * poison_rate)}
            """)
    
    with col3:
        if attack_type == "Exploratory (FGSM)":
            st.success("""
            **Attack Goal:** Evade detection at test time  
            **Attacker Knowledge:** Model gradients  
            **Impact:** Misclassification
            """)
        else:
            st.success("""
            **Attack Goal:** Shift decision boundary  
            **Attacker Knowledge:** Training access  
            **Impact:** Systematic errors
            """)
    
    st.markdown("---")
    
    # Run simulation button
    if st.button("Run Simulation", type="primary", use_container_width=True):
        
        # Create containers for status and video
        status_container = st.empty()
        progress_container = st.empty()
        phase_container = st.empty()
        video_container = st.empty()
        
        # Progress callback function
        def update_progress(progress, message):
            progress_container.progress(progress)
            phase_container.markdown(f'<p class="phase-indicator">{message}</p>', 
                                    unsafe_allow_html=True)
        
        # Show simulation running status
        status_container.markdown("""
        <div class="simulation-status">
            <h2>🎬 Simulation in Progress</h2>
            <p>Generating video frames... Please wait.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate video based on attack type
        if attack_type == "Exploratory (FGSM)":
            model = get_model(model_name)
            video_path, clean_acc, adv_acc, temp_dir = generate_fgsm_video(
                model, X_train, y_train, X_test, y_test,
                epsilon, defense, n_frames, fps, update_progress, feature_names
            )
        else:
            video_path, baseline_acc, poisoned_acc, temp_dir = generate_poisoning_video(
                model_name, X_train, y_train, X_test, y_test,
                poison_rate, n_frames, fps, update_progress, feature_names
            )
            clean_acc = baseline_acc
            adv_acc = poisoned_acc
        
        # Clear status and progress
        status_container.empty()
        progress_container.empty()
        phase_container.empty()
        
        # Display video or error
        if video_path and os.path.exists(video_path):
            st.success("Simulation complete! Watch the video below.")
            
            # Read video file and display
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            
            video_container.video(video_bytes)
            
            # Clean up
            cleanup_temp_dir(temp_dir)
        else:
            st.error("""
            Could not compile video. FFmpeg may not be installed.
            
            Install FFmpeg with: `sudo apt-get install ffmpeg`
            """)
            cleanup_temp_dir(temp_dir)
        
        # Summary
        st.markdown("---")
        st.subheader("Simulation Results")
        
        if attack_type == "Exploratory (FGSM)":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Astute Accuracy", f"{clean_acc:.1%}")
            with col2:
                st.metric("Robust Accuracy", f"{adv_acc:.1%}",
                         f"{(adv_acc - clean_acc)*100:+.1f}%")
            with col3:
                robustness = adv_acc / clean_acc if clean_acc > 0 else 0
                st.metric("Robustness Ratio", f"{robustness:.1%}")
            
            # Recommendations
            st.markdown("### Security Recommendations")
            if robustness < 0.5:
                st.error(f"""
                The model is **highly vulnerable** to FGSM attacks (robustness: {robustness:.1%}).
                
                **Suggested mitigations:**
                - Enable adversarial training with augmented samples
                - Use ensemble methods (Random Forest, Gradient Boosting)
                - Implement input validation and anomaly detection
                """)
            elif robustness < 0.8:
                st.warning(f"""
                The model shows **moderate vulnerability** (robustness: {robustness:.1%}).
                
                **Suggested mitigations:**
                - Consider adversarial training
                - Monitor for distribution drift
                - Implement feature squeezing
                """)
            else:
                st.success(f"""
                The model demonstrates **good robustness** (robustness: {robustness:.1%}).
                
                **Maintain security by:**
                - Regular model retraining
                - Continuous monitoring
                - Defense-in-depth strategy
                """)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baseline Accuracy", f"{clean_acc:.1%}")
            with col2:
                st.metric("Poisoned Accuracy", f"{adv_acc:.1%}",
                         f"{(adv_acc - clean_acc)*100:+.1f}%")
            with col3:
                degradation = (clean_acc - adv_acc) / clean_acc if clean_acc > 0 else 0
                st.metric("Accuracy Degradation", f"{degradation:.1%}")
            
            # Recommendations
            st.markdown("### Security Recommendations")
            if degradation > 0.2:
                st.error(f"""
                The model is **highly susceptible** to data poisoning ({degradation:.1%} degradation).
                
                **Suggested mitigations:**
                - Implement robust training data validation
                - Use outlier detection on training samples
                - Maintain trusted data provenance
                - Consider robust learning algorithms
                """)
            elif degradation > 0.1:
                st.warning(f"""
                The model shows **moderate susceptibility** ({degradation:.1%} degradation).
                
                **Suggested mitigations:**
                - Validate training data sources
                - Implement data quality checks
                - Regular model auditing
                """)
            else:
                st.success(f"""
                The model is **relatively robust** to this level of poisoning ({degradation:.1%} degradation).
                
                **Maintain security by:**
                - Continue data validation practices
                - Monitor for anomalous training patterns
                - Regular security audits
                """)
    
    # Educational content
    with st.expander("📚 Learn More About Adversarial ML"):
        st.markdown("""
        ### Attack Taxonomy
        
        **Exploratory Attacks (Evasion)**
        - Occur at **test time** (inference)
        - Attacker **perturbs inputs** to cause misclassification
        - Model remains unchanged
        - Example: FGSM, PGD, C&W attacks
        
        **Causative Attacks (Poisoning)**
        - Occur at **training time**
        - Attacker **corrupts training data**
        - Model learns incorrect decision boundaries
        - Example: Label flipping, backdoor attacks
        
        ### Key Terminology
        
        - **Astute Accuracy**: Model performance on clean, unperturbed data
        - **Robust Accuracy**: Model performance under adversarial conditions
        - **Robustness Ratio**: Robust accuracy / Astute accuracy
        
        ### Defense Strategies
        
        1. **Adversarial Training**: Include adversarial examples in training
        2. **Input Validation**: Detect and reject anomalous inputs
        3. **Ensemble Methods**: Combine multiple models for robustness
        4. **Feature Squeezing**: Reduce input precision to remove perturbations
        5. **Data Sanitization**: Validate and clean training data
        """)
    
    with st.expander("📊 About the CIC-IIoT-2025 Dataset"):
        st.markdown("""
        ### Real-World IoT Network Traffic Data
        
        The CIC-IIoT-2025 dataset contains **real network traffic** collected from IoT devices,
        including both benign traffic and various attack types.
        
        **Data Characteristics:**
        - Aggregated at configurable time intervals (1-10 seconds)
        - 60+ network features including packet counts, sizes, timing, TCP flags, etc.
        - Multiple attack categories: DDoS, DoS, reconnaissance, and more
        
        **Feature Categories:**
        - **Network Packets**: Count, size, header information
        - **TCP Flags**: SYN, ACK, FIN, RST distributions
        - **Timing**: Inter-packet delays, flow duration
        - **IP Information**: Source/destination diversity, TTL values
        
        **Visualization Approach:**
        - For 2D visualization, we reduce the ~60 features to 2 dimensions
        - **PCA**: Preserves maximum variance in the data
        - **Feature Selection**: Uses the 2 most discriminative features
        
        Using real data demonstrates how adversarial attacks would affect **actual IoT security systems**,
        making the security recommendations more applicable to real-world deployments.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        CIC-IIoT-2025 Security Analysis Project | ML Security - EPITA SCIA 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
