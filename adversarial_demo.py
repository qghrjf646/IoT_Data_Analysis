"""
Adversarial Machine Learning Demonstration
==========================================

This script demonstrates both exploratory and causative attacks on machine learning
classifiers, with decision boundary visualizations.

Exploratory Attack: FGSM (Fast Gradient Sign Method)
    - Perturbs test samples to cause misclassification
    - Does not modify the training process
    
Causative Attack: Data Poisoning
    - Injects malicious samples into training data
    - Shifts the decision boundary to attacker's advantage

Author: CIC-IIoT-2025 Security Analysis Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(n_samples=1000, n_features=2, n_informative=2):
    """Generate synthetic binary classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.1,
        class_sep=1.0,
        random_state=42
    )
    return X, y


def plot_decision_boundary(model, X, y, ax, title, cmap_light=None, cmap_bold=None):
    """Plot decision boundary for a classifier."""
    if cmap_light is None:
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    if cmap_bold is None:
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def compute_gradient_logistic(model, X, y):
    """Compute gradient for logistic regression."""
    # Logistic regression gradient: X^T * (sigmoid(X*w) - y)
    w = model.coef_.flatten()
    b = model.intercept_[0]
    z = X @ w + b
    sigmoid = 1 / (1 + np.exp(-z))
    
    # Gradient of cross-entropy loss w.r.t. input X
    gradient = np.outer((sigmoid - y), w)
    return gradient


def fgsm_attack(model, X, y, epsilon=0.3):
    """
    Fast Gradient Sign Method (FGSM) - Exploratory Attack
    
    Generates adversarial examples by adding small perturbations
    in the direction of the gradient sign.
    
    Parameters:
    -----------
    model : trained classifier (must have coef_ attribute)
    X : input samples
    y : true labels
    epsilon : perturbation magnitude
    
    Returns:
    --------
    X_adv : adversarial examples
    """
    gradient = compute_gradient_logistic(model, X, y)
    perturbation = epsilon * np.sign(gradient)
    X_adv = X + perturbation
    return X_adv


def causative_attack_label_flipping(X_train, y_train, poison_rate=0.1):
    """
    Causative Attack: Label Flipping
    
    Flips labels of a subset of training samples to poison the model.
    
    Parameters:
    -----------
    X_train : training features
    y_train : training labels
    poison_rate : fraction of samples to poison
    
    Returns:
    --------
    X_poisoned : poisoned training features (unchanged)
    y_poisoned : poisoned training labels
    poison_idx : indices of poisoned samples
    """
    n_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), n_poison, replace=False)
    
    y_poisoned = y_train.copy()
    y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]  # Flip labels
    
    return X_train, y_poisoned, poison_idx


def causative_attack_gradient_based(X_train, y_train, model, target_samples, 
                                     poison_rate=0.1, learning_rate=0.1, n_iterations=50):
    """
    Causative Attack: Gradient-Based Poisoning
    
    Crafts poisoning samples that maximize loss on target samples
    when added to the training set.
    
    Parameters:
    -----------
    X_train : training features
    y_train : training labels
    model : model class to poison
    target_samples : samples to cause misclassification on
    poison_rate : fraction of training data to add as poison
    learning_rate : step size for gradient ascent
    n_iterations : number of optimization iterations
    
    Returns:
    --------
    X_poisoned : poisoned training set
    y_poisoned : labels for poisoned training set
    poison_points : the crafted poison samples
    """
    n_poison = max(1, int(len(y_train) * poison_rate))
    
    # Initialize poison points near class boundary
    poison_points = np.random.randn(n_poison, X_train.shape[1]) * 0.5
    poison_labels = np.ones(n_poison)  # Assign to class 1
    
    for iteration in range(n_iterations):
        # Combine training data with current poison points
        X_combined = np.vstack([X_train, poison_points])
        y_combined = np.hstack([y_train, poison_labels])
        
        # Train model on poisoned data
        temp_model = LogisticRegression(max_iter=1000, random_state=42)
        temp_model.fit(X_combined, y_combined)
        
        # Compute gradient to maximize loss on target samples
        w = temp_model.coef_.flatten()
        b = temp_model.intercept_[0]
        
        # Update poison points to shift decision boundary
        for i in range(n_poison):
            # Move poison point in direction that increases loss on targets
            direction = -w  # Move perpendicular to decision boundary
            poison_points[i] += learning_rate * direction
    
    X_poisoned = np.vstack([X_train, poison_points])
    y_poisoned = np.hstack([y_train, poison_labels])
    
    return X_poisoned, y_poisoned, poison_points


def demonstrate_exploratory_attack():
    """Demonstrate FGSM exploratory attack with visualization."""
    print("=" * 60)
    print("EXPLORATORY ATTACK DEMONSTRATION: FGSM")
    print("=" * 60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on clean data
    astute_accuracy = model.score(X_test_scaled, y_test)
    print(f"Astute Accuracy (clean test data): {astute_accuracy:.4f}")
    
    # Generate adversarial examples with different epsilon values
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.5]
    robust_accuracies = []
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, eps in enumerate(epsilons):
        if eps == 0:
            X_adv = X_test_scaled.copy()
        else:
            X_adv = fgsm_attack(model, X_test_scaled, y_test, epsilon=eps)
        
        robust_acc = model.score(X_adv, y_test)
        robust_accuracies.append(robust_acc)
        print(f"Robust Accuracy (epsilon={eps:.1f}): {robust_acc:.4f}")
        
        # Plot
        ax = axes[i]
        plot_decision_boundary(model, X_adv, y_test, ax, 
                              f'FGSM Attack (epsilon={eps:.1f})\nRobust Acc: {robust_acc:.2f}')
    
    # Plot accuracy vs epsilon
    ax = axes[-1]
    ax.plot(epsilons, robust_accuracies, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
    ax.set_xlabel('Epsilon (Perturbation Magnitude)', fontsize=10)
    ax.set_ylabel('Robust Accuracy', fontsize=10)
    ax.set_title('Accuracy Degradation under FGSM Attack', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fgsm_exploratory_attack.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: fgsm_exploratory_attack.png")
    
    return model, scaler


def demonstrate_causative_attack():
    """Demonstrate causative (poisoning) attack with visualization."""
    print("\n" + "=" * 60)
    print("CAUSATIVE ATTACK DEMONSTRATION: DATA POISONING")
    print("=" * 60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train clean model
    clean_model = LogisticRegression(max_iter=1000, random_state=42)
    clean_model.fit(X_train_scaled, y_train)
    clean_accuracy = clean_model.score(X_test_scaled, y_test)
    print(f"Clean Model Accuracy: {clean_accuracy:.4f}")
    
    # Different poison rates
    poison_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    poisoned_accuracies = []
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, rate in enumerate(poison_rates):
        if rate == 0:
            X_poison = X_train_scaled.copy()
            y_poison = y_train.copy()
        else:
            X_poison, y_poison, _ = causative_attack_label_flipping(
                X_train_scaled, y_train, poison_rate=rate
            )
        
        # Train on poisoned data
        poisoned_model = LogisticRegression(max_iter=1000, random_state=42)
        poisoned_model.fit(X_poison, y_poison)
        
        poisoned_acc = poisoned_model.score(X_test_scaled, y_test)
        poisoned_accuracies.append(poisoned_acc)
        print(f"Poisoned Model Accuracy (poison_rate={rate:.2f}): {poisoned_acc:.4f}")
        
        # Plot decision boundary
        ax = axes[i]
        plot_decision_boundary(poisoned_model, X_test_scaled, y_test, ax,
                              f'Label Flipping ({int(rate*100)}% poisoned)\nAccuracy: {poisoned_acc:.2f}')
    
    plt.tight_layout()
    plt.savefig('causative_label_flipping_attack.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: causative_label_flipping_attack.png")
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.array(poison_rates) * 100, poisoned_accuracies, 'ro-', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess')
    ax.set_xlabel('Poison Rate (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Degradation under Causative Attack', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('causative_attack_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved: causative_attack_summary.png")


def demonstrate_decision_boundary_shift():
    """Visualize how decision boundary shifts under different attacks."""
    print("\n" + "=" * 60)
    print("DECISION BOUNDARY SHIFT COMPARISON")
    print("=" * 60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=400)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Clean model
    clean_model = LogisticRegression(max_iter=1000, random_state=42)
    clean_model.fit(X_train_scaled, y_train)
    plot_decision_boundary(clean_model, X_test_scaled, y_test, axes[0, 0],
                          f'Clean Model\nAccuracy: {clean_model.score(X_test_scaled, y_test):.2f}')
    
    # 2. Under FGSM attack (exploratory)
    X_adv = fgsm_attack(clean_model, X_test_scaled, y_test, epsilon=0.3)
    robust_acc = clean_model.score(X_adv, y_test)
    plot_decision_boundary(clean_model, X_adv, y_test, axes[0, 1],
                          f'Exploratory Attack (FGSM, eps=0.3)\nRobust Accuracy: {robust_acc:.2f}')
    
    # 3. Causative attack - Label flipping
    X_poison, y_poison, _ = causative_attack_label_flipping(X_train_scaled, y_train, poison_rate=0.15)
    poisoned_model = LogisticRegression(max_iter=1000, random_state=42)
    poisoned_model.fit(X_poison, y_poison)
    plot_decision_boundary(poisoned_model, X_test_scaled, y_test, axes[1, 0],
                          f'Causative Attack (15% Label Flip)\nAccuracy: {poisoned_model.score(X_test_scaled, y_test):.2f}')
    
    # 4. Causative attack - Gradient-based poisoning
    X_grad_poison, y_grad_poison, poison_pts = causative_attack_gradient_based(
        X_train_scaled, y_train, clean_model, X_test_scaled[:10], 
        poison_rate=0.05, learning_rate=0.5, n_iterations=30
    )
    grad_poisoned_model = LogisticRegression(max_iter=1000, random_state=42)
    grad_poisoned_model.fit(X_grad_poison, y_grad_poison)
    
    ax = axes[1, 1]
    plot_decision_boundary(grad_poisoned_model, X_test_scaled, y_test, ax,
                          f'Causative Attack (Gradient Poisoning)\nAccuracy: {grad_poisoned_model.score(X_test_scaled, y_test):.2f}')
    # Mark poison points
    ax.scatter(poison_pts[:, 0], poison_pts[:, 1], c='green', marker='*', s=200, 
               edgecolors='black', linewidths=1.5, label='Poison Points', zorder=5)
    ax.legend(loc='upper right')
    
    plt.suptitle('Decision Boundary Under Different Attack Scenarios', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('decision_boundary_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: decision_boundary_comparison.png")


def create_animated_attack_demo():
    """Create frame-by-frame visualization for attack progression."""
    print("\n" + "=" * 60)
    print("ATTACK PROGRESSION VISUALIZATION")
    print("=" * 60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Create progression frames
    epsilons = np.linspace(0, 0.5, 10)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, eps in enumerate(epsilons):
        X_adv = fgsm_attack(model, X_test_scaled, y_test, epsilon=eps)
        acc = model.score(X_adv, y_test)
        
        plot_decision_boundary(model, X_adv, y_test, axes[i],
                              f'eps={eps:.2f}, acc={acc:.2f}')
    
    plt.suptitle('FGSM Attack Progression: Increasing Perturbation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fgsm_attack_progression.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: fgsm_attack_progression.png")
    
    # Causative attack progression
    poison_rates = np.linspace(0, 0.3, 10)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, rate in enumerate(poison_rates):
        if rate == 0:
            X_poison, y_poison = X_train_scaled.copy(), y_train.copy()
        else:
            X_poison, y_poison, _ = causative_attack_label_flipping(
                X_train_scaled, y_train, poison_rate=rate
            )
        
        poisoned_model = LogisticRegression(max_iter=1000, random_state=42)
        poisoned_model.fit(X_poison, y_poison)
        acc = poisoned_model.score(X_test_scaled, y_test)
        
        plot_decision_boundary(poisoned_model, X_test_scaled, y_test, axes[i],
                              f'poison={rate:.0%}, acc={acc:.2f}')
    
    plt.suptitle('Causative Attack Progression: Increasing Poison Rate', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('causative_attack_progression.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: causative_attack_progression.png")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# ADVERSARIAL MACHINE LEARNING DEMONSTRATION")
    print("# CIC-IIoT-2025 Security Analysis Project")
    print("#" * 70)
    
    # Run demonstrations
    demonstrate_exploratory_attack()
    demonstrate_causative_attack()
    demonstrate_decision_boundary_shift()
    create_animated_attack_demo()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  1. fgsm_exploratory_attack.png")
    print("  2. causative_label_flipping_attack.png")
    print("  3. causative_attack_summary.png")
    print("  4. decision_boundary_comparison.png")
    print("  5. fgsm_attack_progression.png")
    print("  6. causative_attack_progression.png")
    print("\nKey Concepts Demonstrated:")
    print("  - Exploratory Attack: FGSM perturbs TEST samples")
    print("  - Causative Attack: Poisoning corrupts TRAINING data")
    print("  - Both attacks shift/exploit decision boundaries")
    print("  - Astute accuracy: Performance on clean data")
    print("  - Robust accuracy: Performance under adversarial conditions")


if __name__ == "__main__":
    main()
