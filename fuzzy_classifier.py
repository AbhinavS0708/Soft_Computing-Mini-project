"""
=============================================================
  Soft Computing Mini Project 1
  Title  : Fuzzy Logic Classifier for Iris Dataset
  Author : Abhishek Yadav
  System Id: 2026303464
=============================================================

Description:
  This project implements a Fuzzy Logic Classification system
  using the classic Iris flower dataset. It uses:
    - Triangular / Trapezoidal membership functions
    - Mamdani-style fuzzy inference
    - Centre of Gravity (CoG) defuzzification

  The fuzzy system classifies an iris flower into:
    - Setosa
    - Versicolor
    - Virginica

  based on two features: Sepal Length and Petal Length.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  1.  MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────────────────────

def trimf(x, abc):
    """Triangular membership function.
    
    Args:
        x   : scalar or array, input value(s)
        abc : [a, b, c] – left foot, peak, right foot
    Returns:
        membership degree in [0, 1]
    """
    a, b, c = abc
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-10),
                                     (c - x) / (c - b + 1e-10)))

def trapmf(x, abcd):
    """Trapezoidal membership function.
    
    Args:
        x    : scalar or array, input value(s)
        abcd : [a, b, c, d] – left foot, left shoulder, right shoulder, right foot
    Returns:
        membership degree in [0, 1]
    """
    a, b, c, d = abcd
    left  = (x - a) / (b - a + 1e-10)
    right = (d - x) / (d - c + 1e-10)
    return np.maximum(0, np.minimum(np.minimum(left, 1), right))


# ─────────────────────────────────────────────────────────────
#  2.  FUZZIFICATION
# ─────────────────────────────────────────────────────────────

def fuzzify_sepal_length(sl):
    """Fuzzify sepal length (cm) into three linguistic variables.
    
    Linguistic terms : short (Setosa), medium (Versicolor), long (Virginica)
    Range            : [4.0, 8.5]
    """
    short  = trapmf(sl, [4.0, 4.0, 5.0, 5.8])
    medium = trimf (sl, [5.0, 5.8, 6.8])
    long_  = trapmf(sl, [6.0, 6.8, 8.5, 8.5])
    return {"short": short, "medium": medium, "long": long_}

def fuzzify_petal_length(pl):
    """Fuzzify petal length (cm) into three linguistic variables.
    
    Linguistic terms : small (Setosa), medium (Versicolor), large (Virginica)
    Range            : [1.0, 7.0]
    """
    small  = trapmf(pl, [1.0, 1.0, 2.0, 2.5])
    medium = trimf (pl, [2.0, 3.5, 5.0])
    large  = trapmf(pl, [4.5, 5.5, 7.0, 7.0])
    return {"small": small, "medium": medium, "large": large}


# ─────────────────────────────────────────────────────────────
#  3.  FUZZY INFERENCE (Mamdani rules)
# ─────────────────────────────────────────────────────────────

def infer(sl_memberships, pl_memberships):
    """Apply Mamdani-style fuzzy rules.
    
    Rules:
      R1: IF sepal SHORT  AND petal SMALL  -> Setosa
      R2: IF sepal MEDIUM AND petal MEDIUM -> Versicolor
      R3: IF sepal LONG   AND petal LARGE  -> Virginica
      R4: IF sepal SHORT  AND petal MEDIUM -> Versicolor   (overlap)
      R5: IF sepal MEDIUM AND petal SMALL  -> Setosa       (overlap)
      R6: IF sepal MEDIUM AND petal LARGE  -> Virginica    (overlap)
      R7: IF sepal LONG   AND petal MEDIUM -> Versicolor   (overlap)
    
    Returns:
        dict with firing strengths for each class.
    """
    sl = sl_memberships
    pl = pl_memberships

    # Firing strength = min(antecedent degrees)
    r1 = min(sl["short"],  pl["small"])     # -> Setosa
    r2 = min(sl["medium"], pl["medium"])    # -> Versicolor
    r3 = min(sl["long"],   pl["large"])     # -> Virginica
    r4 = min(sl["short"],  pl["medium"])    # -> Versicolor
    r5 = min(sl["medium"], pl["small"])     # -> Setosa
    r6 = min(sl["medium"], pl["large"])     # -> Virginica
    r7 = min(sl["long"],   pl["medium"])    # -> Versicolor

    # Aggregate (max) outputs per class
    setosa     = max(r1, r5)
    versicolor = max(r2, r4, r7)
    virginica  = max(r3, r6)

    return {"Setosa": setosa, "Versicolor": versicolor, "Virginica": virginica}


# ─────────────────────────────────────────────────────────────
#  4.  DEFUZZIFICATION  (winner-takes-all / scoring)
# ─────────────────────────────────────────────────────────────

def defuzzify(firing_strengths):
    """Convert fuzzy output strengths to a crisp class label.
    
    Uses a simple winner-takes-all strategy on the firing strengths.
    Returns the class with the highest firing strength.
    """
    return max(firing_strengths, key=firing_strengths.get)


# ─────────────────────────────────────────────────────────────
#  5.  FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def fuzzy_classify(sepal_length, petal_length):
    """End-to-end fuzzy classification for a single sample.
    
    Args:
        sepal_length : float, sepal length in cm
        petal_length : float, petal length in cm
    Returns:
        predicted class label (str)
    """
    sl_mem = fuzzify_sepal_length(sepal_length)
    pl_mem = fuzzify_petal_length(petal_length)
    firing = infer(sl_mem, pl_mem)
    return defuzzify(firing)


# ─────────────────────────────────────────────────────────────
#  6.  LOAD DATA & EVALUATE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Soft Computing Project 1 - Fuzzy Logic Classifier")
    print("  Dataset: Iris  |  Features: Sepal & Petal Length")
    print("=" * 60)

    # Load Iris dataset
    iris = load_iris()
    X = iris.data[:, [0, 2]]   # sepal_length, petal_length
    y = iris.target
    class_names = list(iris.target_names)   # setosa, versicolor, virginica
    class_map = {i: name.capitalize() for i, name in enumerate(class_names)}

    # Train/test split (for evaluation purposes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nDataset split : {len(X_train)} training  |  {len(X_test)} testing samples")

    # Classify all test samples
    label_to_idx = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
    y_pred = []
    for sl, pl in X_test:
        pred_label = fuzzy_classify(sl, pl)
        y_pred.append(label_to_idx[pred_label])

    y_pred = np.array(y_pred)

    # Metrics
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"\nFuzzy Classifier Accuracy : {accuracy:.2f}%\n")
    print("Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ─────────────────────────────────────────────────────────
    #  7.  VISUALISATIONS
    # ─────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Fuzzy Logic Classifier - Iris Dataset", fontsize=16, fontweight='bold')

    # 7a. Membership functions - Sepal Length
    ax1 = fig.add_subplot(3, 3, 1)
    x_sl = np.linspace(4.0, 8.5, 300)
    ax1.plot(x_sl, trapmf(x_sl, [4.0, 4.0, 5.0, 5.8]),  'b-',  lw=2, label='Short (Setosa)')
    ax1.plot(x_sl, trimf (x_sl, [5.0, 5.8, 6.8]),         'g-',  lw=2, label='Medium (Versicolor)')
    ax1.plot(x_sl, trapmf(x_sl, [6.0, 6.8, 8.5, 8.5]),    'r-',  lw=2, label='Long (Virginica)')
    ax1.set_title('MF - Sepal Length'); ax1.set_xlabel('Sepal Length (cm)')
    ax1.set_ylabel('Membership Degree'); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    # 7b. Membership functions - Petal Length
    ax2 = fig.add_subplot(3, 3, 2)
    x_pl = np.linspace(1.0, 7.0, 300)
    ax2.plot(x_pl, trapmf(x_pl, [1.0, 1.0, 2.0, 2.5]),  'b-',  lw=2, label='Small (Setosa)')
    ax2.plot(x_pl, trimf (x_pl, [2.0, 3.5, 5.0]),         'g-',  lw=2, label='Medium (Versicolor)')
    ax2.plot(x_pl, trapmf(x_pl, [4.5, 5.5, 7.0, 7.0]),    'r-',  lw=2, label='Large (Virginica)')
    ax2.set_title('MF - Petal Length'); ax2.set_xlabel('Petal Length (cm)')
    ax2.set_ylabel('Membership Degree'); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # 7c. Confusion Matrix
    ax3 = fig.add_subplot(3, 3, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')

    # 7d. Decision regions
    ax4 = fig.add_subplot(3, 3, (4, 6))
    sl_grid = np.linspace(4.0, 8.5, 100)
    pl_grid = np.linspace(1.0, 7.0, 100)
    SL, PL  = np.meshgrid(sl_grid, pl_grid)
    Z = np.zeros(SL.shape, dtype=int)
    for i in range(SL.shape[0]):
        for j in range(SL.shape[1]):
            pred = fuzzy_classify(SL[i, j], PL[i, j])
            Z[i, j] = label_to_idx[pred]

    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(['#AED6F1', '#A9DFBF', '#F9E79F'])
    ax4.contourf(SL, PL, Z, alpha=0.5, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5, 2.5])

    scatter_colors = ['blue', 'green', 'red']
    for cls_idx, (col, name) in enumerate(zip(scatter_colors, class_names)):
        mask = y == cls_idx
        ax4.scatter(X[mask, 0], X[mask, 1], c=col, label=name.capitalize(),
                    edgecolors='k', linewidths=0.5, s=60, alpha=0.8)
    ax4.set_title('Fuzzy Decision Regions (Sepal Length vs Petal Length)')
    ax4.set_xlabel('Sepal Length (cm)'); ax4.set_ylabel('Petal Length (cm)')
    ax4.legend(); ax4.grid(alpha=0.3)

    # 7e. Firing strength bar for a sample point
    ax5 = fig.add_subplot(3, 3, 7)
    sample_sl, sample_pl = 5.1, 1.4    # typical Setosa
    sl_m = fuzzify_sepal_length(sample_sl)
    pl_m = fuzzify_petal_length(sample_pl)
    firing = infer(sl_m, pl_m)
    classes  = list(firing.keys())
    strengths = list(firing.values())
    bars = ax5.bar(classes, strengths, color=['blue', 'green', 'red'], edgecolor='black', alpha=0.8)
    ax5.set_title(f'Firing Strengths\n(SL={sample_sl}, PL={sample_pl})')
    ax5.set_ylabel('Firing Strength'); ax5.set_ylim(0, 1.1)
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, strengths):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

    # 7f. Per-class accuracy
    ax6 = fig.add_subplot(3, 3, 8)
    per_class_acc = []
    for cls_idx in range(3):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            per_class_acc.append(np.mean(y_pred[mask] == y_test[mask]) * 100)
        else:
            per_class_acc.append(0)
    ax6.bar(class_names, per_class_acc, color=scatter_colors, edgecolor='black', alpha=0.8)
    ax6.set_title('Per-class Accuracy'); ax6.set_ylabel('Accuracy (%)')
    ax6.set_ylim(0, 110); ax6.grid(axis='y', alpha=0.3)
    for i, v in enumerate(per_class_acc):
        ax6.text(i, v + 1, f'{v:.0f}%', ha='center', fontweight='bold')

    # 7g. Scatter of actual vs predicted
    ax7 = fig.add_subplot(3, 3, 9)
    correct   = y_pred == y_test
    incorrect = ~correct
    ax7.scatter(X_test[correct, 0],   X_test[correct, 1],
                c='green', marker='o', label='Correct', s=60, alpha=0.8, edgecolors='k', lw=0.5)
    ax7.scatter(X_test[incorrect, 0], X_test[incorrect, 1],
                c='red',   marker='x', label='Incorrect', s=80, alpha=0.9, linewidths=1.5)
    ax7.set_title('Correct vs Incorrect Predictions')
    ax7.set_xlabel('Sepal Length (cm)'); ax7.set_ylabel('Petal Length (cm)')
    ax7.legend(); ax7.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fuzzy_classifier_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved -> fuzzy_classifier_results.png")
    plt.show()

    print("\n" + "=" * 60)
    print("  Fuzzy Classification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
