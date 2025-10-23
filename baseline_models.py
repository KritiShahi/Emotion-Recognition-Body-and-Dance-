

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EmotionDataLoader:
    """Load and preprocess emotion recognition dataset"""

    def __init__(self, features_csv):
        self.features_csv = features_csv
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def load_data(self):
        """Load features from CSV"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)

        self.df = pd.read_csv(self.features_csv)
        print(f"\n✓ Loaded {len(self.df)} samples")
        print(f"✓ Total columns: {len(self.df.columns)}")

        return self

    def prepare_features(self):
        """Prepare feature matrix and labels"""
        print("\n" + "="*60)
        print("PREPROCESSING")
        print("="*60)

        # Identify metadata columns to exclude
        metadata_cols = ['filename', 'actor_ID', 'emotion', 'gender',
                        'scenario_ID', 'version', 'num_frames', 'duration']

        # Get feature columns
        feature_cols = [col for col in self.df.columns if col not in metadata_cols]
        self.feature_names = feature_cols

        print(f"\n✓ Feature columns: {len(feature_cols)}")
        print(f"✓ Target column: emotion")

        # Extract features and labels
        X = self.df[feature_cols].values
        y = self.df['emotion'].values

        # Handle missing values
        print(f"\n→ Checking for missing values...")
        missing_count = np.isnan(X).sum()
        if missing_count > 0:
            print(f"  Found {missing_count} missing values, imputing with median...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        else:
            print(f"  No missing values found")

        # Encode labels
        print(f"\n→ Encoding emotion labels...")
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"\n✓ Emotion classes:")
        for i, emotion in enumerate(self.label_encoder.classes_):
            count = np.sum(y == emotion)
            print(f"  {i}: {emotion} ({count} samples)")

        self.X = X
        self.y = y_encoded

        return self

    def normalize_features(self, X_train, X_test):
        """Normalize features using training set statistics"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def get_emotion_name(self, encoded_label):
        """Convert encoded label back to emotion name"""
        return self.label_encoder.inverse_transform([encoded_label])[0]

class EmotionClassifierTrainer:
    """Train and evaluate multiple classifiers"""

    def __init__(self, X, y, feature_names, label_encoder):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        print("\n" + "="*60)
        print("DATA SPLIT")
        print("="*60)
        print(f"\n✓ Training samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")
        print(f"✓ Test size: {test_size*100:.0f}%")

        # Normalize features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        return self

    def train_random_forest(self, n_estimators=100, max_depth=None):
        """Train Random Forest classifier"""
        print("\n" + "="*60)
        print("TRAINING: RANDOM FOREST")
        print("="*60)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        print(f"\n→ Training with {n_estimators} trees...")
        rf.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = rf.predict(self.X_test)

        # Store results
        self.models['Random Forest'] = rf
        self.results['Random Forest'] = {
            'model': rf,
            'y_pred': y_pred,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }

        print(f"✓ Training completed!")
        print(f"  Accuracy: {self.results['Random Forest']['accuracy']:.4f}")

        return rf

    def train_svm(self, kernel='rbf', C=1.0):
        """Train SVM classifier"""
        print("\n" + "="*60)
        print("TRAINING: SUPPORT VECTOR MACHINE")
        print("="*60)

        svm = SVC(
            kernel=kernel,
            C=C,
            random_state=42,
            probability=True
        )

        print(f"\n→ Training with {kernel} kernel...")
        svm.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = svm.predict(self.X_test)

        # Store results
        self.models['SVM'] = svm
        self.results['SVM'] = {
            'model': svm,
            'y_pred': y_pred,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }

        print(f"✓ Training completed!")
        print(f"  Accuracy: {self.results['SVM']['accuracy']:.4f}")

        return svm

    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1):
        """Train Gradient Boosting classifier"""
        print("\n" + "="*60)
        print("TRAINING: GRADIENT BOOSTING")
        print("="*60)

        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )

        print(f"\n→ Training with {n_estimators} estimators...")
        gb.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = gb.predict(self.X_test)

        # Store results
        self.models['Gradient Boosting'] = gb
        self.results['Gradient Boosting'] = {
            'model': gb,
            'y_pred': y_pred,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }

        print(f"✓ Training completed!")
        print(f"  Accuracy: {self.results['Gradient Boosting']['accuracy']:.4f}")

        return gb

    def train_all_models(self):
        """Train all classifiers"""
        self.train_random_forest(n_estimators=100)
        self.train_svm(kernel='rbf', C=1.0)
        self.train_gradient_boosting(n_estimators=100)
        return self


class ModelEvaluator:
    """Evaluate and visualize model performance"""

    def __init__(self, trainer):
        self.trainer = trainer
        self.results = trainer.results
        self.label_encoder = trainer.label_encoder

    def print_comparison(self):
        """Print comparison of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1'] for r in self.results.values()]
        })

        print("\n" + comparison_df.to_string(index=False))

        # Find best model
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'F1-Score']

        print(f"\n✓ Best Model: {best_model} (F1-Score: {best_f1:.4f})")

        return comparison_df

    def plot_metric_comparison(self, save_path='metric_comparison.png'):
        """Plot comparison of metrics across models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]

            values = [self.results[model][metric] for model in models]
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(models, rotation=15, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved metric comparison to {save_path}")
        plt.close()

    def plot_per_class_metrics(self, save_path='per_class_metrics.png'):
        """Plot precision, recall, F1 for each class across all models"""
        emotion_classes = self.label_encoder.classes_
        n_models = len(self.results)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')

        metric_names = ['Precision', 'Recall', 'F1-Score']

        for metric_idx, (ax, metric_name) in enumerate(zip(axes, metric_names)):
            x = np.arange(len(emotion_classes))
            width = 0.25

            for model_idx, (model_name, result) in enumerate(self.results.items()):
                y_true = self.trainer.y_test
                y_pred = result['y_pred']

                # Calculate per-class metrics
                if metric_name == 'Precision':
                    scores = precision_score(y_true, y_pred, average=None, zero_division=0)
                elif metric_name == 'Recall':
                    scores = recall_score(y_true, y_pred, average=None, zero_division=0)
                else:  # F1-Score
                    scores = f1_score(y_true, y_pred, average=None, zero_division=0)

                offset = width * (model_idx - 1)
                ax.bar(x + offset, scores, width, label=model_name, alpha=0.8)

            ax.set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric_name} by Emotion Class', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(emotion_classes, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class metrics to {save_path}")
        plt.close()

    def plot_confusion_matrices(self, save_path='confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        if n_models == 1:
            axes = [axes]

        emotion_classes = self.label_encoder.classes_

        for ax, (model_name, result) in zip(axes, self.results.items()):
            cm = confusion_matrix(self.trainer.y_test, result['y_pred'])

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=emotion_classes, yticklabels=emotion_classes,
                       ax=ax, cbar_kws={'label': 'Proportion'})

            ax.set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')

        plt.suptitle('Confusion Matrices (Normalized)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrices to {save_path}")
        plt.close()

    def plot_classification_reports(self):
        """Print detailed classification reports"""
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*60)

        emotion_classes = self.label_encoder.classes_

        for model_name, result in self.results.items():
            print(f"\n{'='*60}")
            print(f"{model_name}")
            print(f"{'='*60}")

            y_true = self.trainer.y_test
            y_pred = result['y_pred']

            report = classification_report(
                y_true, y_pred,
                target_names=emotion_classes,
                digits=3
            )
            print(report)

    def plot_feature_importance(self, save_path='feature_importance.png', top_n=20):
        """Plot feature importance for tree-based models"""
        tree_models = ['Random Forest', 'Gradient Boosting']
        available_models = [m for m in tree_models if m in self.results]

        if not available_models:
            print("\n⚠ No tree-based models available for feature importance")
            return

        fig, axes = plt.subplots(1, len(available_models), figsize=(10*len(available_models), 8))

        if len(available_models) == 1:
            axes = [axes]

        for ax, model_name in zip(axes, available_models):
            model = self.results[model_name]['model']
            importances = model.feature_importances_

            # Get top N features
            indices = np.argsort(importances)[-top_n:]
            feature_names = np.array(self.trainer.feature_names)[indices]

            ax.barh(range(top_n), importances[indices], color='steelblue', alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(feature_names, fontsize=9)
            ax.set_xlabel('Feature Importance', fontweight='bold')
            ax.set_title(f'{model_name}\nTop {top_n} Features', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved feature importance to {save_path}")
        plt.close()

    def generate_all_plots(self):
        """Generate all evaluation plots"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        self.plot_metric_comparison()
        self.plot_per_class_metrics()
        self.plot_confusion_matrices()
        self.plot_feature_importance()

        print("\n✓ All visualizations generated!")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""

    print("\n" + "="*60)
    print("EMOTION RECOGNITION ML PIPELINE")
    print("="*60)

    # Configuration
    FEATURES_CSV = 'features_dataset.csv'
    TEST_SIZE = 0.2

    # Step 1: Load data
    loader = EmotionDataLoader(FEATURES_CSV)
    loader.load_data().prepare_features()

    # Step 2: Train models
    trainer = EmotionClassifierTrainer(
        X=loader.X,
        y=loader.y,
        feature_names=loader.feature_names,
        label_encoder=loader.label_encoder
    )

    trainer.split_data(test_size=TEST_SIZE).train_all_models()

    # Step 3: Evaluate and visualize
    evaluator = ModelEvaluator(trainer)
    comparison_df = evaluator.print_comparison()
    evaluator.plot_classification_reports()
    evaluator.generate_all_plots()

    # Save comparison to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print(f"\n✓ Model comparison saved to model_comparison.csv")

    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  • model_comparison.csv")
    print("  • metric_comparison.png")
    print("  • per_class_metrics.png")
    print("  • confusion_matrices.png")
    print("  • feature_importance.png")

    return trainer, evaluator

if __name__ == "__main__":
    trainer, evaluator = main()
