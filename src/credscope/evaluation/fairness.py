"""Fairness Auditing for Credit Risk Models

This module provides fairness metrics and bias detection for ensuring
equitable credit decisions across demographic groups.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class FairnessAuditor:
    """Fairness auditor for evaluating model bias across protected attributes"""

    def __init__(self, protected_attributes: List[str]):
        """Initialize fairness auditor

        Args:
            protected_attributes: List of column names for protected attributes
                                 (e.g., ['CODE_GENDER', 'AGE_GROUP'])
        """
        self.protected_attributes = protected_attributes
        self.metrics = {}

    def demographic_parity(
        self,
        y_pred: np.ndarray,
        protected_attribute: pd.Series
    ) -> Dict[str, float]:
        """Calculate demographic parity (equal acceptance rates)

        Measures whether different groups receive positive outcomes at equal rates.

        Args:
            y_pred: Binary predictions (0 or 1)
            protected_attribute: Protected attribute values (e.g., gender)

        Returns:
            Dictionary with acceptance rates per group
        """
        results = {}

        for group in protected_attribute.unique():
            if pd.isna(group):
                continue
            mask = protected_attribute == group
            acceptance_rate = y_pred[mask].mean()
            results[f"group_{group}"] = acceptance_rate

        # Calculate disparity ratio (min/max)
        if len(results) > 0:
            rates = list(results.values())
            results['disparity_ratio'] = min(rates) / max(rates) if max(rates) > 0 else 0
            results['max_difference'] = max(rates) - min(rates)

        return results

    def equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: pd.Series
    ) -> Dict[str, float]:
        """Calculate equal opportunity (equal true positive rates)

        Measures whether qualified individuals from different groups have
        equal chances of being correctly identified.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values

        Returns:
            Dictionary with TPR per group
        """
        results = {}

        for group in protected_attribute.unique():
            if pd.isna(group):
                continue
            mask = protected_attribute == group

            # True Positive Rate for this group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # TPR = True Positives / (True Positives + False Negatives)
            positives = y_true_group == 1
            if positives.sum() > 0:
                tpr = (y_pred_group[positives] == 1).sum() / positives.sum()
                results[f"group_{group}_tpr"] = tpr

        # Calculate disparity
        if len(results) > 0:
            tprs = list(results.values())
            results['tpr_disparity_ratio'] = min(tprs) / max(tprs) if max(tprs) > 0 else 0
            results['tpr_max_difference'] = max(tprs) - min(tprs)

        return results

    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: pd.Series
    ) -> Dict[str, float]:
        """Calculate equalized odds (equal TPR and FPR)

        Measures whether groups have equal true positive and false positive rates.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values

        Returns:
            Dictionary with TPR and FPR per group
        """
        results = {}

        for group in protected_attribute.unique():
            if pd.isna(group):
                continue
            mask = protected_attribute == group

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # TPR
            positives = y_true_group == 1
            if positives.sum() > 0:
                tpr = (y_pred_group[positives] == 1).sum() / positives.sum()
                results[f"group_{group}_tpr"] = tpr

            # FPR = False Positives / (False Positives + True Negatives)
            negatives = y_true_group == 0
            if negatives.sum() > 0:
                fpr = (y_pred_group[negatives] == 1).sum() / negatives.sum()
                results[f"group_{group}_fpr"] = fpr

        return results

    def predictive_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: pd.Series
    ) -> Dict[str, float]:
        """Calculate predictive parity (equal precision/PPV)

        Measures whether positive predictions are equally accurate across groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values

        Returns:
            Dictionary with precision per group
        """
        results = {}

        for group in protected_attribute.unique():
            if pd.isna(group):
                continue
            mask = protected_attribute == group

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # Precision = True Positives / (True Positives + False Positives)
            predicted_positive = y_pred_group == 1
            if predicted_positive.sum() > 0:
                precision = (y_true_group[predicted_positive] == 1).sum() / predicted_positive.sum()
                results[f"group_{group}_precision"] = precision

        # Calculate disparity
        if len(results) > 0:
            precisions = list(results.values())
            results['precision_disparity_ratio'] = min(precisions) / max(precisions) if max(precisions) > 0 else 0
            results['precision_max_difference'] = max(precisions) - min(precisions)

        return results

    def comprehensive_audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        protected_data: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """Run comprehensive fairness audit across all protected attributes

        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities
            protected_data: DataFrame with protected attribute columns
            threshold: Classification threshold

        Returns:
            DataFrame with all fairness metrics
        """
        logger.info("Running comprehensive fairness audit...")

        all_metrics = []

        for attr in self.protected_attributes:
            if attr not in protected_data.columns:
                logger.warning(f"Protected attribute '{attr}' not found in data")
                continue

            logger.info(f"Analyzing fairness for: {attr}")

            # Demographic parity
            dp_metrics = self.demographic_parity(y_pred, protected_data[attr])

            # Equal opportunity
            eo_metrics = self.equal_opportunity(y_true, y_pred, protected_data[attr])

            # Equalized odds
            eq_metrics = self.equalized_odds(y_true, y_pred, protected_data[attr])

            # Predictive parity
            pp_metrics = self.predictive_parity(y_true, y_pred, protected_data[attr])

            # Combine metrics
            attr_metrics = {
                'protected_attribute': attr,
                **dp_metrics,
                **eo_metrics,
                **eq_metrics,
                **pp_metrics
            }

            all_metrics.append(attr_metrics)

        # Create DataFrame
        metrics_df = pd.DataFrame(all_metrics)

        logger.info("✓ Fairness audit complete")
        return metrics_df

    def visualize_fairness(
        self,
        metrics_df: pd.DataFrame,
        output_path: Optional[str] = None
    ):
        """Create visualization of fairness metrics

        Args:
            metrics_df: DataFrame from comprehensive_audit()
            output_path: Path to save plot
        """
        logger.info("Creating fairness visualization...")

        # Extract disparity ratios
        disparity_cols = [col for col in metrics_df.columns if 'disparity_ratio' in col]

        if len(disparity_cols) == 0:
            logger.warning("No disparity metrics found to visualize")
            return

        fig, axes = plt.subplots(1, len(disparity_cols), figsize=(5*len(disparity_cols), 5))
        if len(disparity_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, disparity_cols):
            values = metrics_df[col].values
            attributes = metrics_df['protected_attribute'].values

            # Create bar plot
            colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in values]
            ax.barh(attributes, values, color=colors)
            ax.axvline(x=0.8, color='green', linestyle='--', label='Fair (≥0.8)')
            ax.axvline(x=0.6, color='orange', linestyle='--', label='Warning (<0.8)')
            ax.set_xlabel('Disparity Ratio')
            ax.set_title(col.replace('_', ' ').title())
            ax.set_xlim(0, 1)
            ax.legend()
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Fairness visualization saved to {output_path}")

        plt.close()

    def generate_fairness_report(
        self,
        metrics_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """Generate human-readable fairness report

        Args:
            metrics_df: DataFrame from comprehensive_audit()
            output_path: Path to save report

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            "FAIRNESS AUDIT REPORT",
            "=" * 80,
            "",
            "This report evaluates model fairness across protected attributes.",
            "Disparity ratios < 0.8 indicate potential bias (per 80% rule).",
            "",
        ]

        for _, row in metrics_df.iterrows():
            attr = row['protected_attribute']
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"Protected Attribute: {attr}")
            report_lines.append(f"{'='*80}")

            # Demographic parity
            if 'disparity_ratio' in row:
                dr = row['disparity_ratio']
                status = "✓ PASS" if dr >= 0.8 else "✗ FAIL"
                report_lines.append(f"\nDemographic Parity: {status}")
                report_lines.append(f"  Disparity Ratio: {dr:.4f}")
                report_lines.append(f"  Max Difference: {row.get('max_difference', 'N/A'):.4f}")

            # Equal opportunity
            if 'tpr_disparity_ratio' in row:
                tpr_dr = row['tpr_disparity_ratio']
                status = "✓ PASS" if tpr_dr >= 0.8 else "✗ FAIL"
                report_lines.append(f"\nEqual Opportunity: {status}")
                report_lines.append(f"  TPR Disparity Ratio: {tpr_dr:.4f}")
                report_lines.append(f"  TPR Max Difference: {row.get('tpr_max_difference', 'N/A'):.4f}")

            # Predictive parity
            if 'precision_disparity_ratio' in row:
                pp_dr = row['precision_disparity_ratio']
                status = "✓ PASS" if pp_dr >= 0.8 else "✗ FAIL"
                report_lines.append(f"\nPredictive Parity: {status}")
                report_lines.append(f"  Precision Disparity Ratio: {pp_dr:.4f}")
                report_lines.append(f"  Precision Max Difference: {row.get('precision_max_difference', 'N/A'):.4f}")

        report_lines.append(f"\n{'='*80}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'='*80}")
        report_lines.append("")
        report_lines.append("• Review any metrics with disparity ratio < 0.8")
        report_lines.append("• Consider resampling or reweighting training data")
        report_lines.append("• Add fairness constraints during model training")
        report_lines.append("• Conduct regular fairness audits on production data")
        report_lines.append("")

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"✓ Fairness report saved to {output_path}")

        return report


def audit_model_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    protected_data: pd.DataFrame,
    protected_attributes: List[str],
    output_dir: str = 'fairness_audit'
) -> Tuple[pd.DataFrame, str]:
    """Run complete fairness audit and generate reports

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        protected_data: DataFrame with protected attributes
        protected_attributes: List of protected attribute column names
        output_dir: Directory to save outputs

    Returns:
        Tuple of (metrics_df, report_text)
    """
    logger.info("Starting comprehensive fairness audit...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create auditor
    auditor = FairnessAuditor(protected_attributes)

    # Run audit
    metrics_df = auditor.comprehensive_audit(
        y_true, y_pred, y_pred_proba, protected_data
    )

    # Save metrics
    metrics_df.to_csv(output_path / 'fairness_metrics.csv', index=False)
    logger.info(f"✓ Metrics saved to {output_path / 'fairness_metrics.csv'}")

    # Create visualization
    auditor.visualize_fairness(
        metrics_df,
        output_path=str(output_path / 'fairness_visual.png')
    )

    # Generate report
    report = auditor.generate_fairness_report(
        metrics_df,
        output_path=str(output_path / 'fairness_report.txt')
    )

    logger.info(f"✓ Fairness audit complete! Results saved to {output_path}")

    return metrics_df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fairness Auditing Module")
    print("=" * 50)
    print("\nUsage:")
    print("""
    from src.credscope.evaluation.fairness import audit_model_fairness

    # Run comprehensive fairness audit
    metrics_df, report = audit_model_fairness(
        y_true=y_val,
        y_pred=predictions,
        y_pred_proba=pred_probabilities,
        protected_data=df[['CODE_GENDER', 'AGE_GROUP']],
        protected_attributes=['CODE_GENDER', 'AGE_GROUP'],
        output_dir='fairness_audit'
    )

    print(report)
    """)
