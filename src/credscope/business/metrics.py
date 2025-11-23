"""Business Metrics and ROI Calculator

Calculates business impact, ROI, and financial metrics for credit decisions.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoanParameters:
    """Parameters for loan calculations"""
    average_loan_amount: float = 500000
    interest_rate: float = 0.12  # 12% annual
    loan_term_months: int = 60
    default_loss_rate: float = 0.70  # 70% loss on default
    processing_cost: float = 500
    annual_operating_cost: float = 1000  # per active loan


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    true_positives: int  # Correctly rejected bad loans
    true_negatives: int  # Correctly approved good loans
    false_positives: int  # Rejected good loans (missed revenue)
    false_negatives: int  # Approved bad loans (losses)


class ROICalculator:
    """Calculate Return on Investment for credit risk model"""

    def __init__(self, loan_params: Optional[LoanParameters] = None):
        """Initialize ROI calculator

        Args:
            loan_params: Loan parameters for calculations
        """
        self.params = loan_params or LoanParameters()

    def calculate_revenue_per_loan(self) -> float:
        """Calculate total revenue from a good loan

        Returns:
            Revenue in currency units
        """
        # Simple interest calculation
        total_interest = (
            self.params.average_loan_amount *
            self.params.interest_rate *
            (self.params.loan_term_months / 12)
        )

        # Net revenue after operating costs
        operating_costs = (
            self.params.annual_operating_cost *
            (self.params.loan_term_months / 12)
        )

        net_revenue = total_interest - operating_costs - self.params.processing_cost

        return net_revenue

    def calculate_loss_per_default(self) -> float:
        """Calculate loss from a defaulted loan

        Returns:
            Loss in currency units
        """
        # Principal loss + processing cost
        loss = (
            self.params.average_loan_amount * self.params.default_loss_rate +
            self.params.processing_cost
        )

        return loss

    def calculate_opportunity_cost(self) -> float:
        """Calculate opportunity cost of rejecting a good loan

        Returns:
            Opportunity cost in currency units
        """
        # Lost revenue + processing cost already spent
        return self.calculate_revenue_per_loan() + self.params.processing_cost

    def calculate_model_roi(
        self,
        performance: ModelPerformance,
        baseline_performance: Optional[ModelPerformance] = None
    ) -> Dict:
        """Calculate comprehensive ROI metrics

        Args:
            performance: Current model performance
            baseline_performance: Baseline model performance for comparison

        Returns:
            Dictionary with ROI metrics
        """
        revenue_per_loan = self.calculate_revenue_per_loan()
        loss_per_default = self.calculate_loss_per_default()
        opportunity_cost = self.calculate_opportunity_cost()

        # Calculate financial impact
        revenue_from_good_loans = performance.true_negatives * revenue_per_loan
        losses_from_bad_loans = performance.false_negatives * loss_per_default
        opportunity_losses = performance.false_positives * opportunity_cost

        # Net profit
        net_profit = revenue_from_good_loans - losses_from_bad_loans - opportunity_losses

        # Total loans processed
        total_loans = (
            performance.true_positives +
            performance.true_negatives +
            performance.false_positives +
            performance.false_negatives
        )

        # Average profit per loan
        avg_profit_per_loan = net_profit / total_loans if total_loans > 0 else 0

        # Calculate baseline comparison if provided
        improvement_metrics = {}
        if baseline_performance:
            baseline_revenue = baseline_performance.true_negatives * revenue_per_loan
            baseline_losses = baseline_performance.false_negatives * loss_per_default
            baseline_opp_losses = baseline_performance.false_positives * opportunity_cost
            baseline_net_profit = baseline_revenue - baseline_losses - baseline_opp_losses

            improvement = net_profit - baseline_net_profit
            improvement_pct = (improvement / abs(baseline_net_profit)) * 100 if baseline_net_profit != 0 else 0

            improvement_metrics = {
                'baseline_net_profit': float(baseline_net_profit),
                'profit_improvement': float(improvement),
                'improvement_percent': float(improvement_pct),
                'additional_loans_approved': performance.true_negatives - baseline_performance.true_negatives,
                'defaults_prevented': baseline_performance.false_negatives - performance.false_negatives
            }

        return {
            'total_loans': total_loans,
            'approved_loans': performance.true_negatives + performance.false_negatives,
            'rejected_loans': performance.true_positives + performance.false_positives,

            # Revenue & Costs
            'total_revenue': float(revenue_from_good_loans),
            'total_losses': float(losses_from_bad_loans),
            'opportunity_costs': float(opportunity_losses),
            'net_profit': float(net_profit),
            'avg_profit_per_loan': float(avg_profit_per_loan),

            # Rates
            'approval_rate': float((performance.true_negatives + performance.false_negatives) / total_loans) if total_loans > 0 else 0,
            'default_rate_on_approved': float(performance.false_negatives / (performance.true_negatives + performance.false_negatives)) if (performance.true_negatives + performance.false_negatives) > 0 else 0,

            # Comparison
            **improvement_metrics
        }

    def calculate_portfolio_metrics(
        self,
        predictions: List[Dict]
    ) -> Dict:
        """Calculate portfolio-level business metrics

        Args:
            predictions: List of prediction dictionaries with 'decision' and 'default_probability'

        Returns:
            Portfolio metrics
        """
        total_predictions = len(predictions)
        if total_predictions == 0:
            return {'error': 'No predictions provided'}

        # Count approvals
        approvals = [p for p in predictions if p.get('decision') == 'APPROVE']
        rejections = [p for p in predictions if p.get('decision') == 'REJECT']

        # Calculate expected values
        total_expected_revenue = 0
        total_expected_loss = 0

        revenue_per_loan = self.calculate_revenue_per_loan()
        loss_per_default = self.calculate_loss_per_default()

        for pred in approvals:
            prob_default = pred.get('default_probability', 0.5)
            prob_repay = 1 - prob_default

            # Expected value = (prob_repay * revenue) - (prob_default * loss)
            expected_value = (prob_repay * revenue_per_loan) - (prob_default * loss_per_default)

            total_expected_revenue += expected_value

        # Expected defaults
        expected_defaults = sum(p.get('default_probability', 0) for p in approvals)

        # Portfolio risk metrics
        avg_default_prob = np.mean([p.get('default_probability', 0) for p in approvals]) if approvals else 0

        risk_levels = {}
        for pred in predictions:
            risk = pred.get('risk_level', 'UNKNOWN')
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

        return {
            'total_applications': total_predictions,
            'approved_count': len(approvals),
            'rejected_count': len(rejections),
            'approval_rate': float(len(approvals) / total_predictions),

            # Financial projections
            'expected_portfolio_value': float(total_expected_revenue),
            'expected_defaults': float(expected_defaults),
            'expected_default_rate': float(avg_default_prob),

            # Risk distribution
            'risk_distribution': risk_levels,

            # Per loan metrics
            'avg_expected_value_per_approval': float(total_expected_revenue / len(approvals)) if approvals else 0
        }


class BusinessMetricsCalculator:
    """Calculate various business metrics"""

    @staticmethod
    def calculate_precision_at_threshold(
        probabilities: np.ndarray,
        actuals: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """Calculate precision at a specific threshold

        Args:
            probabilities: Prediction probabilities
            actuals: Actual labels
            threshold: Decision threshold

        Returns:
            Precision value
        """
        predictions = (probabilities >= threshold).astype(int)
        true_positives = ((predictions == 1) & (actuals == 1)).sum()
        false_positives = ((predictions == 1) & (actuals == 0)).sum()

        if true_positives + false_positives == 0:
            return 0.0

        return true_positives / (true_positives + false_positives)

    @staticmethod
    def calculate_profit_curve(
        probabilities: np.ndarray,
        actuals: np.ndarray,
        revenue_per_good: float = 60000,
        cost_per_bad: float = 350000
    ) -> Dict:
        """Calculate profit curve across thresholds

        Args:
            probabilities: Prediction probabilities
            actuals: Actual labels
            revenue_per_good: Revenue from good loan
            cost_per_bad: Cost of bad loan

        Returns:
            Dictionary with optimal threshold and profit
        """
        thresholds = np.linspace(0, 1, 101)
        profits = []

        for threshold in thresholds:
            predictions = (probabilities < threshold).astype(int)  # Approve if prob < threshold

            # Approvals
            approved = predictions == 0
            good_approvals = (approved & (actuals == 0)).sum()
            bad_approvals = (approved & (actuals == 1)).sum()

            # Calculate profit
            profit = (good_approvals * revenue_per_good) - (bad_approvals * cost_per_bad)
            profits.append(profit)

        profits = np.array(profits)
        optimal_idx = np.argmax(profits)

        return {
            'optimal_threshold': float(thresholds[optimal_idx]),
            'max_profit': float(profits[optimal_idx]),
            'profit_curve': list(zip(thresholds.tolist(), profits.tolist()))
        }

    @staticmethod
    def calculate_conversion_rate(
        total_applicants: int,
        approved: int,
        actually_applied: int
    ) -> Dict:
        """Calculate conversion funnel metrics

        Args:
            total_applicants: Total number of applicants
            approved: Number approved
            actually_applied: Number who actually took the loan

        Returns:
            Conversion metrics
        """
        approval_rate = approved / total_applicants if total_applicants > 0 else 0
        conversion_rate = actually_applied / approved if approved > 0 else 0
        overall_conversion = actually_applied / total_applicants if total_applicants > 0 else 0

        return {
            'total_applicants': total_applicants,
            'approved': approved,
            'actually_applied': actually_applied,
            'approval_rate': float(approval_rate),
            'conversion_rate': float(conversion_rate),
            'overall_conversion_rate': float(overall_conversion)
        }


if __name__ == "__main__":
    # Demo usage
    print("Business Metrics Calculator Demo")
    print("=" * 50)

    # Example model performance
    performance = ModelPerformance(
        true_positives=2000,   # Correctly rejected bad loans
        true_negatives=7000,   # Correctly approved good loans
        false_positives=500,   # Wrongly rejected good loans
        false_negatives=500    # Wrongly approved bad loans
    )

    baseline = ModelPerformance(
        true_positives=1500,
        true_negatives=6500,
        false_positives=1000,
        false_negatives=1000
    )

    # Calculate ROI
    calculator = ROICalculator()
    roi = calculator.calculate_model_roi(performance, baseline)

    print("\nROI Analysis:")
    print(f"  Net Profit: ${roi['net_profit']:,.2f}")
    print(f"  Avg Profit per Loan: ${roi['avg_profit_per_loan']:,.2f}")
    print(f"  Approval Rate: {roi['approval_rate']:.1%}")
    print(f"  Default Rate on Approved: {roi['default_rate_on_approved']:.1%}")

    if 'profit_improvement' in roi:
        print(f"\n  Improvement over Baseline:")
        print(f"    Additional Profit: ${roi['profit_improvement']:,.2f}")
        print(f"    Improvement: {roi['improvement_percent']:.1f}%")
        print(f"    Additional Approvals: {roi['additional_loans_approved']}")
        print(f"    Defaults Prevented: {roi['defaults_prevented']}")
