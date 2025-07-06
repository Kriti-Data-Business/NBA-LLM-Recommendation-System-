"""
A/B Testing Framework for NBA Recommendation System
Handles retail vs non-retail vertical testing and statistical validation
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ABTestingFramework:
    """Comprehensive A/B testing framework for NBA system"""
    
    def __init__(self, significance_level=0.05, power=0.8):
        self.significance_level = significance_level
        self.power = power
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
    
    def calculate_sample_size(self, baseline_conversion_rate, minimum_detectable_effect, 
                            alpha=None, beta=None):
        """Calculate required sample size for A/B test"""
        alpha = alpha or self.significance_level
        beta = beta or (1 - self.power)
        
        # Effect size calculation
        p1 = baseline_conversion_rate
        p2 = baseline_conversion_rate + minimum_detectable_effect
        
        # Pooled standard deviation
        p_pooled = (p1 + p2) / 2
        effect_size = abs(p2 - p1) / np.sqrt(p_pooled * (1 - p_pooled))
        
        # Z-scores for alpha and beta
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(1 - beta)
        
        # Sample size calculation
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        self.logger.info(f"Required sample size per group: {int(np.ceil(n))}")
        return int(np.ceil(n))
    
    def design_experiment(self, customer_segments: List[str], vertical_type: str = "retail"):
        """Design A/B test experiment structure"""
        
        experiment_design = {
            'vertical_type': vertical_type,
            'test_groups': {
                'control': {
                    'description': 'Random assignment baseline',
                    'treatment': 'random_assignment',
                    'allocation': 0.2
                },
                'simple_rules': {
                    'description': 'Simple recommendation rules',
                    'treatment': 'business_rules',
                    'allocation': 0.2
                },
                'ml_ensemble': {
                    'description': 'ML ensemble recommendations',
                    'treatment': 'ensemble_model',
                    'allocation': 0.6
                }
            },
            'customer_segments': customer_segments,
            'primary_metric': 'conversion_rate',
            'secondary_metrics': ['revenue_per_customer', 'engagement_rate', 'retention_rate'],
            'start_date': datetime.now(),
            'duration_days': 28
        }
        
        # Adjust for vertical type
        if vertical_type == "non_retail":
            experiment_design['primary_metric'] = 'engagement_rate'
            experiment_design['secondary_metrics'] = ['conversion_rate', 'time_to_conversion', 'customer_satisfaction']
        
        return experiment_design
    
    def assign_customers_to_groups(self, customers_df: pd.DataFrame, 
                                 experiment_design: Dict) -> pd.DataFrame:
        """Assign customers to test groups with stratification"""
        
        # Stratified sampling by customer segment
        result_df = customers_df.copy()
        result_df['test_group'] = None
        
        for segment in experiment_design['customer_segments']:
            segment_customers = result_df[result_df['customer_segment'] == segment]
            
            if len(segment_customers) == 0:
                continue
            
            # Calculate group sizes
            group_sizes = {}
            for group_name, group_info in experiment_design['test_groups'].items():
                group_sizes[group_name] = int(len(segment_customers) * group_info['allocation'])
            
            # Assign customers to groups
            shuffled_customers = segment_customers.sample(frac=1, random_state=42)
            
            start_idx = 0
            for group_name, size in group_sizes.items():
                end_idx = start_idx + size
                group_customers = shuffled_customers.iloc[start_idx:end_idx]
                result_df.loc[group_customers.index, 'test_group'] = group_name
                start_idx = end_idx
        
        # Remove customers not assigned to any group
        result_df = result_df.dropna(subset=['test_group'])
        
        self.logger.info(f"Assigned {len(result_df)} customers to test groups")
        return result_df
    
    def run_retail_vertical_test(self, test_data: pd.DataFrame, 
                               recommendations: Dict) -> Dict:
        """Run A/B test for retail vertical"""
        
        results = {}
        
        # Primary metric: Conversion rate
        for group in test_data['test_group'].unique():
            group_data = test_data[test_data['test_group'] == group]
            
            # Get recommendations for this group
            group_recommendations = recommendations.get(group, [])
            
            # Calculate metrics
            conversion_rate = self._calculate_conversion_rate(group_data, group_recommendations)
            revenue_per_customer = self._calculate_revenue_per_customer(group_data)
            engagement_rate = self._calculate_engagement_rate(group_data)
            retention_rate = self._calculate_retention_rate(group_data)
            
            results[group] = {
                'sample_size': len(group_data),
                'conversion_rate': conversion_rate,
                'revenue_per_customer': revenue_per_customer,
                'engagement_rate': engagement_rate,
                'retention_rate': retention_rate
            }
        
        return results
    
    def run_non_retail_vertical_test(self, test_data: pd.DataFrame, 
                                   recommendations: Dict) -> Dict:
        """Run A/B test for non-retail vertical"""
        
        results = {}
        
        # Primary metric: Engagement rate
        for group in test_data['test_group'].unique():
            group_data = test_data[test_data['test_group'] == group]
            
            # Get recommendations for this group
            group_recommendations = recommendations.get(group, [])
            
            # Calculate metrics
            engagement_rate = self._calculate_engagement_rate(group_data)
            conversion_rate = self._calculate_conversion_rate(group_data, group_recommendations)
            time_to_conversion = self._calculate_time_to_conversion(group_data)
            customer_satisfaction = self._calculate_customer_satisfaction(group_data)
            
            results[group] = {
                'sample_size': len(group_data),
                'engagement_rate': engagement_rate,
                'conversion_rate': conversion_rate,
                'time_to_conversion': time_to_conversion,
                'customer_satisfaction': customer_satisfaction
            }
        
        return results
    
    def _calculate_conversion_rate(self, data: pd.DataFrame, recommendations: List) -> float:
        """Calculate conversion rate for a group"""
        if len(data) == 0:
            return 0.0
        
        # Simulate conversion based on recommendations quality
        # In real implementation, this would be actual conversion data
        conversions = data['converted'].sum() if 'converted' in data.columns else 0
        return conversions / len(data)
    
    def _calculate_revenue_per_customer(self, data: pd.DataFrame) -> float:
        """Calculate revenue per customer"""
        if len(data) == 0:
            return 0.0
        
        total_revenue = data['revenue'].sum() if 'revenue' in data.columns else 0
        return total_revenue / len(data)
    
    def _calculate_engagement_rate(self, data: pd.DataFrame) -> float:
        """Calculate engagement rate"""
        if len(data) == 0:
            return 0.0
        
        engaged = data['engaged'].sum() if 'engaged' in data.columns else 0
        return engaged / len(data)
    
    def _calculate_retention_rate(self, data: pd.DataFrame) -> float:
        """Calculate retention rate"""
        if len(data) == 0:
            return 0.0
        
        retained = data['retained'].sum() if 'retained' in data.columns else 0
        return retained / len(data)
    
    def _calculate_time_to_conversion(self, data: pd.DataFrame) -> float:
        """Calculate average time to conversion"""
        if len(data) == 0:
            return 0.0
        
        if 'time_to_conversion' in data.columns:
            return data['time_to_conversion'].mean()
        return 0.0
    
    def _calculate_customer_satisfaction(self, data: pd.DataFrame) -> float:
        """Calculate customer satisfaction score"""
        if len(data) == 0:
            return 0.0
        
        if 'satisfaction_score' in data.columns:
            return data['satisfaction_score'].mean()
        return 0.0
    
    def perform_statistical_significance_test(self, results: Dict, 
                                            primary_metric: str = 'conversion_rate') -> Dict:
        """Perform statistical significance testing"""
        
        significance_results = {}
        
        # Get control group results
        control_results = results.get('control', {})
        control_metric = control_results.get(primary_metric, 0)
        control_sample_size = control_results.get('sample_size', 0)
        
        # Test each treatment group against control
        for group_name, group_results in results.items():
            if group_name == 'control':
                continue
            
            treatment_metric = group_results.get(primary_metric, 0)
            treatment_sample_size = group_results.get('sample_size', 0)
            
            # Perform appropriate statistical test
            if primary_metric in ['conversion_rate', 'engagement_rate', 'retention_rate']:
                # Proportion test
                p_value, confidence_interval = self._proportion_test(
                    control_metric, control_sample_size,
                    treatment_metric, treatment_sample_size
                )
            else:
                # T-test for continuous metrics
                p_value, confidence_interval = self._t_test(
                    control_metric, control_sample_size,
                    treatment_metric, treatment_sample_size
                )
            
            # Calculate lift
            lift = ((treatment_metric - control_metric) / control_metric * 100) if control_metric > 0 else 0
            
            significance_results[group_name] = {
                'control_metric': control_metric,
                'treatment_metric': treatment_metric,
                'lift_percentage': lift,
                'p_value': p_value,
                'is_significant': p_value < self.significance_level,
                'confidence_interval': confidence_interval,
                'effect_size': abs(treatment_metric - control_metric)
            }
        
        return significance_results
    
    def _proportion_test(self, p1: float, n1: int, p2: float, n2: int) -> Tuple[float, Tuple[float, float]]:
        """Two-proportion z-test"""
        # Convert proportions to counts
        x1 = int(p1 * n1)
        x2 = int(p2 * n2)
        
        # Pooled proportion
        p_pooled = (x1 + x2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Z-score
        z = (p2 - p1) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Confidence interval
        se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        margin_of_error = stats.norm.ppf(1 - self.significance_level/2) * se_diff
        ci_lower = (p2 - p1) - margin_of_error
        ci_upper = (p2 - p1) + margin_of_error
        
        return p_value, (ci_lower, ci_upper)
    
    def _t_test(self, mean1: float, n1: int, mean2: float, n2: int) -> Tuple[float, Tuple[float, float]]:
        """Two-sample t-test (assuming equal variance)"""
        # Simulate standard deviations (in real implementation, use actual data)
        std1 = mean1 * 0.2  # Assume 20% CV
        std2 = mean2 * 0.2
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Standard error
        se = pooled_std * np.sqrt(1/n1 + 1/n2)
        
        # T-score
        t = (mean2 - mean1) / se if se > 0 else 0
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t), df))
        
        # Confidence interval
        margin_of_error = stats.t.ppf(1 - self.significance_level/2, df) * se
        ci_lower = (mean2 - mean1) - margin_of_error
        ci_upper = (mean2 - mean1) + margin_of_error
        
        return p_value, (ci_lower, ci_upper)
    
    def run_backtesting(self, historical_data: pd.DataFrame, 
                       model_predictions: Dict, time_window: int = 30) -> Dict:
        """Run backtesting on historical data"""
        
        backtesting_results = {}
        
        # Split historical data into time windows
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.sort_values('date')
        
        # Create time windows
        start_date = historical_data['date'].min()
        end_date = historical_data['date'].max()
        current_date = start_date
        
        window_results = []
        
        while current_date < end_date:
            window_end = current_date + timedelta(days=time_window)
            window_data = historical_data[
                (historical_data['date'] >= current_date) & 
                (historical_data['date'] < window_end)
            ]
            
            if len(window_data) == 0:
                current_date = window_end
                continue
            
            # Evaluate model performance for this window
            window_metrics = self._evaluate_window_performance(window_data, model_predictions)
            window_metrics['window_start'] = current_date
            window_metrics['window_end'] = window_end
            
            window_results.append(window_metrics)
            current_date = window_end
        
        backtesting_results['window_results'] = window_results
        backtesting_results['overall_performance'] = self._aggregate_window_results(window_results)
        
        return backtesting_results
    
    def _evaluate_window_performance(self, window_data: pd.DataFrame, 
                                   model_predictions: Dict) -> Dict:
        """Evaluate model performance for a time window"""
        
        metrics = {}
        
        # Calculate actual performance metrics
        actual_conversion_rate = window_data['converted'].mean() if 'converted' in window_data.columns else 0
        actual_revenue = window_data['revenue'].sum() if 'revenue' in window_data.columns else 0
        
        # Compare with model predictions (simulated)
        predicted_conversions = len(window_data) * 0.12  # Simulated prediction
        predicted_revenue = predicted_conversions * 50  # Simulated revenue per conversion
        
        metrics['actual_conversion_rate'] = actual_conversion_rate
        metrics['predicted_conversion_rate'] = predicted_conversions / len(window_data)
        metrics['actual_revenue'] = actual_revenue
        metrics['predicted_revenue'] = predicted_revenue
        metrics['prediction_accuracy'] = 1 - abs(actual_conversion_rate - (predicted_conversions / len(window_data)))
        
        return metrics
    
    def _aggregate_window_results(self, window_results: List[Dict]) -> Dict:
        """Aggregate results across all windows"""
        
        if not window_results:
            return {}
        
        # Calculate overall metrics
        overall_metrics = {
            'avg_conversion_rate': np.mean([w['actual_conversion_rate'] for w in window_results]),
            'avg_prediction_accuracy': np.mean([w['prediction_accuracy'] for w in window_results]),
            'total_revenue': sum([w['actual_revenue'] for w in window_results]),
            'num_windows': len(window_results)
        }
        
        return overall_metrics
    
    def generate_test_report(self, experiment_design: Dict, 
                           test_results: Dict, 
                           significance_results: Dict) -> str:
        """Generate comprehensive test report"""
        
        report = f"""
# A/B Test Results Report

## Experiment Overview
- **Vertical Type**: {experiment_design['vertical_type']}
- **Primary Metric**: {experiment_design['primary_metric']}
- **Test Duration**: {experiment_design['duration_days']} days
- **Start Date**: {experiment_design['start_date'].strftime('%Y-%m-%d')}

## Sample Sizes
"""
        
        for group_name, group_results in test_results.items():
            report += f"- **{group_name}**: {group_results['sample_size']} customers\n"
        
        report += "\n## Results Summary\n"
        
        for group_name, group_results in test_results.items():
            report += f"\n### {group_name.title()} Group\n"
            
            for metric, value in group_results.items():
                if metric != 'sample_size':
                    if isinstance(value, float):
                        report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
                    else:
                        report += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
        
        report += "\n## Statistical Significance\n"
        
        for group_name, sig_results in significance_results.items():
            report += f"\n### {group_name.title()} vs Control\n"
            report += f"- **Lift**: {sig_results['lift_percentage']:.2f}%\n"
            report += f"- **P-value**: {sig_results['p_value']:.4f}\n"
            report += f"- **Statistically Significant**: {'Yes' if sig_results['is_significant'] else 'No'}\n"
            report += f"- **Confidence Interval**: [{sig_results['confidence_interval'][0]:.4f}, {sig_results['confidence_interval'][1]:.4f}]\n"
        
        report += "\n## Recommendations\n"
        
        # Find best performing group
        best_group = max(significance_results.keys(), 
                        key=lambda x: significance_results[x]['lift_percentage'])
        
        if significance_results[best_group]['is_significant']:
            report += f"- **Winner**: {best_group.title()} group with {significance_results[best_group]['lift_percentage']:.2f}% lift\n"
            report += f"- **Recommendation**: Deploy {best_group} treatment to all customers\n"
        else:
            report += "- **Result**: No statistically significant difference found\n"
            report += "- **Recommendation**: Continue testing with larger sample size or longer duration\n"
        
        return report
    
    def visualize_results(self, test_results: Dict, significance_results: Dict, 
                         primary_metric: str = 'conversion_rate'):
        """Create visualizations for test results"""
        
        # Extract data for plotting
        groups = list(test_results.keys())
        metrics = [test_results[group][primary_metric] for group in groups]
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Primary metric comparison
        bars = ax1.bar(groups, metrics, color=['red' if group == 'control' else 'blue' for group in groups])
        ax1.set_title(f'{primary_metric.replace("_", " ").title()} by Group')
        ax1.set_ylabel(primary_metric.replace("_", " ").title())
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Lift comparison
        lifts = [significance_results[group]['lift_percentage'] 
                for group in groups if group in significance_results]
        lift_groups = [group for group in groups if group in significance_results]
        
        colors = ['green' if significance_results[group]['is_significant'] else 'orange' 
                 for group in lift_groups]
        
        bars2 = ax2.bar(lift_groups, lifts, color=colors)
        ax2.set_title('Lift % vs Control')
        ax2.set_ylabel('Lift %')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars2, lifts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ab_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Example usage
if __name__ == "__main__":
    # Initialize A/B testing framework
    ab_tester = ABTestingFramework()
    
    # Create sample data
    np.random.seed(42)
    customers = pd.DataFrame({
        'customer_id': range(1000),
        'customer_segment': np.random.choice(['High', 'Medium', 'Low'], 1000),
        'converted': np.random.choice([0, 1], 1000, p=[0.88, 0.12]),
        'revenue': np.random.exponential(50, 1000),
        'engaged': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'retained': np.random.choice([0, 1], 1000, p=[0.75, 0.25]),
        'date': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
    
    # Design experiment
    experiment_design = ab_tester.design_experiment(['High', 'Medium', 'Low'], 'retail')
    
    # Assign customers to groups
    test_customers = ab_tester.assign_customers_to_groups(customers, experiment_design)
    
    # Run retail vertical test
    recommendations = {
        'control': [],
        'simple_rules': [],
        'ml_ensemble': []
    }
    
    test_results = ab_tester.run_retail_vertical_test(test_customers, recommendations)
    
    # Perform statistical significance testing
    significance_results = ab_tester.perform_statistical_significance_test(test_results)
    
    # Generate report
    report = ab_tester.generate_test_report(experiment_design, test_results, significance_results)
    print(report)
    
    # Visualize results
    ab_tester.visualize_results(test_results, significance_results)
