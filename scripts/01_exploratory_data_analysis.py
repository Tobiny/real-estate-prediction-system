"""
Real Estate Data - Exploratory Data Analysis

This script performs comprehensive exploratory data analysis on the Zillow datasets
to understand data quality, patterns, and relationships for ML model development.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RealEstateEDA:
    """
    Comprehensive Exploratory Data Analysis for Real Estate Data.

    This class provides methods to analyze Zillow datasets, identify patterns,
    and generate insights for feature engineering and model development.
    """

    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the EDA class.

        Args:
            data_dir: Directory containing processed data files
        """
        # Find the project root directory (contains the data folder)
        current_dir = Path.cwd()
        project_root = current_dir

        # Go up until we find the data directory or reach the project root
        while not (project_root / "data").exists() and project_root.parent != project_root:
            project_root = project_root.parent

        # If we're running from any subdirectory, find project root
        if "src" in str(current_dir) or "notebooks" in str(current_dir) or "scripts" in str(current_dir):
            # Go up until we find data directory
            while not (project_root / "data").exists() and project_root.parent != project_root:
                project_root = project_root.parent

        self.data_dir = project_root / data_dir
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.logger = self._setup_logging()

        # Set up plotting parameters
        self.figsize = (15, 8)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        self.logger.info(f"Looking for processed data in: {self.data_dir.absolute()}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all processed datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        self.logger.info("Loading processed datasets...")

        # Define expected files
        file_mapping = {
            'zhvi': 'zhvi_all_homes_processed.csv',
            'zori': 'zori_all_homes_processed.csv',
            'inventory': 'inventory_metro_processed.csv',
            'sales': 'sales_count_metro_processed.csv'
        }

        for name, filename in file_mapping.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                self.logger.info(f"Loading {name} from {filename}")
                df = pd.read_csv(file_path)

                # Debug: Show first few rows to understand data structure
                self.logger.info(f"Preview of {name}: columns = {list(df.columns)}")
                self.logger.info(f"First few Date values: {df['Date'].head().tolist() if 'Date' in df.columns else 'No Date column'}")

                # Clean and convert Date column
                if 'Date' in df.columns:
                    # Remove rows where Date is not actually a date
                    valid_dates = pd.to_datetime(df['Date'], errors='coerce')
                    invalid_mask = valid_dates.isna()

                    if invalid_mask.sum() > 0:
                        self.logger.warning(f"Found {invalid_mask.sum():,} invalid dates in {name}, removing them")
                        # Keep only rows with valid dates
                        df = df[~invalid_mask].copy()
                        df['Date'] = pd.to_datetime(df['Date'])
                    else:
                        df['Date'] = valid_dates

                # Convert Value to numeric, handling any non-numeric values
                if 'Value' in df.columns:
                    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

                    # Remove rows with invalid values
                    invalid_values = df['Value'].isna()
                    if invalid_values.sum() > 0:
                        self.logger.warning(f"Found {invalid_values.sum():,} invalid values in {name}, removing them")
                        df = df[~invalid_values].copy()

                # Final cleanup - ensure we have the expected columns
                expected_cols = ['RegionName', 'Date', 'Value', 'Dataset', 'GeographyLevel']
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"Missing expected columns in {name}: {missing_cols}")

                self.datasets[name] = df
                self.logger.info(f"Successfully loaded {name}: {len(df):,} rows after cleaning")
            else:
                self.logger.warning(f"File not found: {file_path}")

        return self.datasets

    def generate_data_overview(self) -> pd.DataFrame:
        """
        Generate comprehensive data overview.

        Returns:
            DataFrame with summary statistics for all datasets
        """
        overview_data = []

        for name, df in self.datasets.items():
            if df.empty:
                continue

            # Basic statistics
            total_rows = len(df)
            unique_regions = df['RegionName'].nunique() if 'RegionName' in df.columns else 'N/A'
            date_range = f"{df['Date'].min():%Y-%m} to {df['Date'].max():%Y-%m}" if 'Date' in df.columns else 'N/A'

            # Value statistics
            if 'Value' in df.columns:
                value_stats = df['Value'].describe()
                missing_values = df['Value'].isna().sum()
                missing_pct = (missing_values / total_rows) * 100
            else:
                value_stats = pd.Series([np.nan] * 8)
                missing_values = 0
                missing_pct = 0

            overview_data.append({
                'Dataset': name.upper(),
                'Total_Rows': f"{total_rows:,}",
                'Unique_Regions': f"{unique_regions:,}" if unique_regions != 'N/A' else 'N/A',
                'Date_Range': date_range,
                'Missing_Values': f"{missing_values:,} ({missing_pct:.1f}%)",
                'Min_Value': f"${value_stats['min']:,.0f}" if not pd.isna(value_stats['min']) else 'N/A',
                'Max_Value': f"${value_stats['max']:,.0f}" if not pd.isna(value_stats['max']) else 'N/A',
                'Median_Value': f"${value_stats['50%']:,.0f}" if not pd.isna(value_stats['50%']) else 'N/A'
            })

        overview_df = pd.DataFrame(overview_data)

        print("📊 REAL ESTATE DATA OVERVIEW")
        print("=" * 80)
        print(overview_df.to_string(index=False))
        print()

        return overview_df

    def analyze_time_trends(self) -> None:
        """Analyze temporal patterns in the data."""
        print("📈 TIME SERIES ANALYSIS")
        print("=" * 50)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Home Values (ZHVI)', 'Rent Values (ZORI)',
                           'Inventory Levels', 'Sales Activity'],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )

        # ZHVI trends
        if 'zhvi' in self.datasets:
            zhvi_monthly = self.datasets['zhvi'].groupby('Date')['Value'].agg(['mean', 'median', 'count']).reset_index()

            fig.add_trace(
                go.Scatter(x=zhvi_monthly['Date'], y=zhvi_monthly['median'],
                          name='Median Home Value', line=dict(color='#1f77b4')),
                row=1, col=1
            )

            print(f"📍 ZHVI Analysis:")
            print(f"   • Data points: {len(zhvi_monthly):,}")
            print(f"   • Current median: ${zhvi_monthly['median'].iloc[-1]:,.0f}")
            print(f"   • 5-year change: {((zhvi_monthly['median'].iloc[-1] / zhvi_monthly['median'].iloc[-60] - 1) * 100):.1f}%")

        # ZORI trends
        if 'zori' in self.datasets:
            zori_monthly = self.datasets['zori'].groupby('Date')['Value'].agg(['mean', 'median', 'count']).reset_index()

            fig.add_trace(
                go.Scatter(x=zori_monthly['Date'], y=zori_monthly['median'],
                          name='Median Rent', line=dict(color='#ff7f0e')),
                row=1, col=2
            )

            print(f"📍 ZORI Analysis:")
            print(f"   • Data points: {len(zori_monthly):,}")
            print(f"   • Current median: ${zori_monthly['median'].iloc[-1]:,.0f}")
            print(f"   • 5-year change: {((zori_monthly['median'].iloc[-1] / zori_monthly['median'].iloc[-60] - 1) * 100):.1f}%")

        # Inventory trends
        if 'inventory' in self.datasets:
            inv_monthly = self.datasets['inventory'].groupby('Date')['Value'].agg(['mean', 'median', 'sum']).reset_index()

            fig.add_trace(
                go.Scatter(x=inv_monthly['Date'], y=inv_monthly['sum'],
                          name='Total Inventory', line=dict(color='#2ca02c')),
                row=2, col=1
            )

            print(f"📍 Inventory Analysis:")
            print(f"   • Current total: {inv_monthly['sum'].iloc[-1]:,.0f} homes")
            print(f"   • Peak inventory: {inv_monthly['sum'].max():,.0f} homes")

        # Sales trends
        if 'sales' in self.datasets:
            sales_monthly = self.datasets['sales'].groupby('Date')['Value'].agg(['mean', 'median', 'sum']).reset_index()

            fig.add_trace(
                go.Scatter(x=sales_monthly['Date'], y=sales_monthly['sum'],
                          name='Total Sales', line=dict(color='#d62728')),
                row=2, col=2
            )

            print(f"📍 Sales Analysis:")
            print(f"   • Current monthly: {sales_monthly['sum'].iloc[-1]:,.0f} sales")
            print(f"   • Peak sales: {sales_monthly['sum'].max():,.0f} sales")

        fig.update_layout(height=800, title_text="Real Estate Market Trends Over Time")
        fig.show()
        print()

    def analyze_geographic_patterns(self) -> None:
        """Analyze geographic distribution of data."""
        print("🗺️ GEOGRAPHIC ANALYSIS")
        print("=" * 40)

        # Top states by data coverage
        if 'zhvi' in self.datasets:
            state_coverage = self.datasets['zhvi'].groupby('StateName').agg({
                'RegionName': 'nunique',
                'Value': ['count', 'median']
            }).round(0)

            state_coverage.columns = ['Unique_ZIPs', 'Total_Observations', 'Median_Value']
            state_coverage = state_coverage.sort_values('Unique_ZIPs', ascending=False).head(10)

            print("📍 Top 10 States by ZIP Code Coverage:")
            print(state_coverage.to_string())
            print()

            # Visualize top states
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

            # ZIP count by state
            state_coverage['Unique_ZIPs'].plot(kind='bar', ax=ax1, color=self.colors[0])
            ax1.set_title('ZIP Codes by State')
            ax1.set_ylabel('Number of ZIP Codes')
            ax1.tick_params(axis='x', rotation=45)

            # Median value by state
            state_coverage['Median_Value'].plot(kind='bar', ax=ax2, color=self.colors[1])
            ax2.set_title('Median Home Value by State')
            ax2.set_ylabel('Median Home Value ($)')
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

        # Metro analysis
        if 'inventory' in self.datasets:
            metro_analysis = self.datasets['inventory'].groupby('RegionName').agg({
                'Value': ['mean', 'max', 'std']
            }).round(0)

            metro_analysis.columns = ['Avg_Inventory', 'Peak_Inventory', 'Volatility']
            metro_analysis = metro_analysis.sort_values('Avg_Inventory', ascending=False).head(10)

            print("📍 Top 10 Metro Areas by Average Inventory:")
            print(metro_analysis.to_string())
            print()

    def analyze_market_dynamics(self) -> None:
        """Analyze market dynamics and relationships."""
        print("⚡ MARKET DYNAMICS ANALYSIS")
        print("=" * 45)

        # Price-to-rent ratios where we have both data
        if 'zhvi' in self.datasets and 'zori' in self.datasets:
            print("🏠 Price-to-Rent Analysis:")

            # Get latest values for common ZIP codes
            latest_zhvi = self.datasets['zhvi'].loc[
                self.datasets['zhvi']['Date'] == self.datasets['zhvi']['Date'].max()
            ].set_index('RegionName')['Value']

            latest_zori = self.datasets['zori'].loc[
                self.datasets['zori']['Date'] == self.datasets['zori']['Date'].max()
            ].set_index('RegionName')['Value']

            # Calculate price-to-rent ratios
            common_zips = latest_zhvi.index.intersection(latest_zori.index)
            price_to_rent = latest_zhvi[common_zips] / (latest_zori[common_zips] * 12)

            print(f"   • Common ZIP codes: {len(common_zips):,}")
            print(f"   • Median P/R ratio: {price_to_rent.median():.1f}")
            print(f"   • P/R range: {price_to_rent.min():.1f} - {price_to_rent.max():.1f}")

            # Plot distribution
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            price_to_rent.hist(bins=50, alpha=0.7, color=self.colors[0])
            plt.axvline(price_to_rent.median(), color='red', linestyle='--', label=f'Median: {price_to_rent.median():.1f}')
            plt.title('Price-to-Rent Ratio Distribution')
            plt.xlabel('Price-to-Rent Ratio')
            plt.ylabel('Frequency')
            plt.legend()

            # Top/bottom ratios
            plt.subplot(1, 2, 2)
            extreme_ratios = pd.concat([
                price_to_rent.nsmallest(10),
                price_to_rent.nlargest(10)
            ])
            extreme_ratios.plot(kind='bar', color=['green']*10 + ['red']*10)
            plt.title('Extreme Price-to-Rent Ratios')
            plt.ylabel('P/R Ratio')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()
            print()

    def identify_data_quality_issues(self) -> Dict[str, List[str]]:
        """
        Identify potential data quality issues.

        Returns:
            Dictionary of datasets and their quality issues
        """
        print("🔍 DATA QUALITY ASSESSMENT")
        print("=" * 40)

        issues = {}

        for name, df in self.datasets.items():
            dataset_issues = []

            # Check for missing values
            missing_pct = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_pct[missing_pct > 10]
            if not high_missing.empty:
                dataset_issues.append(f"High missing values: {list(high_missing.index)}")

            # Check for outliers in Value column
            if 'Value' in df.columns:
                Q1 = df['Value'].quantile(0.25)
                Q3 = df['Value'].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df['Value'] < (Q1 - 1.5 * IQR)) | (df['Value'] > (Q3 + 1.5 * IQR))]
                outlier_pct = (len(outliers) / len(df)) * 100
                if outlier_pct > 5:
                    dataset_issues.append(f"High outlier percentage: {outlier_pct:.1f}%")

            # Check for duplicate entries
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                dataset_issues.append(f"Duplicate rows: {duplicates:,}")

            # Check date consistency
            if 'Date' in df.columns:
                date_gaps = df.groupby('RegionName')['Date'].apply(
                    lambda x: (x.max() - x.min()).days / 30.44 / len(x)
                ).mean()
                if date_gaps > 2:  # More than 2 months between observations on average
                    dataset_issues.append(f"Irregular date spacing: {date_gaps:.1f} months avg gap")

            issues[name] = dataset_issues

            print(f"📊 {name.upper()}:")
            if dataset_issues:
                for issue in dataset_issues:
                    print(f"   ⚠️  {issue}")
            else:
                print(f"   ✅ No major quality issues detected")
            print()

        return issues

    def generate_correlation_analysis(self) -> None:
        """Generate correlation analysis between different metrics."""
        print("🔗 CORRELATION ANALYSIS")
        print("=" * 35)

        # Create a combined dataset for correlation analysis
        correlation_data = []

        # Get monthly aggregates for each dataset
        monthly_data = {}

        for name, df in self.datasets.items():
            if 'Date' in df.columns and 'Value' in df.columns:
                monthly = df.groupby(['Date'])['Value'].agg(['mean', 'median', 'std']).reset_index()
                monthly.columns = ['Date', f'{name}_mean', f'{name}_median', f'{name}_std']
                monthly_data[name] = monthly

        # Merge all monthly data
        if monthly_data:
            combined = list(monthly_data.values())[0]
            for df in list(monthly_data.values())[1:]:
                combined = pd.merge(combined, df, on='Date', how='outer')

            # Calculate correlations
            numeric_cols = combined.select_dtypes(include=[np.number]).columns
            correlation_matrix = combined[numeric_cols].corr()

            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                       center=0, square=True, fmt='.2f')
            plt.title('Real Estate Metrics Correlation Matrix')
            plt.tight_layout()
            plt.show()

            # Highlight strong correlations
            strong_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_val
                        ))

            if strong_corr:
                print("🔥 Strong Correlations (|r| > 0.7):")
                for var1, var2, corr in strong_corr:
                    print(f"   • {var1} ↔ {var2}: {corr:.3f}")
            print()

    def run_comprehensive_eda(self) -> Dict:
        """
        Run complete exploratory data analysis.

        Returns:
            Dictionary containing analysis results
        """
        print("🏠 COMPREHENSIVE REAL ESTATE DATA ANALYSIS")
        print("=" * 60)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Load data
        self.load_processed_data()

        if not self.datasets:
            print("❌ No datasets loaded. Please check data files.")
            return {}

        # Run all analyses
        results = {}

        # 1. Data Overview
        results['overview'] = self.generate_data_overview()

        # 2. Time Trends
        self.analyze_time_trends()

        # 3. Geographic Patterns
        self.analyze_geographic_patterns()

        # 4. Market Dynamics
        self.analyze_market_dynamics()

        # 5. Data Quality
        results['quality_issues'] = self.identify_data_quality_issues()

        # 6. Correlations
        self.generate_correlation_analysis()

        print("✅ ANALYSIS COMPLETE!")
        print("=" * 30)
        print("🔍 Key Insights:")
        print("   • Multiple time series spanning different periods")
        print("   • ZIP-level granularity for detailed analysis")
        print("   • Strong geographic coverage across US states")
        print("   • Rich feature engineering opportunities")
        print()
        print("🚀 Ready for Phase 3: Feature Engineering!")

        return results

def main():
    """Main execution function."""
    # Initialize EDA
    eda = RealEstateEDA()

    # Run comprehensive analysis
    results = eda.run_comprehensive_eda()

    return results

if __name__ == "__main__":
    results = main()