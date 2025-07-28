"""
Advanced Feature Engineering for Real Estate Prediction

This module creates sophisticated features based on the EDA insights:
- Time-based features (trends, seasonality, market cycles)
- Geographic features (state/metro encodings, regional patterns)
- Market dynamics (supply/demand ratios, price momentum)
- Cross-dataset features (price-to-rent ratios, market efficiency)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for real estate prediction.

    Creates sophisticated features based on market dynamics, temporal patterns,
    geographic clustering, and cross-dataset relationships.
    """

    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the feature engineer.

        Args:
            data_dir: Directory containing processed data files
        """
        # Find project root and set up paths
        current_dir = Path.cwd()
        project_root = current_dir

        while not (project_root / "data").exists() and project_root.parent != project_root:
            project_root = project_root.parent

        self.data_dir = project_root / data_dir
        self.output_dir = project_root / "data" / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.datasets: Dict[str, pd.DataFrame] = {}
        self.feature_datasets: Dict[str, pd.DataFrame] = {}
        self.logger = self._setup_logging()

        # Feature engineering parameters
        self.lag_periods = [1, 3, 6, 12, 24]  # months
        self.rolling_windows = [3, 6, 12, 24]  # months
        self.price_change_periods = [3, 6, 12, 24, 36]  # months

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_datasets(self) -> None:
        """Load all processed datasets."""
        self.logger.info(f"Loading datasets from: {self.data_dir}")

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

                # Debug: Show data structure
                self.logger.info(f"Columns in {name}: {list(df.columns)}")
                if 'Date' in df.columns:
                    self.logger.info(f"Sample Date values: {df['Date'].head().tolist()}")

                # Clean Date column - remove invalid dates
                if 'Date' in df.columns:
                    # Convert to datetime, coercing invalid dates to NaT
                    valid_dates = pd.to_datetime(df['Date'], errors='coerce')
                    invalid_mask = valid_dates.isna()

                    if invalid_mask.sum() > 0:
                        self.logger.warning(f"Removing {invalid_mask.sum():,} rows with invalid dates from {name}")
                        df = df[~invalid_mask].copy()

                    df['Date'] = pd.to_datetime(df['Date'])

                # Clean Value column
                if 'Value' in df.columns:
                    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                    invalid_values = df['Value'].isna()

                    if invalid_values.sum() > 0:
                        self.logger.warning(f"Removing {invalid_values.sum():,} rows with invalid values from {name}")
                        df = df[~invalid_values].copy()

                # Final cleanup - ensure we have required columns
                required_cols = ['Date', 'Value']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    self.logger.error(f"Missing required columns in {name}: {missing_cols}")
                    continue

                # Remove any completely empty rows
                df = df.dropna(how='all')

                self.datasets[name] = df
                self.logger.info(f"Successfully loaded {name}: {len(df):,} rows after cleaning")
            else:
                self.logger.warning(f"File not found: {file_path}")

    def create_temporal_features(self, df: pd.DataFrame, value_col: str = 'Value') -> pd.DataFrame:
        """
        Create advanced temporal features.

        Args:
            df: DataFrame with Date and Value columns
            value_col: Name of the value column

        Returns:
            DataFrame with temporal features added
        """
        self.logger.info("Creating temporal features...")

        # Sort by region and date
        df = df.sort_values(['RegionName', 'Date']).copy()

        # Basic time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week

        # Cyclical encoding for seasonality
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

        # Time since start (months)
        min_date = df['Date'].min()
        df['MonthsSinceStart'] = ((df['Date'] - min_date).dt.days / 30.44).round()

        # Age of data (how recent)
        max_date = df['Date'].max()
        df['MonthsSinceLatest'] = ((max_date - df['Date']).dt.days / 30.44).round()

        # Market era classification based on major events
        def classify_market_era(date):
            if date < pd.Timestamp('2008-01-01'):
                return 'pre_crisis'
            elif date < pd.Timestamp('2012-01-01'):
                return 'crisis_recovery'
            elif date < pd.Timestamp('2020-01-01'):
                return 'post_crisis'
            elif date < pd.Timestamp('2022-01-01'):
                return 'covid_boom'
            else:
                return 'post_covid'

        df['MarketEra'] = df['Date'].apply(classify_market_era)

        # Lag features
        for lag in self.lag_periods:
            df[f'{value_col}_lag_{lag}'] = df.groupby('RegionName')[value_col].shift(lag)

        # Rolling statistics
        for window in self.rolling_windows:
            df[f'{value_col}_rolling_mean_{window}'] = df.groupby('RegionName')[value_col].rolling(
                window=window, min_periods=1
            ).mean().reset_index(level=0, drop=True)

            df[f'{value_col}_rolling_std_{window}'] = df.groupby('RegionName')[value_col].rolling(
                window=window, min_periods=1
            ).std().reset_index(level=0, drop=True)

            df[f'{value_col}_rolling_min_{window}'] = df.groupby('RegionName')[value_col].rolling(
                window=window, min_periods=1
            ).min().reset_index(level=0, drop=True)

            df[f'{value_col}_rolling_max_{window}'] = df.groupby('RegionName')[value_col].rolling(
                window=window, min_periods=1
            ).max().reset_index(level=0, drop=True)

        # Price change features
        for period in self.price_change_periods:
            df[f'{value_col}_pct_change_{period}'] = df.groupby('RegionName')[value_col].pct_change(periods=period)
            df[f'{value_col}_diff_{period}'] = df.groupby('RegionName')[value_col].diff(periods=period)

        # Momentum and trend features
        df[f'{value_col}_momentum_short'] = df[f'{value_col}_rolling_mean_3'] / df[f'{value_col}_rolling_mean_12']
        df[f'{value_col}_momentum_long'] = df[f'{value_col}_rolling_mean_12'] / df[f'{value_col}_rolling_mean_24']

        # Volatility features
        df[f'{value_col}_volatility_3m'] = df[f'{value_col}_rolling_std_3'] / df[f'{value_col}_rolling_mean_3']
        df[f'{value_col}_volatility_12m'] = df[f'{value_col}_rolling_std_12'] / df[f'{value_col}_rolling_mean_12']

        # Trend strength (linear regression slope)
        def calculate_trend(series):
            if len(series) < 3:
                return 0
            x = np.arange(len(series))
            return np.polyfit(x, series, 1)[0]

        for window in [6, 12, 24]:
            df[f'{value_col}_trend_{window}'] = df.groupby('RegionName')[value_col].rolling(
                window=window, min_periods=3
            ).apply(calculate_trend).reset_index(level=0, drop=True)

        return df

    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create geographic and regional features.

        Args:
            df: DataFrame with geographic information

        Returns:
            DataFrame with geographic features added
        """
        self.logger.info("Creating geographic features...")

        # State-level statistics
        state_stats = df.groupby('StateName')['Value'].agg([
            'mean', 'median', 'std', 'count'
        ]).add_prefix('State_')

        df = df.merge(
            state_stats,
            left_on='StateName',
            right_index=True,
            how='left'
        )

        # Metro-level statistics (if available)
        if 'Metro' in df.columns:
            metro_stats = df.groupby('Metro')['Value'].agg([
                'mean', 'median', 'std', 'count'
            ]).add_prefix('Metro_')

            df = df.merge(
                metro_stats,
                left_on='Metro',
                right_index=True,
                how='left'
            )

        # Region size ranking
        df['StateRank'] = df.groupby('Date')['State_count'].rank(ascending=False)

        # Relative value position within state
        df['ValueRankInState'] = df.groupby(['StateName', 'Date'])['Value'].rank(pct=True)

        # Geographic clustering based on value patterns
        if len(df) > 1000:  # Only cluster if we have enough data
            region_profiles = df.groupby('RegionName')['Value'].agg([
                'mean', 'std', 'min', 'max'
            ]).fillna(0)

            if len(region_profiles) > 10:
                kmeans = KMeans(n_clusters=min(10, len(region_profiles)//100), random_state=42)
                region_profiles['Cluster'] = kmeans.fit_predict(region_profiles)

                df = df.merge(
                    region_profiles[['Cluster']],
                    left_on='RegionName',
                    right_index=True,
                    how='left'
                )
                df['GeographicCluster'] = df['Cluster'].fillna(-1).astype(int)
                df = df.drop('Cluster', axis=1)

        return df

    def create_market_dynamics_features(self) -> pd.DataFrame:
        """
        Create market dynamics features using multiple datasets.

        Returns:
            Combined DataFrame with market dynamics features
        """
        self.logger.info("Creating market dynamics features...")

        # Start with ZHVI as base (most comprehensive)
        if 'zhvi' not in self.datasets:
            raise ValueError("ZHVI dataset required for market dynamics features")

        base_df = self.datasets['zhvi'].copy()

        # Add price-to-rent ratios where possible
        if 'zori' in self.datasets:
            zori_df = self.datasets['zori'][['RegionName', 'Date', 'Value']].rename(
                columns={'Value': 'Rent'}
            )

            base_df = base_df.merge(
                zori_df,
                on=['RegionName', 'Date'],
                how='left'
            )

            # Calculate price-to-rent ratio
            base_df['PriceToRentRatio'] = base_df['Value'] / (base_df['Rent'] * 12)
            base_df['PriceToRentRatio'] = base_df['PriceToRentRatio'].replace([np.inf, -np.inf], np.nan)

            # Rent affordability (rent as % of median income - proxy using national median)
            median_income = 70000  # Approximate US median household income
            base_df['RentAffordabilityRatio'] = (base_df['Rent'] * 12) / median_income

            # Market efficiency indicators
            base_df['PriceRentSpread'] = base_df['Value'] - (base_df['Rent'] * 12 * 20)  # 20x rent rule

        # Add inventory and sales data aggregated to ZIP level
        # (Inventory and sales are at metro level, so we'll create regional indicators)

        if 'inventory' in self.datasets:
            # Get state-level inventory trends
            inventory_state = self.datasets['inventory'].copy()
            if 'StateName' in inventory_state.columns:
                state_inventory = inventory_state.groupby(['StateName', 'Date'])['Value'].agg([
                    'mean', 'sum'
                ]).reset_index()
                state_inventory.columns = ['StateName', 'Date', 'State_Inventory_Mean', 'State_Inventory_Total']

                base_df = base_df.merge(
                    state_inventory,
                    on=['StateName', 'Date'],
                    how='left'
                )

        if 'sales' in self.datasets:
            # Get state-level sales trends
            sales_state = self.datasets['sales'].copy()
            if 'StateName' in sales_state.columns:
                state_sales = sales_state.groupby(['StateName', 'Date'])['Value'].agg([
                    'mean', 'sum'
                ]).reset_index()
                state_sales.columns = ['StateName', 'Date', 'State_Sales_Mean', 'State_Sales_Total']

                base_df = base_df.merge(
                    state_sales,
                    on=['StateName', 'Date'],
                    how='left'
                )

        # Market velocity and liquidity indicators
        if 'State_Inventory_Total' in base_df.columns and 'State_Sales_Total' in base_df.columns:
            # Months of supply
            base_df['MonthsOfSupply'] = base_df['State_Inventory_Total'] / base_df['State_Sales_Total']
            base_df['MonthsOfSupply'] = base_df['MonthsOfSupply'].replace([np.inf, -np.inf], np.nan)

            # Market liquidity
            base_df['MarketLiquidity'] = base_df['State_Sales_Total'] / base_df['State_Inventory_Total']
            base_df['MarketLiquidity'] = base_df['MarketLiquidity'].replace([np.inf, -np.inf], np.nan)

        # National market indicators
        national_stats = base_df.groupby('Date')['Value'].agg([
            'mean', 'median', 'std'
        ]).add_prefix('National_')

        base_df = base_df.merge(
            national_stats,
            left_on='Date',
            right_index=True,
            how='left'
        )

        # Relative position vs national market
        base_df['ValueVsNational'] = base_df['Value'] / base_df['National_mean']
        base_df['ValueVsNationalMedian'] = base_df['Value'] / base_df['National_median']

        return base_df

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with target variables added
        """
        self.logger.info("Creating target variables...")

        # Sort by region and date
        df = df.sort_values(['RegionName', 'Date']).copy()

        # Future price targets (what we want to predict)
        target_horizons = [1, 3, 6, 12]  # months ahead

        for horizon in target_horizons:
            # Future value
            df[f'target_value_{horizon}m'] = df.groupby('RegionName')['Value'].shift(-horizon)

            # Future return (percentage change)
            df[f'target_return_{horizon}m'] = (
                df[f'target_value_{horizon}m'] / df['Value'] - 1
            ) * 100

            # Binary targets (up/down)
            df[f'target_direction_{horizon}m'] = (df[f'target_return_{horizon}m'] > 0).astype(int)

            # Categorical targets (strong down, down, flat, up, strong up)
            def categorize_return(ret):
                if pd.isna(ret):
                    return np.nan
                elif ret < -5:
                    return 0  # strong down
                elif ret < -1:
                    return 1  # down
                elif ret < 1:
                    return 2  # flat
                elif ret < 5:
                    return 3  # up
                else:
                    return 4  # strong up

            df[f'target_category_{horizon}m'] = df[f'target_return_{horizon}m'].apply(categorize_return)

        return df

    def run_feature_engineering(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete feature engineering pipeline.

        Returns:
            Dictionary of feature-engineered datasets
        """
        self.logger.info("🔧 STARTING ADVANCED FEATURE ENGINEERING")
        self.logger.info("=" * 60)

        # Load datasets
        self.load_datasets()

        if not self.datasets:
            raise ValueError("No datasets loaded. Please check data files.")

        # Create market dynamics features (combines all datasets)
        combined_df = self.create_market_dynamics_features()
        self.logger.info(f"Created market dynamics features: {combined_df.shape}")

        # Add temporal features
        combined_df = self.create_temporal_features(combined_df)
        self.logger.info(f"Added temporal features: {combined_df.shape}")

        # Add geographic features
        combined_df = self.create_geographic_features(combined_df)
        self.logger.info(f"Added geographic features: {combined_df.shape}")

        # Create target variables
        combined_df = self.create_target_variables(combined_df)
        self.logger.info(f"Added target variables: {combined_df.shape}")

        # Remove rows without targets (latest data points)
        training_df = combined_df.dropna(subset=[
            'target_value_1m', 'target_value_3m', 'target_value_6m', 'target_value_12m'
        ])

        self.logger.info(f"Training dataset shape: {training_df.shape}")

        # Feature summary
        feature_cols = [col for col in combined_df.columns
                       if col not in ['RegionID', 'RegionName', 'Date', 'Value', 'Dataset', 'GeographyLevel']]
        target_cols = [col for col in combined_df.columns if col.startswith('target_')]

        self.logger.info(f"Total features created: {len(feature_cols)}")
        self.logger.info(f"Target variables: {len(target_cols)}")

        # Save feature-engineered datasets
        output_file = self.output_dir / "feature_engineered_dataset.csv"
        combined_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved complete dataset: {output_file}")

        training_file = self.output_dir / "training_dataset.csv"
        training_df.to_csv(training_file, index=False)
        self.logger.info(f"Saved training dataset: {training_file}")

        # Create feature summary
        feature_summary = pd.DataFrame({
            'Feature': feature_cols,
            'Type': ['temporal' if any(x in col for x in ['lag', 'rolling', 'trend', 'pct_change'])
                    else 'geographic' if any(x in col for x in ['State', 'Metro', 'Cluster'])
                    else 'market_dynamics' if any(x in col for x in ['Ratio', 'Supply', 'National'])
                    else 'time' if any(x in col for x in ['Year', 'Month', 'Quarter'])
                    else 'other' for col in feature_cols]
        })

        summary_file = self.output_dir / "feature_summary.csv"
        feature_summary.to_csv(summary_file, index=False)

        # Print summary
        print("\n🎯 FEATURE ENGINEERING COMPLETE!")
        print("=" * 50)
        print(f"📊 Total observations: {len(combined_df):,}")
        print(f"📈 Training observations: {len(training_df):,}")
        print(f"🔧 Features created: {len(feature_cols):,}")
        print(f"🎯 Target variables: {len(target_cols):,}")
        print()
        print("📁 Feature categories:")
        print(feature_summary['Type'].value_counts().to_string())
        print()
        print("📂 Files saved:")
        print(f"   • {output_file}")
        print(f"   • {training_file}")
        print(f"   • {summary_file}")
        print()
        print("🚀 Ready for Phase 4: Model Development!")

        return {
            'complete': combined_df,
            'training': training_df,
            'feature_summary': feature_summary
        }

def main():
    """Main execution function."""
    engineer = AdvancedFeatureEngineer()
    results = engineer.run_feature_engineering()
    return results

if __name__ == "__main__":
    results = main()