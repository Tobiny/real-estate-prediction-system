import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DatasetInfo:
    """
    Information about a manually downloaded dataset.

    Attributes:
        name: Dataset name
        file_path: Path to the CSV file
        geography_level: Geographic granularity (zip, metro, etc.)
        date_columns: List of column names that contain dates
        value_column: Main value column name
        description: Dataset description
    """
    name: str
    file_path: str
    geography_level: str
    date_columns: List[str]
    value_column: str
    description: str


class ManualZillowDataCollector:
    """
    Collector for manually downloaded Zillow datasets.

    This class handles loading, cleaning, and standardizing
    manually downloaded Zillow research data files.
    """

    def __init__(self, data_dir: str = "data/raw/zillow"):
        """
        Initialize the manual data collector.

        Args:
            data_dir: Directory containing the downloaded CSV files
        """
        # Find the project root directory (contains the data folder)
        current_dir = Path.cwd()
        project_root = current_dir

        # Go up until we find the data directory or reach the project root
        while not (project_root / "data").exists() and project_root.parent != project_root:
            project_root = project_root.parent

        # If we're running from src/data/collectors, go up 3 levels
        if "src" in str(current_dir) and "collectors" in str(current_dir):
            project_root = current_dir.parent.parent.parent

        self.data_dir = project_root / data_dir
        self.logger = self._setup_logging()
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.dataset_info: Dict[str, DatasetInfo] = {}

        self.logger.info(f"Looking for data files in: {self.data_dir.absolute()}")

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.

        Returns:
            Configured logger instance
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def register_dataset(self, dataset_info: DatasetInfo) -> None:
        """
        Register a dataset for collection.

        Args:
            dataset_info: Information about the dataset
        """
        self.dataset_info[dataset_info.name] = dataset_info
        self.logger.info(f"Registered dataset: {dataset_info.name}")

    def setup_default_datasets(self) -> None:
        """
        Set up the default Zillow datasets configuration.
        """
        # ZHVI All Homes (ZIP Level)
        self.register_dataset(DatasetInfo(
            name="zhvi_all_homes",
            file_path="zhvi_all_homes_zip.csv",
            geography_level="zip",
            date_columns=[],  # Will be detected automatically
            value_column="",  # Will be detected automatically
            description="Zillow Home Value Index - All Homes at ZIP level"
        ))

        # ZORI All Homes (ZIP Level)
        self.register_dataset(DatasetInfo(
            name="zori_all_homes",
            file_path="zori_all_homes_zip.csv",
            geography_level="zip",
            date_columns=[],
            value_column="",
            description="Zillow Observed Rent Index - All Homes at ZIP level"
        ))

        # For-Sale Inventory (Metro Level)
        self.register_dataset(DatasetInfo(
            name="inventory_metro",
            file_path="inventory_metro.csv",
            geography_level="metro",
            date_columns=[],
            value_column="",
            description="For-Sale Inventory at Metro level"
        ))

        # Sales Count (Metro Level)
        self.register_dataset(DatasetInfo(
            name="sales_count_metro",
            file_path="sales_count_metro.csv",
            geography_level="metro",
            date_columns=[],
            value_column="",
            description="Sales Count Nowcast at Metro level"
        ))

    def detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect date columns in a dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            List of column names that appear to contain dates
        """
        date_columns = []

        for col in df.columns:
            # Skip known non-date columns
            if col.lower() in ['regionid', 'sizerank', 'regionname', 'regiontype',
                               'statename', 'state', 'metro', 'countyname']:
                continue

            # Check if column contains date-like values
            try:
                # Try to parse a few values as dates
                sample_values = df[col].dropna().head(3)
                if len(sample_values) > 0:
                    # Check if it looks like a date (YYYY-MM-DD format)
                    first_val = str(sample_values.iloc[0])
                    if '-' in first_val and len(first_val) == 10:
                        pd.to_datetime(first_val)
                        date_columns.append(col)
                    # Check if it looks like a date (MM/DD/YYYY format)
                    elif '/' in first_val:
                        pd.to_datetime(first_val)
                        date_columns.append(col)
            except:
                continue

        return date_columns

    def detect_value_columns(self, df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
        """
        Detect numeric value columns (excluding geography and date columns).

        Args:
            df: DataFrame to analyze
            exclude_cols: Columns to exclude from detection

        Returns:
            List of numeric column names
        """
        value_columns = []

        for col in df.columns:
            if col in exclude_cols:
                continue

            # Skip known geography columns
            if col.lower() in ['regionid', 'sizerank', 'regionname', 'regiontype',
                               'statename', 'state', 'metro', 'countyname']:
                continue

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                value_columns.append(col)

        return value_columns

    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load a single dataset from CSV file.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            DataFrame with the loaded data or None if error
        """
        if dataset_name not in self.dataset_info:
            self.logger.error(f"Dataset {dataset_name} not registered")
            return None

        dataset_info = self.dataset_info[dataset_name]
        file_path = self.data_dir / dataset_info.file_path

        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            self.logger.info(f"Please download and place the file at: {file_path}")
            return None

        try:
            # Load the CSV
            self.logger.info(f"Loading dataset: {dataset_name}")
            df = pd.read_csv(file_path)

            # Log basic info
            self.logger.info(f"Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")

            # Detect date and value columns if not specified
            if not dataset_info.date_columns:
                date_cols = self.detect_date_columns(df)
                dataset_info.date_columns = date_cols
                self.logger.info(f"Detected date columns: {date_cols}")

            if not dataset_info.value_column:
                value_cols = self.detect_value_columns(df, dataset_info.date_columns)
                if value_cols:
                    dataset_info.value_column = value_cols[0]  # Use first numeric column
                    self.logger.info(f"Detected value columns: {value_cols}")

            return df

        except Exception as e:
            self.logger.error(f"Error loading {dataset_name}: {e}")
            return None

    def transform_to_long_format(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> pd.DataFrame:
        """
        Transform dataset from wide format (columns = dates) to long format.

        Args:
            df: DataFrame in wide format
            dataset_info: Information about the dataset

        Returns:
            DataFrame in long format with columns: RegionID, RegionName, Date, Value
        """
        # Identify geography columns (non-date columns)
        geography_cols = []
        for col in df.columns:
            if col.lower() in ['regionid', 'regionname', 'regiontype', 'statename',
                               'state', 'metro', 'countyname', 'sizerank']:
                geography_cols.append(col)

        self.logger.info(f"Geography columns: {geography_cols}")

        # Identify date columns - these should be in YYYY-MM-DD format
        date_cols = []
        for col in df.columns:
            if col not in geography_cols:
                # Check if column name looks like a date
                try:
                    # Try to parse the column name as a date
                    pd.to_datetime(col)
                    date_cols.append(col)
                except:
                    # If it's not a date, it might be another geography column
                    self.logger.warning(f"Column '{col}' doesn't look like a date, skipping")

        self.logger.info(f"Date columns found: {len(date_cols)} (sample: {date_cols[:5]})")

        if not date_cols:
            self.logger.error("No valid date columns found! Check data format.")
            return pd.DataFrame()

        # Melt the dataframe - use only valid date columns as value_vars
        try:
            df_long = pd.melt(
                df,
                id_vars=geography_cols,
                value_vars=date_cols,
                var_name='Date',
                value_name='Value'
            )

            # Convert Date column to datetime
            df_long['Date'] = pd.to_datetime(df_long['Date'])

            # Convert Value to numeric
            df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

            # Add dataset metadata
            df_long['Dataset'] = dataset_info.name
            df_long['GeographyLevel'] = dataset_info.geography_level

            # Remove rows with missing values
            initial_rows = len(df_long)
            df_long = df_long.dropna(subset=['Value'])
            final_rows = len(df_long)

            self.logger.info(
                f"Transformed to long format: {final_rows:,} rows (removed {initial_rows - final_rows:,} missing values)")

            return df_long

        except Exception as e:
            self.logger.error(f"Error during transformation: {e}")
            self.logger.error(f"Geography columns: {geography_cols}")
            self.logger.error(f"Date columns sample: {date_cols[:5]}")
            return pd.DataFrame()

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all registered datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        self.logger.info("Loading all datasets...")

        for dataset_name in self.dataset_info.keys():
            df = self.load_dataset(dataset_name)
            if df is not None:
                # Transform to long format
                dataset_info = self.dataset_info[dataset_name]
                df_long = self.transform_to_long_format(df, dataset_info)
                self.datasets[dataset_name] = df_long
            else:
                self.logger.warning(f"Skipping {dataset_name} - file not found")

        self.logger.info(f"Successfully loaded {len(self.datasets)} datasets")
        return self.datasets

    def get_data_summary(self) -> pd.DataFrame:
        """
        Get a summary of all loaded datasets.

        Returns:
            DataFrame with summary statistics
        """
        summary_data = []

        for name, df in self.datasets.items():
            if df is not None and not df.empty:
                dataset_info = self.dataset_info[name]

                # Safely get numeric values
                try:
                    # Convert Value column to numeric, errors='coerce' will turn non-numeric to NaN
                    numeric_values = pd.to_numeric(df['Value'], errors='coerce')
                    value_min = numeric_values.min()
                    value_max = numeric_values.max()
                    non_null_count = numeric_values.notna().sum()
                except:
                    value_min = 'N/A'
                    value_max = 'N/A'
                    non_null_count = 'N/A'

                summary_data.append({
                    'Dataset': name,
                    'Description': dataset_info.description,
                    'Geography': dataset_info.geography_level,
                    'Rows': len(df),
                    'Non_Null_Values': non_null_count,
                    'Unique_Regions': df['RegionName'].nunique() if 'RegionName' in df.columns else 'N/A',
                    'Date_Range_Start': df['Date'].min() if 'Date' in df.columns else 'N/A',
                    'Date_Range_End': df['Date'].max() if 'Date' in df.columns else 'N/A',
                    'Value_Range_Min': value_min,
                    'Value_Range_Max': value_max
                })

        return pd.DataFrame(summary_data)

    def save_processed_data(self, output_dir: str = "data/processed") -> None:
        """
        Save processed datasets to output directory.

        Args:
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, df in self.datasets.items():
            if df is not None and not df.empty:
                file_path = output_path / f"{name}_processed.csv"
                df.to_csv(file_path, index=False)
                self.logger.info(f"Saved {name} to {file_path}")

    def get_sample_data(self, dataset_name: str, n_rows: int = 10) -> Optional[pd.DataFrame]:
        """
        Get a sample of data from a specific dataset.

        Args:
            dataset_name: Name of the dataset
            n_rows: Number of rows to sample

        Returns:
            Sample DataFrame or None if dataset not found
        """
        if dataset_name not in self.datasets:
            self.logger.error(f"Dataset {dataset_name} not loaded")
            return None

        df = self.datasets[dataset_name]
        return df.head(n_rows)


def main():
    """
    Main function to demonstrate the manual data collector.
    """
    print("🏠 MANUAL ZILLOW DATA COLLECTOR")
    print("=" * 50)

    # Initialize collector
    collector = ManualZillowDataCollector()

    # Set up default datasets
    collector.setup_default_datasets()

    print("\n📋 REGISTERED DATASETS:")
    for name, info in collector.dataset_info.items():
        print(f"  • {name}: {info.description}")
        print(f"    File: {info.file_path}")
        print(f"    Geography: {info.geography_level}")
        print()

    print("\n📂 PLEASE DOWNLOAD THE FOLLOWING FILES:")
    print("=" * 50)
    print("1. Go to https://www.zillow.com/research/data/")
    print("2. Download these datasets and save them in data/raw/zillow/:")
    print()

    for name, info in collector.dataset_info.items():
        print(f"   📄 {info.file_path}")
        print(f"      → {info.description}")

    print("\n3. Then run this script again to load the data!")
    print()

    # Try to load datasets
    datasets = collector.load_all_datasets()

    if datasets:
        print("\n✅ SUCCESSFULLY LOADED DATASETS:")
        print("=" * 40)

        # Show summary
        summary = collector.get_data_summary()
        print(summary.to_string(index=False))

        # Show sample data
        print("\n🔍 SAMPLE DATA:")
        print("=" * 20)
        for name in datasets.keys():
            sample = collector.get_sample_data(name, 3)
            if sample is not None:
                print(f"\n{name.upper()}:")
                print(sample.to_string(index=False))
                print()

        # Save processed data
        collector.save_processed_data()
        print("\n💾 Processed data saved to data/processed/")

    else:
        print("\n❌ No datasets loaded. Please download the files first.")


if __name__ == "__main__":
    main()