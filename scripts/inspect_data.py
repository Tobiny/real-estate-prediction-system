"""
Quick Data Inspector

Inspect the processed data files to understand what's wrong with the Date column.
"""

import pandas as pd
from pathlib import Path


def inspect_data():
    """Inspect processed data files."""

    # Find project root
    current_dir = Path.cwd()
    project_root = current_dir
    while not (project_root / "data").exists() and project_root.parent != project_root:
        project_root = project_root.parent

    data_dir = project_root / "data" / "processed"

    print("🔍 DATA INSPECTION REPORT")
    print("=" * 50)
    print(f"Looking in: {data_dir}")
    print()

    files = [
        'zhvi_all_homes_processed.csv',
        'zori_all_homes_processed.csv',
        'inventory_metro_processed.csv',
        'sales_count_metro_processed.csv'
    ]

    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"📄 {filename}")
            print("-" * 30)

            # Read first few rows
            df = pd.read_csv(filepath, nrows=10)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            if 'Date' in df.columns:
                print("Date column sample:")
                print(df['Date'].head().tolist())
                print(f"Date column dtype: {df['Date'].dtype}")

                # Check unique values in Date column
                unique_dates = df['Date'].unique()
                print(f"Unique Date values (first 10): {unique_dates[:10].tolist()}")

            if 'Value' in df.columns:
                print("Value column sample:")
                print(df['Value'].head().tolist())
                print(f"Value column dtype: {df['Value'].dtype}")

            print()
        else:
            print(f"❌ {filename} - File not found")
            print()


if __name__ == "__main__":
    inspect_data()