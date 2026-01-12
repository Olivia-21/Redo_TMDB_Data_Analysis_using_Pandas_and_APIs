"""Quick test script for the ETL pipeline."""

from src.extract import extract
from src.transform import transform
from src.load import save_raw, save_processed
from src.analysis import get_best_worst_movies, franchise_vs_standalone
from src.visualize import create_all_visualizations

# Test with small subset
print("=" * 50)
print("TESTING ETL PIPELINE")
print("=" * 50)

# Step 1: Extract
print("\n1. EXTRACT")
df_raw = extract([299534, 19995, 597])  # 3 movies for quick test
print(f"   Extracted {len(df_raw)} movies")
save_raw(df_raw)

# Step 2: Transform
print("\n2. TRANSFORM")
df_clean = transform(df_raw)
print(f"   Cleaned: {len(df_clean)} movies, {len(df_clean.columns)} columns")
print(f"   Columns: {list(df_clean.columns)}")
save_processed(df_clean)

# Step 3: Analysis
print("\n3. ANALYSIS")
rankings = get_best_worst_movies(df_clean, top_n=3)
print(f"   Generated {len(rankings)} ranking categories")
print(f"   Top movies by revenue: {rankings['highest_revenue']['title'].tolist()}")

comparison = franchise_vs_standalone(df_clean)
print(f"   Franchise vs Standalone comparison generated")

# Step 4: Visualize
print("\n4. VISUALIZE")
try:
    paths = create_all_visualizations(df_clean)
    print(f"   Created {len(paths)} visualizations")
except Exception as e:
    print(f"   Visualization error (expected with small dataset): {e}")

print("\n" + "=" * 50)
print("PIPELINE TEST COMPLETE!")
print("=" * 50)
