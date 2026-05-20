"""Quick probe: read 5 rows with polars inferred schema and print dtypes."""
import tempfile, os
import polars as pl

src = "/home/krishoffmann/projects/thesis_repository/final_dataset_creation/dataset_construction/outputs/stage6_full_predictions.jsonl"
lines = []
with open(src) as f:
    for i, line in enumerate(f):
        lines.append(line.strip())
        if i >= 9:
            break

with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
    tmp.write("\n".join(lines))
    tmpname = tmp.name

df = pl.read_ndjson(tmpname)
print("Inferred schema:")
for name, dtype in df.schema.items():
    print(f"  {name}: {dtype}")

os.unlink(tmpname)
