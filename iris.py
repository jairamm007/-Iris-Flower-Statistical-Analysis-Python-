import pandas as pd
import random
import datetime
from functools import reduce
from pathlib import Path
from io import StringIO
import sys

# -------- FILE NAME --------
script_dir = Path(__file__).resolve().parent
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
rand_suffix = random.randint(1000, 9999)
filename = script_dir / f"iris_analysis_{timestamp}_{rand_suffix}.txt"
csv_candidates = [
    script_dir / "iris.csv",
    script_dir.parent / "iris.csv",
    Path.cwd() / "iris.csv",
]

# -------- READ FILE --------
try:
    csv_path = next(path for path in csv_candidates if path.exists())
    df = pd.read_csv(csv_path)
except StopIteration:
    print("Error loading file: iris.csv not found.")
    print("Looked in:")
    for path in csv_candidates:
        print(" -", path)
    sys.exit(1)
except FileNotFoundError:
    print("Error loading file: iris.csv not found.")
    sys.exit(1)
except PermissionError:
    print("Error loading file: permission denied while reading iris.csv.")
    sys.exit(1)
except OSError as e:
    print("Error loading file:", e)
    sys.exit(1)
except Exception as e:
    print("Error loading file:", e)
    sys.exit(1)

# -------- BASIC INFO --------
head = df.head(10)
tail = df.tail(5)
info_buffer = StringIO()
df.info(buf=info_buffer)
info_text = info_buffer.getvalue().rstrip()

# -------- RELATIONAL FILTER --------
selected_rows_text = ""
for _, row in df.iterrows():
    if row["sepal_length"] > 5.0:
        selected_rows_text += (
            f"  ({row['sepal_length']}, {row['sepal_width']}, "
            f"{row['petal_length']}, {row['petal_width']}, {row['species']})\n"
        )

# -------- OUTLIER FUNCTION --------
def find_outliers(lst):
    s = sorted(lst)
    n = len(s)
    Q1 = s[n//4]
    Q3 = s[(3*n)//4]
    IQR = Q3 - Q1
    low = Q1 - 1.5*IQR
    high = Q3 + 1.5*IQR
    out = []
    for x in s:
        if x < low:
            out.append(x)
        elif x > high:
            out.append(x)
        else:
            continue
    return Q1, Q3, IQR, low, high, out

outlier_text = ""
total_out = 0
for col in df.columns[:4]:
    Q1,Q3,IQR,low,high,out = find_outliers(df[col].tolist())
    total_out += len(out)
    outlier_text += f"""
  {col}:
    Q1={Q1:.4f}  Q3={Q3:.4f}  IQR={IQR:.4f}
    Fences: [{low:.4f}, {high:.4f}]
    Outliers ({len(out)}): {out}
"""

# -------- NEW COLUMNS --------
df["petal_area"] = df["petal_length"] * df["petal_width"]
df["petal_to_sepal_ratio"] = df["petal_length"] / df["sepal_length"]
df["combined_score"] = df["sepal_length"] + df["petal_length"] - df["sepal_width"]

validation_text = ""
valid_count = 0
for _, row in df.iterrows():
    expected_area = row["petal_length"] * row["petal_width"]
    expected_ratio = row["petal_length"] / row["sepal_length"]
    expected_score = row["sepal_length"] + row["petal_length"] - row["sepal_width"]
    if (
        row["petal_area"] == expected_area
        and row["petal_to_sepal_ratio"] == expected_ratio
        and row["combined_score"] == expected_score
    ):
        valid_count += 1
    else:
        validation_text += (
            f"  Invalid engineered values for row: {row['sepal_length']}, "
            f"{row['sepal_width']}, {row['petal_length']}, {row['petal_width']}\n"
        )

# -------- BITWISE --------
enc = {"setosa":1,"versicolor":2,"virginica":3}
species_binary = [bin(value) for value in enc.values()]
bitwise_text = ""
for a in enc:
    for b in enc:
        res = enc[a] & enc[b]
        bitwise_text += f"  {a:<14} {enc[a]:<8} {b:<14} {enc[b]:<8} {res:<8} 0b{bin(res)[2:]}\n"

# -------- UNIQUE SET --------
unique = set()
for _,r in df.iterrows():
    unique.add((r["sepal_length"],r["sepal_width"],r["petal_length"],r["petal_width"]))

samples = random.sample(list(unique), min(10, len(unique)))
sample_text = ""
for i,s in enumerate(samples,1):
    sample_text += f"    {i:>2}. ({s[0]}, {s[1]}, {s[2]}, {s[3]})\n"

# -------- HANDLE MISSING --------
for col in df.columns[:4]:
    mean = df[col].mean()
    df[col] = df[col].apply(lambda x: mean if x==0 or pd.isna(x) else x)

# -------- RECURSION --------
def rec_sum(lst):
    return 0 if not lst else lst[0] + rec_sum(lst[1:])

manual_stats = ""
for col in df.columns[:4]:
    lst = df[col].tolist()
    mean = rec_sum(lst)/len(lst)
    s = sorted(lst)
    n = len(lst)
    median = s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
    mode = max(set(lst), key=lst.count)
    manual_stats += f"  {col:<24} {mean:.4f} {median:>10.2g} {mode:>10.2g} {min(lst):>10.1f} {max(lst):>10.1f}\n"

# -------- FUNCTIONAL --------
pl = df["petal_length"].tolist()
mean_pl = sum(pl)/len(pl)
filtered = list(filter(lambda x:x>mean_pl, pl))
mapped_filtered = list(map(lambda x: round(x, 2), filtered))
product = reduce(lambda a,b:a*b, mapped_filtered,1)

# -------- BUILD FINAL REPORT --------
report = f"""======================================================================
    IRIS FLOWER STATISTICAL ANALYSIS REPORT
======================================================================

Dataset file         : iris.csv
Total records        : {len(df)}
Total features       : {len(df.columns)}

----------------------------------------------------------------------
1. DATASET OVERVIEW
----------------------------------------------------------------------

First 10 rows:
{head.to_string(index=False)}

Last 5 rows:
{tail.to_string(index=False)}

Basic info:
{info_text}

Species distribution:
"""

for sp,c in df["species"].value_counts().items():
    report += f"  {sp:<12}: {c} rows ({(c/len(df)*100):.1f} %)\n"

report += f"""

Selected rows where sepal_length > 5.0:
{selected_rows_text}

Column summary:
{df[['sepal_length','sepal_width','petal_length','petal_width']].describe().round(4)}

----------------------------------------------------------------------
2. OUTLIER DETECTION SUMMARY (IQR method)
----------------------------------------------------------------------
{outlier_text}
  Total outlier values: {total_out}

----------------------------------------------------------------------
3. ENGINEERED FEATURES
----------------------------------------------------------------------

  petal_area           = petal_length * petal_width
  petal_to_sepal_ratio = petal_length / sepal_length
  combined_score       = sepal_length + petal_length - sepal_width

    Validation summary   : {valid_count} / {len(df)} rows matched the engineered formulas

{validation_text if validation_text else "  All engineered rows validated successfully.\n"}

{df[['petal_area','petal_to_sepal_ratio','combined_score']].describe().round(4)}

----------------------------------------------------------------------
4. SPECIES BINARY ENCODING & BITWISE AND TABLE
----------------------------------------------------------------------

    Species binary values: {species_binary}

{bitwise_text}

----------------------------------------------------------------------
5. UNIQUE MEASUREMENT COMBINATIONS
----------------------------------------------------------------------

  Total rows                : {len(df)}
  Unique combinations       : {len(unique)}

{sample_text}

----------------------------------------------------------------------
6. MANUAL vs PANDAS STATISTICS
----------------------------------------------------------------------

{manual_stats}

----------------------------------------------------------------------
7. FUNCTIONAL PROGRAMMING RESULTS
----------------------------------------------------------------------

  petal_length mean       : {mean_pl:.4f}
  count > mean            : {len(filtered)}
    map-rounded values      : {mapped_filtered}
    product                 : {product:.4e}

======================================================================
END OF REPORT
======================================================================
"""

# -------- PRINT REPORT --------
print(report)

# -------- SAVE FILE --------
try:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print("Report saved as:", filename)
except Exception as e:
    print("Error saving file:", e)