"""TRANSFERFIT - Data inspection script.

Scans C:\\Football for all datasets, dumps schema/quality summaries.
Read-only: never modifies any raw file.
"""
import os
import json
import zipfile
import io
import pandas as pd

RAW = r"C:\Football"
OUT = r"C:\Football\DATA\project\outputs\tables"

os.makedirs(OUT, exist_ok=True)

EXT_MAP = {
    ".csv": "csv", ".parquet": "parquet", ".xlsx": "xlsx",
    ".json": "json", ".zip": "zip", ".pdf": "pdf",
}


def quick_read(path, ext, max_guess=True):
    """Return a dataframe preview where possible."""
    try:
        if ext == "parquet":
            return pd.read_parquet(path)
        if ext == "csv":
            return pd.read_csv(path)
        if ext == "xlsx":
            return pd.read_excel(path)
        if ext == "json":
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict) and "players" in obj:
                obj = obj["players"]
            if isinstance(obj, list):
                return pd.json_normalize(obj)
            return pd.DataFrame()
        if ext == "zip":
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                if names and names[0].endswith(".csv"):
                    with zf.open(names[0]) as fh:
                        return pd.read_csv(fh)
                return pd.DataFrame()
    except Exception as e:  # noqa: BLE001
        print(f"    [read error] {type(e).__name__}: {e}")
    return None


def hashable_frame(df):
    try:
        df.duplicated()
        return df
    except TypeError:
        df = df.copy()
        for c in df.columns:
            if df[c].map(lambda x: isinstance(x, (list, dict))).any():
                df[c] = df[c].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x)
        return df


def describe_frame(df, name):
    if df is None or len(df) == 0:
        return {"file": name, "rows": 0, "cols": 0, "columns": [], "dtypes": {},
                "nulls": {}, "duplicates": 0}
    dup = int(hashable_frame(df).duplicated().sum())
    return {
        "file": name,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicates": dup,
    }


files = sorted(os.listdir(RAW))
meta = {}
for f in files:
    path = os.path.join(RAW, f)
    if os.path.isdir(path):
        continue
    ext = os.path.splitext(f)[1].lower()
    kind = EXT_MAP.get(ext, "other")
    size = os.path.getsize(path)
    print(f"# {f}  [{kind}, {size:,} bytes]")
    if kind in ("csv", "parquet", "xlsx", "json", "zip"):
        df = quick_read(path, kind)
        d = describe_frame(df, f)
        d["size_bytes"] = size
        meta[f] = d
        print(f"  rows={d['rows']}  cols={d['cols']}  duplicates={d['duplicates']}")
        print(f"  columns: {', '.join(d['columns'])}")
        nz = {k: v for k, v in d["nulls"].items() if v > 0}
        if nz:
            print(f"  nulls>0 ({len(nz)}): {nz}")
        dt = set(d["dtypes"].values())
        print(f"  dtypes: {sorted(dt)}")
    else:
        print("  (no tabular read)")

with open(os.path.join(OUT, "dataset_inventory.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print("\nWrote outputs/tables/dataset_inventory.json")