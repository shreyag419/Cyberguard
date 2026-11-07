import glob
import pandas as pd

files = sorted(glob.glob("data/raw/*.csv"))
print(len(files), "files found")
for f in files:
    print(" ", f)

df = pd.read_csv(files[0], nrows=5)
df.columns = df.columns.str.strip()
print("Total columns:", len(df.columns))

need = ["Flow Packets/s", "SYN Flag Count", "RST Flag Count", "Label"]
print({c: c in df.columns for c in need})