import pandas as pd
from pandas_datareader import data as web

def fetch_fred_yields(start="2000-01-01", end=None) -> pd.DataFrame:
    codes = {"y2": "DGS2", "y5": "DGS5", "y10": "DGS10"}
    df = web.DataReader(list(codes.values()), "fred", start, end)
    df = df.rename(columns={v: k for k, v in codes.items()})
    df = (df/ 100.00) # to decimals
    # Reindex to business days; forward-fill short gaps (holidays)
    idx = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(idx).ffill(limit=3).dropna()
    df.index.name = "date"
    return df[["y2", "y5", "y10"]]

if __name__ == "__main__":
    y = fetch_fred_yields()
    y.to_csv("yields.csv")
    print("yields.csv written successfully")