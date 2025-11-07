# AAPL Exploratory Analysis Script
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "AAPL_aggregated.json"   # change to your path
OUTPUT_DIR = "aapl_exploratory_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

history = data["history"]

rows = []
for d in history:
    date = d["date"]
    md = d.get("market_data", {}) or {}
    headlines = d.get("news_headlines", []) or []
    rows.append({
        "date": pd.to_datetime(date).date(),
        "high": md.get("high", np.nan),
        "low": md.get("low", np.nan),
        "close": md.get("close", np.nan),
        "volume": md.get("volume", np.nan),
        "headline_count": len(headlines),
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

for col in ["high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

start_date = df["date"].min()
end_date = df["date"].max()

full_range = pd.date_range(start=start_date, end=end_date, freq="D")
coverage_pct = len(df) / len(full_range) * 100

missing_calendar_dates = set(full_range.date) - set(df["date"].dt.date)
has_calendar_gaps = len(missing_calendar_dates) > 0

flat_close_runs = (df["close"].diff() == 0).rolling(5).sum().fillna(0)
num_flat_runs_ge5 = int((flat_close_runs >= 5).sum())

df["close_shift1"] = df["close"].shift(1)
df["ret"] = (df["close"] - df["close_shift1"]) / df["close_shift1"]
for w in [7, 30, 180]:
    df[f"ma_{w}"] = df["close"].rolling(w, min_periods=1).mean()

df["abs_ret"] = df["ret"].abs()

df["has_news"] = df["headline_count"] > 0
num_news_days = int(df["has_news"].sum())
pct_news_days = 100.0 * num_news_days / len(df)

event_idx = df.index[df["has_news"]].tolist()
window = 3
aligned = defaultdict(list)
for idx in event_idx:
    for tau in range(-window, window+1):
        j = idx + tau
        if 0 <= j < len(df):
            aligned[tau].append(df.loc[j, "ret"])

event_study_mean = {tau: (np.nan if len(v)==0 else float(np.nanmean(v))) for tau, v in aligned.items()}
event_study_count = {tau: len([x for x in v if pd.notna(x)]) for tau, v in aligned.items()}

headline_day_ret_std = float(df.loc[df["has_news"], "ret"].std(skipna=True)) if num_news_days > 0 else np.nan
noheadline_day_ret_std = float(df.loc[~df["has_news"], "ret"].std(skipna=True)) if num_news_days < len(df) else np.nan
headline_nextday_ret_mean = float(df.loc[df["has_news"], "ret"].shift(-1).mean(skipna=True)) if num_news_days > 0 else np.nan
noheadline_nextday_ret_mean = float(df.loc[~df["has_news"], "ret"].shift(-1).mean(skipna=True)) if num_news_days < len(df) else np.nan

lag_corr_same = float(pd.concat([df["headline_count"], df["ret"]], axis=1).corr().iloc[0,1])
lag_corr_next = float(pd.concat([df["headline_count"], df["ret"].shift(-1)], axis=1).corr().iloc[0,1])

summary = {
    "date_range": [str(start_date.date()), str(end_date.date())],
    "rows": int(len(df)),
    "calendar_days_in_range": int(len(full_range)),
    "calendar_coverage_pct": coverage_pct,
    "has_calendar_gaps": bool(has_calendar_gaps),
    "num_flat_close_window5_markers": num_flat_runs_ge5,
    "num_news_days": num_news_days,
    "pct_news_days": pct_news_days,
    "headline_day_ret_std": headline_day_ret_std,
    "noheadline_day_ret_std": noheadline_day_ret_std,
    "headline_nextday_ret_mean": headline_nextday_ret_mean,
    "noheadline_nextday_ret_mean": noheadline_nextday_ret_mean,
    "lag_corr_same_day_headlines_vs_ret": lag_corr_same,
    "lag_corr_next_day_headlines_vs_ret": lag_corr_next,
    "event_study_mean_ret": event_study_mean,
    "event_study_counts": event_study_count,
}

summary_df = pd.DataFrame({
    "metric": list(summary.keys()),
    "value": [str(v) for v in summary.values()]
})
summary_csv_path = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
summary_df.to_csv(summary_csv_path, index=False)

plt.figure(figsize=(10,5))
plt.plot(df["date"], df["close"], label="Close")
plt.plot(df["date"], df["ma_7"], label="MA 7")
plt.plot(df["date"], df["ma_30"], label="MA 30")
plt.plot(df["date"], df["ma_180"], label="MA 180")
plt.title("AAPL Close with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
fig1_path = os.path.join(OUTPUT_DIR, "aapl_close_ma.png")
plt.tight_layout()
plt.savefig(fig1_path, dpi=150)
plt.close()

plt.figure(figsize=(10,5))
plt.plot(df["date"], df["volume"])
plt.title("AAPL Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
fig2_path = os.path.join(OUTPUT_DIR, "aapl_volume.png")
plt.tight_layout()
plt.savefig(fig2_path, dpi=150)
plt.close()

plt.figure(figsize=(8,5))
hist_vals = df["ret"].replace([np.inf, -np.inf], np.nan).dropna()
plt.hist(hist_vals, bins=100)
plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
fig3_path = os.path.join(OUTPUT_DIR, "aapl_return_distribution.png")
plt.tight_layout()
plt.savefig(fig3_path, dpi=150)
plt.close()

plt.figure(figsize=(10,5))
plt.plot(df["date"], df["headline_count"])
plt.title("Headline Count Per Day")
plt.xlabel("Date")
plt.ylabel("Count")
fig4_path = os.path.join(OUTPUT_DIR, "aapl_headline_count.png")
plt.tight_layout()
plt.savefig(fig4_path, dpi=150)
plt.close()

taus = sorted(event_study_mean.keys())
means = [event_study_mean[t] for t in taus]
plt.figure(figsize=(8,5))
plt.bar([str(t) for t in taus], means)
plt.title("Event Study: Mean Return around Headline Days")
plt.xlabel("Relative Day (tau)")
plt.ylabel("Mean Return")
fig5_path = os.path.join(OUTPUT_DIR, "aapl_event_study_mean_returns.png")
plt.tight_layout()
plt.savefig(fig5_path, dpi=150)
plt.close()

print("Outputs saved in:", os.path.abspath(OUTPUT_DIR))
print("Figures:")
print("-", fig1_path)
print("-", fig2_path)
print("-", fig3_path)
print("-", fig4_path)
print("-", fig5_path)
print("Tables:")
print("-", summary_csv_path)
print("-", os.path.join(OUTPUT_DIR, "event_study_table.csv"))
