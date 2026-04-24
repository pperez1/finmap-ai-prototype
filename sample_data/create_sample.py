import pandas as pd

df = pd.DataFrame({
    "ReportDate": ["2026-01-01", "2026-02-01", "2026-03-01"],
    "Revenue": [100000, 150000, 120000],
    "OperatingCost": [70000, 90000, 80000],
    "DSCR": [1.4, 1.6, 1.5]
})

df.to_excel("sample_data/sample.xlsx", index=False)
exit()