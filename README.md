# Data folder

Place the three Excel files here so `ts-handbook.qmd` analyses your own data.
If a file is missing, the document generates clearly labelled simulated data instead.

| File name (exact) | Layout | Notes |
|---|---|---|
| `Nifty 50.xlsx` | Column 1 `Date`, column 2 `Price` | Extra columns are ignored. Commas in prices are removed automatically. |
| `World Indices.xlsx` | Column 1 `Date`, then one price column per market | Column names become series names (spaces become dots). |
| `Sector Wise.xlsx` | Column 1 `Date`, then `Auto`, `Bank`, `FS`, `FMCG`, `Healthcare` | These five names must be present. |

Dates may be real Excel dates or text in one of these forms: `2025-03-31`, `31-03-2025`,
`31/03/2025`, `31032025`, `2025/03/31`. Prices must be levels (not returns); the
document takes logs and differences itself. To use different file names, edit
`cfg$files` in the *Configuration* section of `ts-handbook.qmd`.
