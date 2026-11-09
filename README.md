# NVDA Stock Direction Classification

A reproducible notebook that adapts the [GeeksforGeeks stock prediction workflow](https://www.geeksforgeeks.org/machine-learning/stock-price-prediction-using-machine-learning-in-python/) to daily NVIDIA (NVDA) market data.

## Project question

Can daily OHLC price features and a quarter-end indicator classify whether NVDA's next trading-day closing price will be higher than the current day's close?

This is a binary classification exercise. It does not predict a future price level or establish a profitable trading strategy.

## Workflow

1. Download and inspect ten years of daily data from Yahoo Finance with yfinance.
2. Explore the price history, missing values, distributions and outliers.
3. Engineer open-close, low-high and quarter-end features; define the next-day direction target.
4. Split chronologically, using the latest 10% as a holdout.
5. Compare Logistic Regression, polynomial SVC and XGBoost.
6. Evaluate ROC-AUC, accuracy, balanced accuracy, precision, recall, F1 and a confusion matrix.

The notebook improves on a naive random split by preserving time order and fitting preprocessing only on training data. It also removes the final row, whose next-day target is unknown, rather than labeling it as a down day.

## Run

Open [ML_stock_data_sci_model.ipynb](ML_stock_data_sci_model.ipynb) in Jupyter or Google Colab. Install the dependencies with:

```bash
pip install -r requirements.txt
```

The notebook downloads data at runtime. Yahoo Finance data can change and may be unavailable temporarily.

## Limitations

- The target is next-day direction, not price or return magnitude.
- A classification score does not account for transaction costs, execution timing, position sizing or risk.
- Results depend on the downloaded date range and should not be treated as investment advice.
