# K-Means Segmentation

A minimal Streamlit app that clusters data using K-Means and visualises results with animated 2D and interactive 3D scatter plots.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datasets

Two bundled CSVs in `datasets/`:

| Dataset           | Features                         | Rows |
| ----------------- | -------------------------------- | ---- |
| Customer Spending | Age, AnnualIncome, SpendingScore | 80   |
| Student Marks     | Math, Science, English           | 60   |

## How It Works

1. Load — reads the selected CSV from `datasets/`
2. Scale — StandardScaler normalises features to mean=0, std=1
3. Cluster — KMeans with k-means++ init, default K=2
4. Visualise — animated 2D scatter showing iteration convergence, plus interactive 3D scatter

## Project Structure

```
K-means/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── datasets/
    ├── customer_spending.csv
    └── student_marks.csv
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Select your repo, branch main, main file app.py
4. Click Deploy

## Tech Stack

- Streamlit — UI framework
- scikit-learn — KMeans, StandardScaler
- Plotly — 2D and 3D charts
- pandas — data handling

## License

MIT
