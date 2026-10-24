# Mobile Product Segmentation and Recommendation System

## Project overview
An end-to-end analytics and machine-learning project based on the supplied **Global Mobile Reviews** dataset. It cleans review records, performs exploratory analysis, aggregates reviews to product level, applies K-Means segmentation, and recommends similar products using cosine similarity.

## Dataset snapshot
- Rows: 50,000
- Columns: 22
- Brands: 7
- Models: 22
- Countries: 8
- Average raw rating: 3.12/5
- Missing cells in raw file: 12,227
- Exact duplicate rows in raw file: 0

These are descriptive dataset statistics, not model-performance claims.

## Features
- Data cleaning: whitespace normalization, numeric/date conversion, duplicate removal, missing-value imputation.
- EDA: review counts by brand, sentiment distribution, average rating by brand, review trend over time.
- Product-level aggregation by brand/model.
- K-Means clustering (default 4 clusters) on scaled product attributes.
- Cosine-similarity recommendations based on price, ratings, and helpful votes.
- Streamlit dashboard with filters, charts, tables, and CSV downloads.

## Project structure
```text
Mobile_Product_Segmentation_Project/
├── app.py
├── run_analysis.py
├── requirements.txt
├── README.md
├── data/
│   ├── mobile_reviews_raw.csv
│   └── mobile_reviews_cleaned.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   └── modeling.py
└── outputs/
```

## Run on Windows (VS Code)
1. Extract the ZIP file.
2. Open the `Mobile_Product_Segmentation_Project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install libraries:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Generate analysis outputs:
   ```powershell
   python run_analysis.py
   ```
8. Launch the app:
   ```powershell
   streamlit run app.py
   ```
9. Streamlit normally opens the dashboard in your browser. Stop it with `Ctrl+C`.

## How the model works
1. Reviews are cleaned and aggregated into one row per brand/model.
2. Numeric product attributes are standardized using `StandardScaler`.
3. K-Means assigns each product to one of four clusters (or fewer if there are fewer products).
4. Recommendations rank other products by cosine similarity across standardized structured attributes.

## Important interpretation notes
- Cluster numbers (0–3) are labels, not ordered categories. Inspect cluster summaries before naming segments such as budget or premium.
- Similarity scores indicate closeness under the chosen features, not measured customer satisfaction or guaranteed recommendation quality.
- The dataset contains review-level fields. It does not necessarily contain all technical specifications for every phone, so the implementation uses the available numeric rating/price features.
- The project does not claim supervised prediction accuracy because no held-out supervised prediction task is defined.

## Troubleshooting
- If `requirements.txt` is not found, ensure the VS Code terminal is inside the folder that contains `app.py` and `requirements.txt`.
- If `streamlit` is not recognized after installation, activate `.venv` and run `python -m streamlit run app.py`.
- If file paths fail, keep the `data` folder beside `app.py`.

## Evaluation checklist
- [x] CSV loaded with Pandas
- [x] Missing values and duplicates handled
- [x] EDA visualizations
- [x] Feature scaling and K-Means clustering
- [x] Similarity-based recommendation logic
- [x] Streamlit interface
- [x] CSV exports and project documentation
