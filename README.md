# Project 28 — Traffic Pattern Discovery in Smart Cities
**B.Tech machine learning project | Unsupervised clustering**

## Quick start (Google Colab, easiest)
1. Extract this ZIP and upload `Project28_Traffic_Pattern_Discovery.ipynb` to Google Colab using File > Upload notebook.
2. Choose Runtime > Run all. The notebook generates its bundled simulated data itself; no CSV upload, internet data fetch, or GPU is needed. Colab needs its normal Python packages.
3. At the end, edit the location, hour, vehicle_count and weekend inputs in the prototype cell and run it again. The cluster is a relative volume pattern, not a traffic forecast or measured congestion level.
4. Open the `outputs` directory in the Colab Files panel for CSVs and charts. Pre-executed results are also included in this ZIP.

## Run locally
Use Python 3.10+ and a terminal opened inside this extracted folder:
```
pip install -r requirements.txt
python traffic_project.py
```
Or open the notebook in Jupyter. All models are trained from scratch on each run. Seed 42. Reference run used Python 3.11+, scikit-learn 1.8.0, pandas 2.2.3 and NumPy 2.3.5; tiny score differences across versions are possible.

## Data and academic honesty
The default dataset is **synthetic**: 5 fictional roads × 28 days × 24 hours = 3,360 unique simulated location-hours, plus 12 exact duplicate rows. 20 readings are missing. Cleaning leaves 3,340 rows. The generator imposes rush-hour peaks and weekend effects and adds random variation/disruptions. It has no target labels. The project does not establish actual traffic patterns in Pune or elsewhere.

A live download of a real public dataset was not available in the execution environment. If your faculty requires real observations, obtain the NYC DOT Automated Traffic Volume Counts CSV from:
https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/data
Filter a manageable time window (at least 10 distinct dates with complete counts and several locations; ideally the same locations in each split), export CSV, then run:
```
python import_nyc.py raw_nyc.csv normalized_nyc.csv
python traffic_project.py normalized_nyc.csv
```
For Colab, upload normalized_nyc.csv and set DATA_FILE to its path in the dataset cell before running. The converter aggregates four 15-minute volume records into one hourly count, preserving location and direction; incomplete hours are excluded. NYC surveys do not cover every location every day. The real-data path is optional and has not been tested against a downloaded NYC file. Recompute the report and slides if you change data; the supplied report describes the simulation only. For large inputs the clustering code samples up to 3,500 train rows and 2,000 rows for each evaluation split.

## Files
- `Project28_Traffic_Pattern_Discovery.ipynb`: self-contained executed notebook and prototype
- `traffic_project.py`: same reproducible Python workflow
- `traffic_dataset.csv`: simulated raw dataset
- `import_nyc.py`: optional NYC CSV converter
- `outputs/`: model grid, comparisons, test assignments, cluster profiles, ambiguity diagnostics and four charts
- `Project_Report.docx`: editable report with actual results from the simulation
- `Project_Presentation.pptx`: presentation slides
- `requirements.txt`: Python dependencies

## Interpretation
K-Means and Ward are nearly tied. DBSCAN is sensitive to distance and density settings and produces fragmented clusters on this representation. Silhouette is not classification accuracy. Use results for academic analysis only, not live traffic signal control. Fill in your name, roll number and team contribution before submission; obtain faculty approval of the dataset.
