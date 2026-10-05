# Spotify StreamPulse

A data warehousing and data mining platform for Spotify streaming data, built with
Python, SQLite and Streamlit.

- **Data warehouse:** ETL pipeline with validation rules and a star schema (SQLite)
- **OLAP:** slice, dice, roll-up, drill-down and pivot
- **Machine learning:** linear regression (simple and multiple), decision tree, Naive Bayes,
  K-Means, hierarchical clustering, Apriori association rules, and a what-if simulator
- **Security:** role-based access (Admin and Data Analyst), salted PBKDF2 password hashes,
  login lockout, structured audit logging

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e . --no-deps
streamlit run app.py
```

On first launch the app builds the warehouse and trains the models automatically (about 10 seconds).

## Demo accounts

| Role | Username | Password |
|------|----------|----------|
| Admin (6 pages) | `admin` | `Admin@123` |
| Data Analyst (8 pages) | `analyst` | `Analyst@123` |

These are demo credentials for evaluation only.

## Tests

```bash
pytest tests -q
```
