Intelligent Financial Audit Dashboard with AI-Driven Anomaly Detection

This project delivers a complete financial analytics solution combining Power BI, Python, and Machine Learning to detect unusual transaction patterns and assist in financial auditing.

🔹 Project Features

Cleaned and standardized transaction dataset

Interactive Power BI dashboard with 3 pages:

Sales Overview

Product & Payment Insights

AI-Based Anomaly Detection

Python Isolation Forest model detecting anomalies

Integrated anomaly scoring & flags inside Power BI

🔹 Repo Structure
├── data/  
├── powerbi/  
├── python/
├── docs/
└── README.md

🔹 Technologies Used

Python (Pandas, Scikit-Learn)

Power BI (DAX, Power Query)

Power Query M

Machine Learning (Isolation Forest)

🔹 Dashboard Preview

(Insert screenshots from docs/screenshots)
Example:

🔹 Machine Learning Model

The Isolation Forest model was trained on:

Price

Quantity

Total Amount

Encoded Product, Payment, Status

Outputs:

anomaly_flag

anomaly_score

🔹 How to Use

Download the .pbix file

Load datasets from /data/

Explore the dashboard and anomaly insights
