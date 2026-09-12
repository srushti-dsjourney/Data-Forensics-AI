from flask import Flask, request, send_file, render_template

from dotenv import load_dotenv

load_dotenv()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "data_forensics_db")
}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )


# ============================================================
# SAVE FORENSIC RESULT TO MYSQL
# ============================================================

def save_forensic_result(
    dataset_name,
    rows_count,
    columns_count,
    missing_values,
    duplicate_records,
    invalid_values,
    outlier_count,
    forensic_score
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        INSERT INTO forensic_datasets
        (
            dataset_name,
            rows_count,
            columns_count,
            missing_values,
            duplicate_records,
            invalid_values,
            outlier_count,
            forensic_score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        dataset_name,
        rows_count,
        columns_count,
        missing_values,
        duplicate_records,
        invalid_values,
        outlier_count,
        forensic_score
    )

    cursor.execute(query, values)

    connection.commit()

    cursor.close()
    connection.close()


# ============================================================
# BASIC FORENSIC ANALYSIS
# ============================================================

def basic_forensic_analysis(df):

    rows = len(df)

    columns = len(df.columns)

    missing_values = int(df.isnull().sum().sum())

    missing_percentage = round(
        (missing_values / (rows * columns)) * 100,
        2
    ) if rows > 0 and columns > 0 else 0

    duplicate_records = int(
        df.duplicated().sum()
    )

    data_types = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    return {
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "missing_percentage": missing_percentage,
        "duplicate_records": duplicate_records,
        "data_types": data_types
    }


# ============================================================
# INVALID VALUE DETECTION
# ============================================================

def detect_invalid_values(df):

    invalid_count = 0

    checks = {
        "Age": lambda x: x < 0,
        "Rating": lambda x: (x < 0) | (x > 5),
        "Purchase_Amount": lambda x: x < 0,
        "Orders": lambda x: x < 0
    }

    for column, condition in checks.items():

        if column in df.columns:

            try:

                invalid_count += int(
                    condition(
                        pd.to_numeric(
                            df[column],
                            errors="coerce"
                        )
                    ).sum()
                )

            except Exception:

                pass

    return invalid_count


# ============================================================
# OUTLIER DETECTION USING IQR
# ============================================================

def detect_outliers(df):

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    total_outliers = 0

    outlier_details = {}

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:

            continue

        Q1 = series.quantile(0.25)

        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)

        upper_bound = Q3 + (1.5 * IQR)

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        count = len(outliers)

        if count > 0:

            outlier_details[column] = count

            total_outliers += count

    return {
        "total_outliers": total_outliers,
        "details": outlier_details
    }


# ============================================================
# FORENSIC SCORE
# ============================================================

def calculate_forensic_score(
    df,
    missing_values,
    duplicate_records,
    invalid_values,
    outlier_count
):

    total_records = len(df)

    if total_records == 0:

        return 0

    missing_penalty = min(
        (missing_values / total_records) * 20,
        20
    )

    duplicate_penalty = min(
        (duplicate_records / total_records) * 20,
        20
    )

    invalid_penalty = min(
        (invalid_values / total_records) * 20,
        20
    )

    outlier_penalty = min(
        (outlier_count / total_records) * 20,
        20
    )

    score = (
        100
        - missing_penalty
        - duplicate_penalty
        - invalid_penalty
        - outlier_penalty
    )

    return round(
        max(score, 0),
        2
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:

        return None

    return numeric_df.corr()


# ============================================================
# VISUALIZATION GENERATION
# ============================================================

def generate_visualizations(df):

    os.makedirs(
        "static",
        exist_ok=True
    )

    chart_paths = {}

    # --------------------------------------------------------
    # Missing Values Chart
    # --------------------------------------------------------

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) > 0:

        plt.figure(
            figsize=(10, 5)
        )

        missing.plot(
            kind="bar"
        )

        plt.title(
            "Missing Values by Column"
        )

        plt.xlabel(
            "Columns"
        )

        plt.ylabel(
            "Missing Values"
        )

        plt.tight_layout()

        path = os.path.join(
            "static",
            "missing_values.png"
        )

        plt.savefig(path)

        plt.close()

        chart_paths[
            "missing_values"
        ] = "/data/missing_values.png"

    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] >= 2:

        plt.figure(
            figsize=(10, 7)
        )

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(
            "Correlation Heatmap"
        )

        plt.tight_layout()

        path = os.path.join(
            "static",
            "correlation_heatmap.png"
        )

        plt.savefig(path)

        plt.close()

        chart_paths[
            "correlation"
        ] = "/data/correlation_heatmap.png"

    # --------------------------------------------------------
    # Numeric Distributions
    # --------------------------------------------------------

    if numeric_df.shape[1] > 0:

        numeric_df.hist(
            figsize=(12, 8)
        )

        plt.tight_layout()

        path = os.path.join(
            "static",
            "numeric_distributions.png"
        )

        plt.savefig(path)

        plt.close()

        chart_paths[
            "distribution"
        ] = "/data/numeric_distributions.png"

    # --------------------------------------------------------
    # Boxplots
    # --------------------------------------------------------

    if numeric_df.shape[1] > 0:

        plt.figure(
            figsize=(12, 6)
        )

        numeric_df.boxplot()

        plt.title(
            "Numeric Feature Boxplots"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        path = os.path.join(
            "static",
            "boxplots.png"
        )

        plt.savefig(path)

        plt.close()

        chart_paths[
            "boxplot"
        ] = "/data/boxplots.png"

    return chart_paths


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(df):

    cleaned_df = df.copy()

    # Remove duplicate records
    cleaned_df = cleaned_df.drop_duplicates()

    # Fill missing numeric values
    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].median()
        )

    # Fill missing text values
    text_columns = cleaned_df.select_dtypes(
        exclude="number"
    ).columns

    for column in text_columns:

        cleaned_df[column] = cleaned_df[column].fillna(
            "Unknown"
        )

    return cleaned_df


# ============================================================
# MACHINE LEARNING ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):

    numeric_df = df.select_dtypes(
        include="number"
    ).copy()

    if (
        numeric_df.shape[0] < 2
        or
        numeric_df.shape[1] < 1
    ):

        return {
            "anomalies": [],
            "total_anomalies": 0,
            "total_records": len(df)
        }

    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_df
    )

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    predictions = model.fit_predict(
        scaled_data
    )

    scores = model.decision_function(
        scaled_data
    )

    result_df = df.copy()

    result_df[
        "Anomaly_Label"
    ] = predictions

    result_df[
        "Anomaly_Score"
    ] = scores.round(4)

    anomaly_df = result_df[
        result_df[
            "Anomaly_Label"
        ] == -1
    ]

    anomalies = []

    for index, row in anomaly_df.iterrows():

        record = {}

        if "Customer_ID" in row.index:

            record[
                "Customer_ID"
            ] = row[
                "Customer_ID"
            ]

        else:

            record[
                "Row_Index"
            ] = index

        record[
            "Anomaly_Score"
        ] = row[
            "Anomaly_Score"
        ]

        anomalies.append(
            record
        )

    return {
        "anomalies": anomalies,
        "total_anomalies": len(anomalies),
        "total_records": len(df)
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    missing_values,
    duplicate_records,
    invalid_values,
    outlier_count,
    anomaly_count
):

    recommendations = []

    if missing_values > 0:

        recommendations.append(
            "Handle missing values using appropriate imputation techniques."
        )

    if duplicate_records > 0:

        recommendations.append(
            "Remove duplicate records to improve dataset consistency."
        )

    if invalid_values > 0:

        recommendations.append(
            "Review invalid or logically inconsistent values."
        )

    if outlier_count > 0:

        recommendations.append(
            "Investigate statistical outliers before using the dataset for modelling."
        )

    if anomaly_count > 0:

        recommendations.append(
            "Review machine-learning detected anomalies for possible suspicious records."
        )

    if not recommendations:

        recommendations.append(
            "Dataset quality looks good. Continue with further analysis."
        )

    return recommendations


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD AND ANALYSIS
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        return """
        <h2>No file selected.</h2>
        <a href="/">Go Back</a>
        """

    file = request.files["file"]

    if file.filename == "":

        return """
        <h2>No file selected.</h2>
        <a href="/">Go Back</a>
        """

    if not file.filename.lower().endswith(".csv"):

        return """
        <h2>Only CSV files are supported.</h2>
        <a href="/">Go Back</a>
        """

    os.makedirs(
        "data",
        exist_ok=True
    )

    os.makedirs(
        "static",
        exist_ok=True
    )

    filepath = os.path.join(
        "data",
        file.filename
    )

    file.save(filepath)

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            filepath,
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        df = pd.read_csv(
            filepath,
            encoding="latin1"
        )

    # --------------------------------------------------------
    # Clean Dataset
    # --------------------------------------------------------

    cleaned_df = clean_dataset(
        df
    )

    cleaned_file = os.path.join(
        "static",
        "cleaned_dataset.csv"
    )

    cleaned_df.to_csv(
        cleaned_file,
        index=False
    )

    # --------------------------------------------------------
    # Basic Analysis
    # --------------------------------------------------------

    basic_results = basic_forensic_analysis(
        df
    )

    missing_values = basic_results[
        "missing_values"
    ]

    duplicate_records = basic_results[
        "duplicate_records"
    ]

    # --------------------------------------------------------
    # Invalid Values
    # --------------------------------------------------------

    invalid_values = detect_invalid_values(
        df
    )

    # --------------------------------------------------------
    # Outliers
    # --------------------------------------------------------

    outlier_results = detect_outliers(
        df
    )

    outlier_count = outlier_results[
        "total_outliers"
    ]

    # --------------------------------------------------------
    # Forensic Score
    # --------------------------------------------------------

    forensic_score = calculate_forensic_score(
        df,
        missing_values,
        duplicate_records,
        invalid_values,
        outlier_count
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation = correlation_analysis(
        df
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    chart_paths = generate_visualizations(
        df
    )

    # --------------------------------------------------------
    # ML Anomaly Detection
    # --------------------------------------------------------

    anomaly_results = detect_anomalies(
        df
    )

    anomaly_count = anomaly_results[
        "total_anomalies"
    ]

    total_records = anomaly_results[
        "total_records"
    ]

    anomaly_rate = round(
        (
            anomaly_count /
            total_records
        ) * 100,
        2
    ) if total_records > 0 else 0

    # --------------------------------------------------------
    # Sort Top 10 Most Suspicious Records
    # --------------------------------------------------------

    sorted_anomalies = sorted(
        anomaly_results["anomalies"],
        key=lambda x: x["Anomaly_Score"]
    )

    top_anomalies = sorted_anomalies[:10]

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = generate_recommendations(
        missing_values,
        duplicate_records,
        invalid_values,
        outlier_count,
        anomaly_count
    )

    # --------------------------------------------------------
    # Save Result To MySQL
    # --------------------------------------------------------

    try:

        save_forensic_result(
            file.filename,
            basic_results["rows"],
            basic_results["columns"],
            missing_values,
            duplicate_records,
            invalid_values,
            outlier_count,
            forensic_score
        )

        mysql_status = (
            "MYSQL DATABASE CONNECTED SUCCESSFULLY"
        )

    except Exception as e:

        mysql_status = (
            "MYSQL DATABASE ERROR: "
            + str(e)
        )

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    html = """

    <!DOCTYPE html>

    <html>

    <head>

        <title>Forensic Analysis Result</title>

        <style>

            body {
                font-family: Arial;
                background: #0f172a;
                color: white;
                padding: 30px;
            }

            .container {
                max-width: 1200px;
                margin: auto;
            }

            h1 {
                color: #60a5fa;
            }

            .grid {
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(180px, 1fr));

                gap: 20px;

                margin: 25px 0;
            }

            .card {
                background: #1e293b;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
            }

            .card h2 {
                margin: 0;
                color: #60a5fa;
            }

            .section {
                background: #1e293b;
                padding: 25px;
                border-radius: 15px;
                margin-top: 25px;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }

            th,
            td {
                padding: 12px;
                border-bottom: 1px solid #334155;
                text-align: center;
            }

            th {
                background: #334155;
            }

            img {
                max-width: 100%;
                margin-top: 20px;
                border-radius: 10px;
                background: white;
            }

            .button {
                display: inline-block;
                padding: 12px 20px;
                background: #2563eb;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                margin-top: 15px;
            }

            .button:hover {
                background: #1d4ed8;
            }

            li {
                margin: 10px 0;
            }

        </style>

    </head>


    <body>

        <div class="container">

            <h1>
                🔍 Data Forensics AI — Analysis Result
            </h1>

            <p>
                Dataset:
                <strong>
                    """ + str(file.filename) + """
                </strong>
            </p>


            <div class="grid">

                <div class="card">
                    <h2>
                        """ + str(forensic_score) + """
                    </h2>
                    <p>Forensic Score</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(basic_results["rows"]) + """
                    </h2>
                    <p>Records</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(basic_results["columns"]) + """
                    </h2>
                    <p>Columns</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(missing_values) + """
                    </h2>
                    <p>Missing Values</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(duplicate_records) + """
                    </h2>
                    <p>Duplicates</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(invalid_values) + """
                    </h2>
                    <p>Invalid Values</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(outlier_count) + """
                    </h2>
                    <p>Statistical Outliers</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(anomaly_count) + """
                    </h2>
                    <p>ML Anomalies</p>
                </div>

                <div class="card">
                    <h2>
                        """ + str(anomaly_rate) + """%
                    </h2>
                    <p>Anomaly Rate</p>
                </div>

            </div>


            <div class="section">

                <h2>
                    🤖 Top 10 Most Suspicious Records
                </h2>

                <table>

                    <tr>

                        <th>
                            Rank
                        </th>

                        <th>
                            Row / Customer
                        </th>

                        <th>
                            Anomaly Score
                        </th>

                    </tr>
    """

    # --------------------------------------------------------
    # Top 10 Anomalies Table
    # --------------------------------------------------------

    if top_anomalies:

        for rank, anomaly in enumerate(
            top_anomalies,
            start=1
        ):

            if "Customer_ID" in anomaly:

                identifier = anomaly[
                    "Customer_ID"
                ]

            else:

                identifier = anomaly[
                    "Row_Index"
                ]

            html += f"""

                    <tr>

                        <td>
                            {rank}
                        </td>

                        <td>
                            {identifier}
                        </td>

                        <td>
                            {anomaly["Anomaly_Score"]}
                        </td>

                    </tr>

            """

    else:

        html += """

                    <tr>

                        <td colspan="3">
                            No anomalies detected.
                        </td>

                    </tr>

        """

    html += """

                </table>

            </div>


            <div class="section">

                <h2>
                    📊 Outlier Analysis
                </h2>

                <table>

                    <tr>

                        <th>
                            Column
                        </th>

                        <th>
                            Outlier Count
                        </th>

                    </tr>
    """

    # --------------------------------------------------------
    # Outlier Details
    # --------------------------------------------------------

    if outlier_results["details"]:

        for column, count in (
            outlier_results["details"].items()
        ):

            html += f"""

                    <tr>

                        <td>
                            {column}
                        </td>

                        <td>
                            {count}
                        </td>

                    </tr>

            """

    else:

        html += """

                    <tr>

                        <td colspan="2">
                            No statistical outliers detected.
                        </td>

                    </tr>

        """

    html += """

                </table>

            </div>


            <div class="section">

                <h2>
                    💡 Recommendations
                </h2>

                <ul>
    """

    # --------------------------------------------------------
    # Recommendations HTML
    # --------------------------------------------------------

    for recommendation in recommendations:

        html += f"""

                    <li>
                        {recommendation}
                    </li>

        """

    html += """

                </ul>

            </div>


            <div class="section">

                <h2>
                    📈 Visualizations
                </h2>
    """

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    if "missing_values" in chart_paths:

        html += """

                <h3>
                    Missing Values
                </h3>

                <img
                    src="/data/missing_values.png"
                    alt="Missing Values Chart"
                >

        """

    if "correlation" in chart_paths:

        html += """

                <h3>
                    Correlation Heatmap
                </h3>

                <img
                    src="/data/correlation_heatmap.png"
                    alt="Correlation Heatmap"
                >

        """

    if "distribution" in chart_paths:

        html += """

                <h3>
                    Numeric Distributions
                </h3>

                <img
                    src="/data/numeric_distributions.png"
                    alt="Numeric Distributions"
                >

        """

    if "boxplot" in chart_paths:

        html += """

                <h3>
                    Boxplots
                </h3>

                <img
                    src="/data/boxplots.png"
                    alt="Boxplots"
                >

        """

    html += """

            </div>


            <div class="section">

                <h2>
                    🗄️ Database Status
                </h2>

                <p>
    """

    html += mysql_status

    html += """

                </p>

            </div>


            <div class="section">

                <h2>
                    🧹 Cleaned Dataset
                </h2>

                <p>
                    Duplicate records removed and missing values
                    handled using appropriate replacements.
                </p>

                <a
                    href="/download-cleaned"
                    class="button"
                >
                    ⬇ Download Cleaned Dataset
                </a>

            </div>


            <br>

            <a
                href="/"
                class="button"
            >
                ← Analyze Another Dataset
            </a>


        </div>

    </body>

    </html>

    """

    return html


# ============================================================
# DOWNLOAD CLEANED DATASET
# ============================================================

@app.route(
    "/download-cleaned"
)
def download_cleaned():

    cleaned_file = os.path.join(
        "static",
        "cleaned_dataset.csv"
    )

    if not os.path.exists(
        cleaned_file
    ):

        return """
        <h2>Cleaned dataset not available.</h2>
        <a href="/">Go Back</a>
        """

    return send_file(
        cleaned_file,
        as_attachment=True,
        download_name="cleaned_dataset.csv"
    )


# ============================================================
# SERVE CHART FILES
# ============================================================

@app.route(
    "/data/<path:filename>"
)
def serve_data(filename):

    return send_file(
        os.path.join(
            "static",
            filename
        )
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    try:

        connection = get_db_connection()

        print(
            "MYSQL DATABASE CONNECTED SUCCESSFULLY"
        )

        connection.close()

    except Exception as e:

        print(
            "MYSQL DATABASE CONNECTION ERROR:",
            e
        )

    app.run(
        debug=True
    )
    