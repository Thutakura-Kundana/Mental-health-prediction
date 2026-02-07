from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
import numpy as np
import pandas as pd
import os
import json
import plotly
import plotly.express as px
from model import dataset_handling
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from utils.recommender import get_recommendations
from datetime import datetime

app = Flask(__name__)
d = dataset_handling()

# 📊 Load and preprocess dataset
data = pd.read_csv("dataframe_hrv.csv")
data = data.drop(['marker', 'SDNN', 'TP', 'ULF', 'VLF', 'LF', 'HF', 'LF_HF'], axis=1, errors='ignore')
df = pd.DataFrame(data).dropna(how='any')
df.stress = df.stress.astype(int)

X = df.drop(['stress'], axis=1)
y = df['stress']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# 🤖 Train ensemble model
clf1 = SVC(gamma='auto')
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = DecisionTreeClassifier()
eclf1 = VotingClassifier(estimators=[('svc', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')
eclf1.fit(X_train, y_train)

# Define mapping for stress levels
stress_map = {0: 'Low', 1: 'Medium', 2: 'High'}


# 📈 Utility functions
def dataset():
    dataframe_hrv = d.creating_dataframe()
    dataframe_hrv = d.fix_stress_labels(df=dataframe_hrv)
    dataframe_hrv = d.missing_values(dataframe_hrv)
    selected_x_columns, exported_pipeline = d.train_and_test(dataframe_hrv)
    return selected_x_columns, exported_pipeline


def plot():
    selected_x_columns, exported_pipeline = dataset()
    fig = []
    for file in os.listdir(path="datasets"):
        input_df = pd.read_csv("datasets/" + file)
        figure = d.plotFitBitReading(input_df, exported_pipeline, selected_x_columns)
        fig.append(figure)
    figureJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    ids = ['figure-{}'.format(i) for i, _ in enumerate(fig)]
    return figureJSON, ids


# 🌐 Routes
@app.route('/index')
def index():
    figureJSON, ids = plot()
    return render_template('index.html', ids=ids, figuresJSON=figureJSON)


@app.route('/')
@app.route('/index1')
def index1():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle stress prediction and log the result."""
    # Step 1: Read user input
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = eclf1.predict(final)[0]

    # Step 2: Convert numeric prediction to label
    stress_level = stress_map.get(prediction, "Low")

    # Step 3: Get recommendations
    suggestions = get_recommendations(stress_level)

    # Step 4: Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Step 5: Log to CSV (always with headers)
    os.makedirs('logs', exist_ok=True)
    log_file = 'logs/prediction_log.csv'
    file_exists = os.path.isfile(log_file)

    log_entry = {
        'timestamp': timestamp,
        'features': json.dumps(int_features),
        'prediction': stress_level,
        'suggestions': json.dumps(suggestions)
    }

    log_df = pd.DataFrame([log_entry])
    # Write header only once
    log_df.to_csv(log_file, mode='a', header=not file_exists or os.stat(log_file).st_size == 0, index=False)

    # Step 6: Display result page
    return render_template(
        'predict.html',
        prediction=stress_level,
        suggestions=suggestions,
        timestamp=timestamp
    )


@app.route('/recommend', methods=['POST'])
def recommend():
    """API route for real-time recommendations."""
    data = request.get_json()
    stress_level_num = int(data.get('stress_level', 0))
    confidence = float(data.get('confidence', 1.0))

    stress_level_label = stress_map.get(stress_level_num, 'Low')
    suggestions = get_recommendations(stress_level_label, confidence)

    return jsonify({'recommendations': suggestions})


@app.route('/dashboard')
def dashboard():
    """Visualize stress levels over time."""
    try:
        log_file = 'logs/prediction_log.csv'
        if not os.path.exists(log_file):
            return "<h3>No data available yet. Please make some predictions first.</h3>"

        df = pd.read_csv(log_file)

        # If headers are missing, fix automatically
        expected_cols = ['timestamp', 'features', 'prediction', 'suggestions']
        if len(df.columns) == 4 and df.columns[0].startswith('202'):
            df = pd.read_csv(log_file, header=None)
            df.columns = expected_cols

        if not all(col in df.columns for col in expected_cols):
            return "<h3>Error: Unexpected CSV format</h3>"

        # Convert timestamp safely
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # Convert text labels to numbers for plotting
        level_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['stress_level'] = df['prediction'].map(level_map)

        # Create interactive line chart
        fig = px.line(
            df,
            x='timestamp',
            y='stress_level',
            title='Stress Level Trend Over Time',
            labels={'timestamp': 'Time', 'stress_level': 'Stress Level'},
            markers=True
        )

        # Show readable y-axis labels
        fig.update_yaxes(
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        )

        # Convert chart to HTML
        graph_html = fig.to_html(full_html=False)
        return render_template('dashboard.html', graph=Markup(graph_html))

    except Exception as e:
        return f"<h3>Error loading dashboard: {e}</h3>"


if __name__ == '__main__':
    app.run(debug=True)

