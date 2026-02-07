# utils/logger.py

import csv
from datetime import datetime

def log_prediction(user_id, features, prediction, suggestions):
    """
    Logs prediction results to logs/prediction_log.csv

    Parameters:
    - user_id (str): Identifier for the user (can be 'guest')
    - features (list): Input features used for prediction
    - prediction (int): Predicted stress level
    - suggestions (list): Recommended actions
    """
    with open('logs/prediction_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            user_id,
            features,
            prediction,
            "; ".join(suggestions)
        ])

        writer.writerow([
    datetime.now().isoformat(),
    user_id,
    str(features),  # logs as one string
    prediction,
    "; ".join(suggestions)
])