import json
import numpy as np
import os
import joblib


def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "best_random_forest_model_hyperdrive.pkl"
    )
    model = joblib.load(model_path)


def run(data):
    try:
        data = np.array(json.loads(data)["data"])
        result = model.predict(data)
        return {"results": result.tolist()}
    except Exception as e:
        error = str(e)
        return error
