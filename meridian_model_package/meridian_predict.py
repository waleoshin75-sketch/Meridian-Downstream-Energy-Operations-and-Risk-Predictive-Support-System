"""Score new Meridian data with the saved models.
Input: a DataFrame that already went through cleaning, feature engineering and one-hot encoding (the same columns as the training
data). Missing columns are added as 0/NaN, extra columns are ignored.
    import meridian_predict as mp
    out = mp.predict(df, objective=3)      # 1 maintenance, 2 margin, 3 late delivery, 4 off-spec, 5 inventory anomalies
"""
import os
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def load(n):
    model = joblib.load(os.path.join(HERE, f"obj{n}_model.joblib"))
    scaler = joblib.load(os.path.join(HERE, f"obj{n}_scaler.joblib"))
    feats = joblib.load(os.path.join(HERE, f"obj{n}_features.joblib"))
    return model, scaler, feats

def prepare(df, n):
    model, scaler, feats = load(n)
    x = df.reindex(columns=feats["feature_names"]).replace([np.inf, -np.inf], np.nan)
    fill = feats["fill_values"]
    x = x.fillna(fill)
    x[feats["scaled_columns"]] = scaler.transform(x[feats["scaled_columns"]])
    return x[feats["model_input_columns"]]

def predict(df, objective):
    n = objective
    model, scaler, feats = load(n)
    reg = joblib.load(os.path.join(HERE, "model_registry.joblib"))["objectives"][str(n)]
    x = prepare(df, n)
    out = pd.DataFrame(index=df.index)
    if reg["task"] == "classification":
        out["probability"] = model.predict_proba(x)[:, 1]
        out["flag"] = (out["probability"] >= reg["decision_cutoff"]).astype(int)
    elif reg["task"] == "regression":
        out["predicted_margin_usd_per_bbl"] = model.predict(x)
    else:
        out["anomaly_score"] = -model.decision_function(x)      # higher = more unusual
        out["flag"] = (model.predict(x) == -1).astype(int)
    return out
