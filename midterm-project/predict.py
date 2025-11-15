import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import pandas as pd

app = FastAPI(title="prediction model")

#with open('pipeline_v1.bin', 'rb') as f_in:
#    pipeline = pickle.load(f_in)

with open('score_predict_model_lr.pkl', 'rb') as f:
    dv, modelm, modelr, modelw = pickle.load(f)


def predict_single(student):
    student_df = pd.DataFrame([student])
    df_dict = student_df.to_dict(orient='records')
    X = dv.transform(df_dict)

    y_pred_m = float(modelm.predict(X)[0]) 
    y_pred_r = float(modelr.predict(X)[0]) 
    y_pred_w = float(modelw.predict(X)[0]) 

    return [y_pred_m, y_pred_r, y_pred_w]


@app.post("/predict")
def predict(student: Dict[str, Any]):
    results = predict_single(student)

    return {
        "math score": results[0],
        "reading score": results[1],
        "writing score": results[2]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
