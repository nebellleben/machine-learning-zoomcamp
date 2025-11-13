import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

app = FastAPI(title="prediction model")

#with open('pipeline_v1.bin', 'rb') as f_in:
#    pipeline = pickle.load(f_in)

with open('score_predict_model.pkl', 'rb') as f:
    loaded_models = pickle.load(f)


def predict_single(student):
    result = [loaded_models[0].predict(student), loaded_models[1].predict(student), loaded_models[2].predict(student)]
    return result


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



#with open('score_predict_model.pkl', 'rb') as f:
#    loaded_models = pickle.load(f)


#loaded_models[0].predict(X_test)...
#loaded_models[1].predict(X_test)...
#loaded_models[2].predict(X_test)...