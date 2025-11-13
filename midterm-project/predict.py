import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import pandas as pd

app = FastAPI(title="prediction model")

#with open('pipeline_v1.bin', 'rb') as f_in:
#    pipeline = pickle.load(f_in)

with open('score_predict_model.pkl', 'rb') as f:
    dv, models = pickle.load(f)


def predict_single(student):

    #df = pd.DataFrame(student)
    #df_dict = df.to_dict(orient='records')
    df_dict = student.to_dict(orient='records')
    X = dv.transform(df_dict)

    #result = [loaded_models[0].predict(student), loaded_models[1].predict(student), loaded_models[2].predict(student)]
    result = models[0].predict(X) # need to predict all actually
    return result


@app.post("/predict")
def predict(student: Dict[str, Any]):
    results = predict_single(student)

    return {
        "math score": results
#        "math score": results[0],
#        "reading score": results[1],
#        "writing score": results[2]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)



#with open('score_predict_model.pkl', 'rb') as f:
#    loaded_models = pickle.load(f)


#loaded_models[0].predict(X_test)...
#loaded_models[1].predict(X_test)...
#loaded_models[2].predict(X_test)...