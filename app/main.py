import io

import mlflow
import pandas as pd
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi_mlflow.predictors import build_predictor

# Create FastAPI instance
app = FastAPI()


remote_server_uri = "http://localhost:5001"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
model_name = "cbnGDP"
model_version = 1

model = mlflow.statsmodels.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
endog = pd.read_csv("endog.csv")
endog = endog.set_index("date")


@app.post("/predict")
async def predict(file: bytes = File(...)):
    print("[+] Initiate Prediction")
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    test_df = test_df.set_index("date")
    print(test_df)

    preds = model.forecast(endog.values[-1:], steps=4, exog_future=test_df)
    preds = pd.DataFrame(
        data=preds,
        columns=["pc1", "pc2", "pc3", "pc4", "pc5", "dlRY"],
        index=test_df.index,
    )
    preds = preds["dlRY"]

    json_compatible_item_data = jsonable_encoder(preds)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/")
async def main():
    content = """
    <body>
    <form action="/predict/" enctype="multipart/form-data" method="post">
    <input name="file" type="file" multiple>
    <input type="submit">
    </form>
    </body>
     """
    return HTMLResponse(content=content)
