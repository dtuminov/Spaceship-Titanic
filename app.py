from fastapi import FastAPI
from pydantic import BaseModel
import skops.io as sio
import numpy as np
from fastapi.responses import FileResponse, JSONResponse

class Item(BaseModel):
    PassengerId: object
    HomePlanet: object
    CryoSleep: bool
    Cabin: object
    Destination: object
    Age: float
    VIP: bool
    RoomService: float
    FoodCourt: float
    ShoppingMall: float
    Spa: float
    VRDeck: float
    Name: object


app = FastAPI()


def covert_data(arr: list):
    arr[2] = int(arr[2])
    arr[6] = int(arr[6])
    if arr[1] == 'Earth':
        arr[1] = 1
    elif arr[1] == 'Europa':
        arr[1] = 0
    elif arr[1] == 'Mars':
        arr[1] = 2
    if arr[4] == 'TRAPPIST-1e':
        arr[4] = 0
    elif arr[4] == '55 Cancri e':
        arr[4] = 1
    elif arr[4] == 'PSO J318.5-22':
        arr[4] = 2

    str = arr[3]
    if str[0] == 'D':
        arr[3] = 0
    elif str[0] == 'B':
        arr[3] = 1
    elif str[0] == 'C':
        arr[3] = 2
    elif str[0] == 'A':
        arr[3] = 3
    elif str[0] == 'E':
        arr[3] = 4
    elif str[0] == 'F':
        arr[3] = 5
    elif str[0] == 'T':
        arr[3] = 6
    elif str[0] == 'G':
        arr[3] = 7

    arr = np.asarray(arr)
    arr = np.delete(arr, [0, 7, 8, 9, 10, 11])
    arr = arr.tolist()
    int_list = [int(float(i)) for i in arr]
    int_list = np.asarray(int_list)
    int_list = int_list.reshape(1, -1)
    return int_list


@app.post("/predict")
async def prediction(item: Item):
    arr = [item.PassengerId,item.HomePlanet,item.CryoSleep,item.Cabin,
           item.Destination, item.Age, item.VIP, item.RoomService,
           item.FoodCourt, item.ShoppingMall, item.Spa, item.VRDeck]
    model = sio.load('my-model.skops')
    prediction = model.predict(covert_data(arr))
    prediction_proba = model.predict_proba(covert_data(arr))
    prediction = prediction.tolist()
    prediction_proba = prediction_proba.tolist()
    return {'prediction': prediction, 'prediction_proba': prediction_proba}

@app.get("/get_model")
async def download_file():
    return FileResponse(path='Kosmo_titan/spaceship-titanic/ans.csv', media_type='application/octet-stream',
                        filename='Обученная модель.skops')
