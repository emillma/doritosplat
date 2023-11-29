import asyncio
import numpy as np
from pathlib import Path
from PIL import Image
from websockets.server import serve, WebSocketServerProtocol
import cv2

img_path = Path("cat.jpg")
img_array = np.array(Image.open(img_path))[::2,::2,::-1]



cat_bytes = Path("cat.jpg").read_bytes()
connection = 0
disconnected = 0
async def echo(websocket: WebSocketServerProtocol, path: str):
    global connection, disconnected
    try:
        connection +=  1
        print("Connected")
        for i in range(10000):
            print(f"{connection} {disconnected}")
            img_rolled = np.roll(img_array, 5*i, 0)
            success, a_numpy = cv2.imencode('.jpg', img_rolled)
            await websocket.send(a_numpy.tobytes())
            await asyncio.sleep(.02)
    except Exception as e:
        disconnected += 1
        print("Exception")
        print(type(e), e)
    # async for message in websocket:
    #     await websocket.send(message)

async def main():
    async with serve(echo, "localhost", 5170):
        await asyncio.Future()  # run forever

asyncio.run(main())