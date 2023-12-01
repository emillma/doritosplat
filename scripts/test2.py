import plotly.graph_objects as go
import numpy as np
from setup_browser import setup_browser

setup_browser()


x, y = np.mgrid[0:1:200j, 0:1:200j]
z = np.where(x + y < 1, x * y * (1 - x - y), np.nan)


fig = go.Figure(data=[go.Surface(z=z)])

# fig.update_layout(
#     title="Mt Bruno Elevation",
#     autosize=False,
#     width=500,
#     height=500,
#     # margin=dict(l=0, r=1, b=65, t=90),
# )
fig.show()
