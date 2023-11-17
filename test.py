import plotly.graph_objects as go
import numpy as np
from setup_browser import setup_browser
import triangle as tr

setup_browser()
# Download data set from plotly repo
v0 = tr.Vertex(0, 0, 0, r=1.0)
v1 = tr.Vertex(0, 1, 0, g=1.0)
v2 = tr.Vertex(1, 1, 0, b=1.0)
v3 = tr.Vertex(1, 0, 0, a=0.5)
t0 = tr.Triangle(v0, v1, v2)
t1 = tr.Triangle(v0, v2, v3)
data = tr.mesh([t0, t1])


fig = go.Figure(data=[go.Mesh3d(**data)])
fig.show()
# with open("file.html", "w") as f:
#     f.write(fig.to_html())
# fig.write_html("file.html")
