import plotly.graph_objects as go
import numpy as np
from setup_browser import setup_browser
import triangle as tr

setup_browser()
# Download data set from plotly repo

n = 7
t = np.linspace(0, 2 * np.pi, n, endpoint=False)
xs, ys = np.cos(t), np.sin(t)
v0 = tr.Vertex(0, 0, 1, r=1.0, a=1.0)
v_outer = [tr.Vertex(x, y, 0, r=1.0, a=0.0) for x, y in zip(xs, ys)]


triangles = [tr.Triangle(v0, v_outer[i], v_outer[(i + 1) % n]) for i in range(0, n, 1)]

# v1 = tr.Vertex(0, 1, 0, g=1.0)
# v2 = tr.Vertex(1, 1, 0, b=1.0)


# combinations = [[v0, v1, v2], [v1, v2, v0], [v2, v0, v1]]
# vertex_inner = [v0, v1, v2]
# vertex_outer = [
#     va.move_towards((vb, 1.0), (vc, 1), a=0.0) for va, vb, vc in combinations
# ]

# triangles = [tr.Triangle(v0, v1, v2)]
# # triangles = []
# for i in range(3):
#     va, vb, vc = [vertex_inner[k % 3] for k in range(i, i + 3)]
#     vd, ve, vf = [vertex_outer[k % 3] for k in range(i, i + 3)]
#     triangles.append(tr.Triangle(va, vf, vb))
#     # triangles.append(tr.Triangle(va, vc, ve))


data = tr.mesh(triangles)


fig = go.Figure(
    data=[
        go.Mesh3d(
            **data,
            contour=dict(show=False),
            lighting=dict(ambient=1.0, fresnel=0.0, specular=0.0, diffuse=0.5),
            flatshading=False,
        )
    ]
)
# fig.update_traces(
#     selector=dict(type="mesh3d"),
# )
fig.show()
# with open("file.html", "w") as f:
#     f.write(fig.to_html())
# fig.write_html("file.html")
