from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Vertex:
    x: float
    y: float
    z: float
    r: float = field(default=0.0)
    g: float = field(default=0.0)
    b: float = field(default=0.0)
    a: float = field(default=1.0)

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    @property
    def color(self):
        return (self.r, self.g, self.b, self.a)

    def translate(self, x, y, z):
        return Vertex(self.x + x, self.y + y, self.z + z, *self.color)

    def recolor(self, r=None, g=None, b=None, a=None):
        r = self.r if r is None else r
        g = self.g if g is None else g
        b = self.b if b is None else b
        a = self.a if a is None else a
        return Vertex(*self.pos, r, g, b, a)


@dataclass(frozen=True)
class Triangle:
    v1: Vertex
    v2: Vertex
    v3: Vertex

    @property
    def vertices(self) -> tuple[Vertex, Vertex, Vertex]:
        return (self.v1, self.v2, self.v3)


def mesh(triangles: list[Triangle]):
    vertices_unique = set([v for t in triangles for v in t.vertices])
    v_map = {v: i for i, v in enumerate(vertices_unique)}
    xyz = np.array([v.pos for v in vertices_unique])
    rgba = np.array([v.color for v in vertices_unique])
    ijk = np.array([[v_map[v] for v in t.vertices] for t in triangles])

    return dict(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        vertexcolor=rgba,
        i=ijk[:, 0],
        j=ijk[:, 1],
        k=ijk[:, 2],
    )
