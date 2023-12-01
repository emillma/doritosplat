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
        return Vertex(*self.pos, r, g, b, a)

    def realpha(self, alpha):
        return Vertex(*self.pos, self.r, self.g, self.b, alpha)

    def move_towards(
        self, *moves: tuple["Vertex", float], r=None, g=None, b=None, a=None
    ):
        """Move towards a vertex by a given distance."""
        move = np.array([0.0, 0.0, 0.0])
        pos = np.array(self.pos)
        for v, d in moves:
            move += (np.array(v.pos) - pos) * d
        r = r if r is not None else self.r
        g = g if g is not None else self.g
        b = b if b is not None else self.b
        a = a if a is not None else self.a
        return Vertex(*(pos + move), r, g, b, a)


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
