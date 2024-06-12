import numpy as np

import argparse
from collections import defaultdict

import open3d as o3d


class AABB:
    def __init__(self, points: np.ndarray):
        self._min = points.min(axis=0)
        self._max = points.max(axis=0)

        self._center = (self.min + self.max) / 2
        self._size = np.linalg.norm(self.max - self.min)

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size


class Sphere:
    def __init__(self, center: np.ndarray, radius: float):
        # center should be a single 3D point
        assert len(center.shape) == 1
        assert center.shape[0] >= 3

        self._center = center[:3]
        self._radius = abs(radius)

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    def is_point_in_sphere(self, point: np.ndarray):
        assert len(point.shape) == 1
        assert point.shape[0] >= 3

        return np.linalg.norm(point - self.center) <= self.radius


class Tetrahedron:
    def __init__(self, vertices: np.ndarray, indices: np.ndarray) -> None:
        # we need a list of vertices
        assert len(vertices.shape) == 2
        # each vertex must be a 3d coord
        assert vertices.shape[1] == 3
        # we need at least 4
        assert len(indices) > 3

        self._global_vertices = vertices
        self._indices = indices

        self._ensure_tetrahedron()

        self._faces = self._calculate_faces()
        self._circumsphere = self._calculate_circumsphere()

    @property
    def vertices(self):
        return self._global_vertices[self.indices]

    @property
    def indices(self):
        return self._indices

    @property
    def faces(self):
        return self._faces

    def is_point_in_circumsphere(self, point: np.ndarray):
        return self._circumsphere.is_point_in_sphere(point)

    def _ensure_tetrahedron(self):
        v1 = self.vertices[1] - self.vertices[0]
        v1 /= np.linalg.norm(v1)
        v2 = self.vertices[2] - self.vertices[0]
        v2 /= np.linalg.norm(v2)
        v3 = self.vertices[3] - self.vertices[0]
        v3 /= np.linalg.norm(v3)

        normal = np.cross(v1, v2)
        angle = np.dot(normal, v3)

        assert angle != 0

    def _calculate_circumsphere(self):

        v1 = self.vertices[1] - self.vertices[0]
        v2 = self.vertices[2] - self.vertices[0]
        v3 = self.vertices[3] - self.vertices[0]

        m = np.zeros((3, 3))
        m[0, :] = v1
        m[1, :] = v2
        m[2, :] = v3

        a = np.inner(v1, v1) * np.cross(v2, v3)
        b = np.inner(v2, v2) * np.cross(v3, v1)
        c = np.inner(v3, v3) * np.cross(v1, v2)

        origin = self.vertices[0] + (a + b + c) / (np.linalg.det(m) * 2)
        radius = np.linalg.norm(origin - v1)

        return Sphere(origin, radius)

    def _calculate_faces(self):
        return np.array(
            [
                # If we make sure that our indices are always sorted,
                # it becomes easy to compare faces later on
                np.sort([self.indices[0], self.indices[1], self.indices[2]]),
                np.sort([self.indices[1], self.indices[3], self.indices[2]]),
                np.sort([self.indices[3], self.indices[0], self.indices[2]]),
                np.sort([self.indices[0], self.indices[1], self.indices[3]]),
            ]
        )


def bowyer_watson(points: np.ndarray):
    vertices = np.zeros((4 + len(points), 3))
    vertices[:4] = _create_super_tetrahedron_points(points)
    super_tetrahedron = Tetrahedron(vertices, np.arange(4))

    triangulation = [super_tetrahedron]

    for point_idx, point in enumerate(points):
        # Find the tetrahedra which violate the triangulation if the point is inserted
        invalid_tetrahedra = [tetra for tetra in triangulation if tetra.is_point_in_circumsphere(point)]

        # find the face of the bad tetrahedra - these are the hull of the hole we are going to make
        # We count how often we encounter each face
        face_counts = defaultdict(int)
        for tetra in invalid_tetrahedra:
            for face in tetra.faces:
                face_counts[tuple(face.flatten())] += 1

        # only keep the faces that are not multiple times in the dict
        hull_faces = [np.array(face).reshape(-1, 3) for face, count in face_counts.items() if count < 2]

        # remove invalid tetrahedra from the triangulation
        for tetra in invalid_tetrahedra:
            triangulation.remove(tetra)

        # add the new vertex to our list of vertices
        vertices[4 + point_idx] = point

        # Triangulate the cavity
        for face in hull_faces:
            indices = np.zeros(4, dtype=int)
            indices[1:4] = face
            indices[0] = point_idx + 4
            triangulation.append(Tetrahedron(vertices, indices))

    # Remove all tetrahedra that are part of the supertetrahedron
    # I.e., no index belongs to the supertetrahedron vertices
    final_tetrahedra = [tetra for tetra in triangulation if np.all(tetra.indices > 3)]

    return final_tetrahedra


def _create_super_tetrahedron_points(points: np.ndarray):
    aabb = AABB(points)

    vertices = np.array(
        [
            np.array([2 * aabb.size, 0, -aabb.size]),
            np.array([-aabb.size, 2 * aabb.size, -aabb.size]),
            np.array([-aabb.size, -aabb.size, 2 * aabb.size]),
            np.array([-aabb.size, -aabb.size, -aabb.size]),
        ]
    )

    vertices += aabb.center

    return vertices


def _create_random_points(xdim, ydim, zdim, n):

    return np.vstack(
        (
            np.random.uniform(-xdim, xdim, n),
            np.random.uniform(-ydim, zdim, n),
            np.random.uniform(-zdim, zdim, n),
        )
    ).T


def _create_mesh_points(xdim, ydim, zdim, n):

    x, y, z = np.meshgrid(
        np.linspace(-xdim, xdim, n),
        np.linspace(-ydim, ydim, n),
        np.linspace(-zdim, zdim, n),
    )

    return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T


def _create_tetrahedron_mesh(vertices):
    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(
        [
            [0, 1, 2],
            [1, 3, 2],
            [3, 0, 2],
            [0, 1, 3],
        ]
    )

    mesh.compute_vertex_normals()

    return mesh


def _create_tetrahedron_wireframe(tetra: Tetrahedron):

    edges = []
    for face in tetra.faces:
        edges.append((face[0], face[1]))
        edges.append((face[1], face[2]))
        edges.append((face[2], face[0]))

    # remove duplicates
    edges = list(set(tuple(sorted(edge)) for edge in edges))

    lines = [[tetra.indices.tolist().index(edge[0]), tetra.indices.tolist().index(edge[1])] for edge in edges]

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(tetra.vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])

    return lineset


def key_callback(vis):
    print("Key pressed")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-points",
        "-n",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--xdim",
        "-x",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--ydim",
        "-y",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--zdim",
        "-z",
        type=float,
        default=100,
    )

    parser.add_argument(
        "--grid",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    points = (
        _create_mesh_points(
            args.xdim,
            args.ydim,
            args.zdim,
            args.num_points,
        )
        if args.grid
        else _create_random_points(
            args.xdim,
            args.ydim,
            args.zdim,
            args.num_points,
        )
    )

    triangulation = bowyer_watson(points)

    print(f"Found: {len(triangulation)} tetrahedrons!")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # vis.register_key_callback(32, key_callback)

    for tetra in triangulation:
        # vis.add_geometry(_create_tetrahedron_mesh(tetra.vertices))
        vis.add_geometry(_create_tetrahedron_wireframe(tetra))

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    _main()
