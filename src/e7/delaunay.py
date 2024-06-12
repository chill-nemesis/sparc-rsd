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
        self._mesh = self._create_mesh()

    @property
    def vertices(self):
        return self._global_vertices[self.indices]

    @property
    def indices(self):
        return self._indices

    @property
    def faces(self):
        return self._faces

    @property
    def mesh(self):
        return self._mesh

    @property
    def is_super_tetrahedron(self):
        return np.any(self.indices < 4)

    def is_point_in_circumsphere(self, point: np.ndarray):
        return self._circumsphere.is_point_in_sphere(point)

    def _create_mesh(self):
        edges = []

        for face in self.faces:
            edges.append((face[0], face[1]))
            edges.append((face[1], face[2]))
            edges.append((face[2], face[0]))

        # remove duplicates
        edges = list(set(tuple(sorted(edge)) for edge in edges))

        lines = [[self.indices.tolist().index(edge[0]), self.indices.tolist().index(edge[1])] for edge in edges]

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self.vertices)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        if self.is_super_tetrahedron:
            lineset.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines))])
        else:
            lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])

        return lineset

    def _ensure_tetrahedron(self):
        if (
            np.array_equal(self.vertices[0], self.vertices[1])
            or np.array_equal(self.vertices[0], self.vertices[2])
            or np.array_equal(self.vertices[0], self.vertices[3])
        ):
            raise ValueError()

    def _calculate_circumsphere(self):
        v0_2 = np.linalg.norm(self.vertices[0]) ** 2
        v1 = self.vertices[1] - self.vertices[0]
        v2 = self.vertices[2] - self.vertices[0]
        v3 = self.vertices[3] - self.vertices[0]

        a = np.array([v1, v2, v3])
        b = np.array(
            [
                [(np.linalg.norm(self.vertices[1]) ** 2 - v0_2) / 2],
                [(np.linalg.norm(self.vertices[2]) ** 2 - v0_2) / 2],
                [(np.linalg.norm(self.vertices[3]) ** 2 - v0_2) / 2],
            ]
        )

        center = np.linalg.solve(a, b).flatten()
        radius = np.linalg.norm(self.vertices[0] - center)

        radius_v0 = np.linalg.norm(self.vertices[0] - center)
        radius_v1 = np.linalg.norm(self.vertices[1] - center)
        radius_v2 = np.linalg.norm(self.vertices[2] - center)
        radius_v3 = np.linalg.norm(self.vertices[3] - center)

        assert abs(radius_v0 - radius_v1) < 1 and abs(radius_v0 - radius_v2) < 1 and abs(radius_v0 - radius_v3) < 1

        return Sphere(center, radius)

        # Alternative method
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

        center = self.vertices[0] + (a + b + c) / (np.linalg.det(m) * 2)
        radius = np.linalg.norm(self.vertices[0] - center)

        return Sphere(center, radius)

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


class BowyerWatsonVisualiser:
    def __init__(self, points: np.ndarray) -> None:
        self._points = points
        self._point_iter = iter(range(len(self._points)))

        self._vertices = np.zeros((4 + len(points), 3))
        self._vertices[:4] = self._create_super_tetrahedron_points()
        self._super_tetrahedron = Tetrahedron(self._vertices, np.arange(4))

        self._triangulation = [self._super_tetrahedron]

        # setup visualization
        self._renderer = o3d.visualization.VisualizerWithKeyCallback()
        self._renderer.register_key_callback(32, self._next_iteration)
        self._renderer.create_window()

        # Adding the mesh once sets the camera bb (which is not updated by draw)
        self._renderer.add_geometry(self._super_tetrahedron.mesh)
        self._draw()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._points)
        self._renderer.add_geometry(pcd)

        self._renderer.run()

    def __del__(self):
        self._renderer.destroy_window()

    def _next_iteration(self, vis):
        try:
            self._iteration(next(iter(self._point_iter)))
        except StopIteration:
            # if no more elements are available, clean up
            self._clean_up()

        self._draw()

    def _iteration(self, point_idx):
        point = self._points[point_idx]

        # Find the tetrahedra which violate the triangulation if the point is inserted
        invalid_tetrahedra = [tetra for tetra in self._triangulation if tetra.is_point_in_circumsphere(point)]

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
            self._triangulation.remove(tetra)

        # add the new vertex to our list of vertices
        self._vertices[4 + point_idx] = point

        # Triangulate the cavity
        for face in hull_faces:
            indices = np.zeros(4, dtype=int)
            indices[1:4] = face
            indices[0] = point_idx + 4
            self._triangulation.append(Tetrahedron(self._vertices, indices))

    def _create_super_tetrahedron_points(self):
        aabb = AABB(self._points)

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

    def _clean_up(self):
        # Remove all tetrahedra that are part of the supertetrahedron
        # I.e., no index belongs to the supertetrahedron vertices
        self._triangulation = [tetra for tetra in self._triangulation if not tetra.is_super_tetrahedron]

    def _draw(self):
        self._renderer.clear_geometries()

        for tetra in sorted(self._triangulation, key=lambda x: x.is_super_tetrahedron):
            self._renderer.add_geometry(tetra.mesh, reset_bounding_box=False)

            # sphere = o3d.geometry.TriangleMesh.create_sphere(tetra._circumsphere.radius)
            # sphere.translate(tetra._circumsphere.center)
            # sphere.compute_vertex_normals()

            # lines = []
            # for triangle in sphere.triangles:
            #     triangle = triangle.tolist()
            #     lines.extend(
            #         [
            #             [triangle[0], triangle[1]],
            #             [triangle[1], triangle[2]],
            #             [triangle[2], triangle[0]],
            #         ]
            #     )

            # # Create the line set
            # line_set = o3d.geometry.LineSet(
            #     points=sphere.vertices,
            #     lines=o3d.utility.Vector2iVector(lines),
            # )

            # self._renderer.add_geometry(line_set, reset_bounding_box=False)

        self._renderer.update_renderer()


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


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-points",
        "-n",
        type=int,
        default=5,
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

    np.random.seed(0)

    BowyerWatsonVisualiser(points)


if __name__ == "__main__":
    _main()
