"""Code template for exercise 7 - delaunay triangulation"""

import argparse
from collections import defaultdict

import numpy as np
import open3d as o3d


class Sphere:
    def __init__(self, center: np.ndarray, radius: float):
        # center should be a single 3D point
        assert len(center.shape) == 1
        assert center.shape[0] >= 3

        self._center = center[:3]
        self._radius = abs(radius)

        self._mesh = self._create_mesh()

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    @property
    def mesh(self):
        return self._mesh

    def is_point_in_sphere(self, point: np.ndarray):
        assert len(point.shape) == 1
        assert point.shape[0] >= 3

        return np.linalg.norm(point - self.center) <= self.radius

    def _create_mesh(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(self.radius)
        sphere.translate(self.center)
        sphere.compute_vertex_normals()

        lines = []
        for triangle in sphere.triangles:
            triangle = triangle.tolist()
            lines.extend(
                [
                    [triangle[0], triangle[1]],
                    [triangle[1], triangle[2]],
                    [triangle[2], triangle[0]],
                ]
            )

        # Create the line set
        return o3d.geometry.LineSet(
            points=sphere.vertices,
            lines=o3d.utility.Vector2iVector(lines),
        )


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
        # v0_2 = np.linalg.norm(self.vertices[0]) ** 2
        # v1 = self.vertices[1] - self.vertices[0]
        # v2 = self.vertices[2] - self.vertices[0]
        # v3 = self.vertices[3] - self.vertices[0]

        # a = np.array([v1, v2, v3])
        # b = np.array(
        #     [
        #         [(np.linalg.norm(self.vertices[1]) ** 2 - v0_2) / 2],
        #         [(np.linalg.norm(self.vertices[2]) ** 2 - v0_2) / 2],
        #         [(np.linalg.norm(self.vertices[3]) ** 2 - v0_2) / 2],
        #     ]
        # )

        # center = np.linalg.solve(a, b).flatten()
        # radius = np.linalg.norm(self.vertices[0] - center)

        # radius_v0 = np.linalg.norm(self.vertices[0] - center)
        # radius_v1 = np.linalg.norm(self.vertices[1] - center)
        # radius_v2 = np.linalg.norm(self.vertices[2] - center)
        # radius_v3 = np.linalg.norm(self.vertices[3] - center)

        # assert abs(radius_v0 - radius_v1) < 1 and abs(radius_v0 - radius_v2) < 1 and abs(radius_v0 - radius_v3) < 1

        # return Sphere(center, radius)

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
        self._renderer.register_key_callback(85, self._draw)
        self._renderer.register_key_callback(80, self._toggle_pcd)
        self._renderer.register_key_callback(257, self._finalize_iteration)
        self._renderer.create_window()

        # create a pointcloud of all our points we want to add
        self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(self._points)
        self._draw_pcd = True

        # Adding the mesh once sets the camera bb (which is not updated by draw)
        self._renderer.add_geometry(self._super_tetrahedron.mesh)
        self._draw()

        self._renderer.run()

    def __del__(self):
        self._renderer.destroy_window()

    def _next_iteration(self, vis):
        try:
            self._iteration(next(self._point_iter))
        except StopIteration:
            # if no more elements are available, clean up
            self._clean_up()

        self._draw()

    def _finalize_iteration(self, vis):
        for idx in self._point_iter:
            self._iteration(idx)

        self._clean_up()
        self._draw()

    def _toggle_pcd(self, vis):
        self._draw_pcd = not self._draw_pcd
        self._draw(vis)

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
        # Create the encompassing sphere of all points
        center = np.mean(self._points, axis=0)
        radius = np.max(np.linalg.norm(self._points - center, axis=1))

        # place that sphere within a regular tetrahedron
        tetra_edge_length = 12 * radius / np.sqrt(6)
        a = tetra_edge_length / (2 * np.sqrt(2))

        vertices = np.array(
            [
                [a, a, a],
                [a, -a, -a],
                [-a, a, -a],
                [-a, -a, a],
            ]
        )

        vertices += center

        return vertices

    def _clean_up(self):
        # Remove all tetrahedra that are part of the supertetrahedron
        # I.e., no index belongs to the supertetrahedron vertices
        self._triangulation = [tetra for tetra in self._triangulation if not tetra.is_super_tetrahedron]

    def _draw(self, vis=None):
        self._renderer.clear_geometries()

        for tetra in sorted(self._triangulation, key=lambda x: x.is_super_tetrahedron):
            self._renderer.add_geometry(tetra.mesh, reset_bounding_box=False)

        if self._draw_pcd:
            self._renderer.add_geometry(self._pcd, reset_bounding_box=False)

        self._renderer.update_renderer()


def _create_random_points(xdim, ydim, zdim, n):

    return np.vstack(
        (
            np.random.uniform(-xdim, xdim, n),
            np.random.uniform(-ydim, zdim, n),
            np.random.uniform(-zdim, zdim, n),
        )
    ).T


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-points",
        "-n",
        type=int,
        default=50,
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

    args = parser.parse_args()

    points = _create_random_points(
        args.xdim,
        args.ydim,
        args.zdim,
        args.num_points,
    )

    BowyerWatsonVisualiser(points)


if __name__ == "__main__":
    _main()
