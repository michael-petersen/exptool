"""

24 Sep 2023  Introduction

# Create an instance of the Icosphere class with 3 subdivisions
icosphere = Icosphere(subdivisions=3)

# Access the generated points
points = icosphere.points

# Use the points as needed
print(points)


# Create a 3D plot to visualize the evenly spaced points on the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)

# Set equal aspect ratio to show a perfect sphere
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.title('Evenly Spaced Points on a Sphere')
plt.show()


"""

import numpy as np

class Icosphere:
    def __init__(self, subdivisions):
        self.subdivisions = subdivisions
        self.phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

        # Define the vertices of an icosahedron
        self.vertices = np.array([
            [-1, 0, self.phi],
            [1, 0, self.phi],
            [-1, 0, -self.phi],
            [1, 0, -self.phi],
            [0, self.phi, 1],
            [0, self.phi, -1],
            [0, -self.phi, 1],
            [0, -self.phi, -1],
            [self.phi, 1, 0],
            [self.phi, -1, 0],
            [-self.phi, 1, 0],
            [-self.phi, -1, 0]
        ])

        # Define the faces of the icosahedron
        self.faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ], dtype=int)

        # Initialize the list of points
        self.points = []

        # Divide each triangle face into smaller triangles
        for face in self.faces:
            a, b, c = self.vertices[face]
            self._divide_triangle(a, b, c, self.subdivisions)

        # Convert points to a NumPy array
        self.points = np.array(self.points)

        # verify that no anomalous points have been found
        radii = np.linalg.norm(self.points,axis=1)
        self.points = self.points[radii<=1.0]

    def _normalize(self, v):
        """Normalize a vector to have unit length."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _divide_triangle(self, a, b, c, depth):
        """Recursively divide a triangle into smaller triangles."""
        if depth == 0:
            self.points.extend([a, b, c])
        else:
            ab = self._normalize((a + b) / 2)
            ac = self._normalize((a + c) / 2)
            bc = self._normalize((b + c) / 2)
            self._divide_triangle(a, ab, ac, depth - 1)
            self._divide_triangle(ab, b, bc, depth - 1)
            self._divide_triangle(ac, bc, c, depth - 1)
            self._divide_triangle(ab, bc, ac, depth - 1)
