import freud
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def compute_voronoi(points, bbox=[-1, 1, -1, 1, -1, 1]):
    Lx = bbox[1] - bbox[0]
    Ly = bbox[3] - bbox[2]
    Lz = bbox[5] - bbox[4]

    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)
    voro = freud.locality.Voronoi()

    # Preserve original indices and select points inside bbox
    orig_indices = np.arange(points.shape[0])
    mask = (
        (points[:, 0] >= bbox[0]) & (points[:, 0] <= bbox[1]) &
        (points[:, 1] >= bbox[2]) & (points[:, 1] <= bbox[3]) &
        (points[:, 2] >= bbox[4]) & (points[:, 2] <= bbox[5])
    )
    inside_indices = orig_indices[mask]
    points_in = points[mask]

    # center points inside bbox around zero
    center = np.array([(bbox[0] + bbox[1]) / 2.0,
                       (bbox[2] + bbox[3]) / 2.0,
                       (bbox[4] + bbox[5]) / 2.0])
    points_in = points_in - center

    # If no points inside bbox, return arrays of NaNs matching original length
    if points_in.size == 0:
        volume_all = np.full(points.shape[0], np.nan, dtype=float)
        normalized_all = np.full(points.shape[0], np.nan, dtype=float)
        return volume_all, normalized_all

    # Compute Voronoi for points inside bbox
    voro.compute((box, points_in))

    # volumes correspond to points_in in the same order; map them back to original indices
    volumes = np.asarray(voro.volumes)
    mean_vol = np.mean(volumes) if volumes.size > 0 else np.nan
    if mean_vol == 0 or np.isnan(mean_vol):
        normalized = np.full_like(volumes, np.nan, dtype=float)
    else:
        normalized = volumes / mean_vol

    volume_all = np.full(points.shape[0], np.nan, dtype=float)
    normalized_all = np.full(points.shape[0], np.nan, dtype=float)
    volume_all[inside_indices] = volumes
    normalized_all[inside_indices] = normalized

    return volume_all, normalized_all


def plot_voronoi(points, bbox=[-1, 1, -1, 1, -1, 1]):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_title("Points")
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_zlim(bbox[3], bbox[4])

    Lx = bbox[1] - bbox[0]
    Ly = bbox[3] - bbox[2]
    Lz = bbox[5] - bbox[4]

    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)
    voro = freud.locality.Voronoi()

    points = points[(points[:, 0] >= bbox[0]) & (points[:, 0] <= bbox[1]) &
                    (points[:, 1] >= bbox[2]) & (points[:, 1] <= bbox[3]) &
                    (points[:, 2] >= bbox[4]) & (points[:, 2] <= bbox[5])]

    center = np.array([(bbox[0] + bbox[1]) / 2.0,
                       (bbox[2] + bbox[3]) / 2.0,
                       (bbox[4] + bbox[5]) / 2.0])
    points = points - center

    cells = voro.compute((box, points)).polytopes

    print(cells)

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")

    # plot seeds
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="k", s=20)

    # draw Voronoi cells as translucent polyhedra
    try:
        have_hull = True
    except Exception:
        have_hull = False

    # `cells` may contain either vertex indices or explicit coordinates.
    for cell in cells:
        if len(cell) == 0:
            continue

        # Resolve to vertex coordinates
        if np.issubdtype(getattr(cell, "dtype", np.array(cell).dtype), np.integer):
            verts = np.asarray(voro.vertices)[cell]
        else:
            verts = np.asarray(cell)

        if verts.shape[0] < 3:
            continue

        faces = []
        if have_hull and verts.shape[0] >= 4:
            try:
                hull = ConvexHull(verts)
                # hull.simplices gives triangular faces for 3D hull
                for simplex in hull.simplices:
                    faces.append(verts[simplex])
            except Exception:
                pass

        if not faces:
            # fallback: try to order vertices in a plane via PCA and make one polygon face
            centroid = verts.mean(axis=0)
            try:
                u, s, vh = np.linalg.svd(verts - centroid)
                v1, v2 = vh[0], vh[1]
                angles = np.arctan2((verts - centroid) @ v2,
                                    (verts - centroid) @ v1)
                ordered = verts[np.argsort(angles)]
                faces.append(ordered)
            except Exception:
                # final fallback: connect all pairs (will be messy but visible)
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        faces.append(np.array([verts[i], verts[j]]))

        # add faces to plot
        color = np.random.rand(3)
        for face in faces:
            if face.shape[0] == 2:
                # plot an edge
                ax.plot(face[:, 0], face[:, 1], face[:, 2],
                        color="k", alpha=0.5, linewidth=0.5)
            else:
                poly = Poly3DCollection([face], alpha=0.15)
                poly.set_facecolor(color)
                poly.set_edgecolor("k")
                ax.add_collection3d(poly)

    # ax.set_xlim(bbox[0], bbox[1])
    # ax.set_ylim(bbox[2], bbox[3])
    # ax.set_zlim(bbox[4], bbox[5])
    ax.set_title("3D Voronoi cells")
    plt.show()

    plt.hist(voro.volumes)
    plt.title("Voronoi cell volumes")
    plt.show()
