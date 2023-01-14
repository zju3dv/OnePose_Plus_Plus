import numpy as np

# ----- Similarity-Affinity-Perspective (SAP) impl ----- #

def similarity_mat(angle, tx, ty, s):
    theta = np.deg2rad(angle)
    return np.array([[s*np.cos(theta), -s*np.sin(theta), tx], [s*np.sin(theta), s*np.cos(theta), ty], [0, 0, 1]])


def affinity_mat(k0, k1):
    return np.array([[k0, k1, 0], [0, 1/k0, 0], [0, 0, 1]])


def perspective_mat(v0, v1):
    return np.array([[1, 0, 0], [0, 1, 0], [v0, v1, 1]])


def compute_homography_sap(h, w, angle=0, tx=0, ty=0, scale=1, k0=1, k1=0, v0=0, v1=0):
    """
    Args:
        img_size: (h, w)
        angle: in degree, goes clock-wise in image-coordinate-system
        tx, ty: displacement
        scale: factor to zoom in, by default 1
        k0: non-isotropic squeeze factor - 1 +(stretch x, squeeze y) [0.5, 1.5]
        k1: non-isotropic skew factor, - 0 +(up-to-left, down-to-right) [-0.5, 0.5]
        v0: left-right perspective factor, - 0 +(move left) [-1, 1]
        v1: up-down perspective factor, - 0 +(move up)  [-1, 1]
    """
    # move image to its center
    max_size = max(w/2, h/2)
    M_norm = similarity_mat(0, 0, 0, 1/max_size).dot(similarity_mat(0, -w/2, -h/2, 1))
    M_denorm = similarity_mat(0, w/2, h/2, 1).dot(similarity_mat(0, 0, 0, max_size))

    # compute HS, HA and HP accordingly
    HS = similarity_mat(angle, tx, ty, scale)
    HA = affinity_mat(k0, k1)
    HP = perspective_mat(v0, v1)

    # final H
    H = M_denorm.dot(HS).dot(HA).dot(HP).dot(M_norm)
    return H


def sample_homography_sap(h, w, **kwargs):
    angle = np.random.uniform(-180, 180)
    tx = np.random.uniform(-0.25, 0.25)
    ty = np.random.uniform(-0.25, 0.25)
    scale = np.random.uniform(0.25, 1)

    k0 = 1  # similar effects as the ratio of xy-focal lengths
    # k1 = 0
    k1 = np.random.uniform(-0.1, 0.1)

    v0 = np.random.uniform(-0.5, 0.5)
    v1 = np.random.uniform(-0.5, 0.5)

    H = compute_homography_sap(h, w, angle, tx, ty, scale, k0, k1, v0, v1)
    return H
