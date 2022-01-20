# This module provides an implementation of the heap queue algorithm, also known as the priority queue algorithm.
import heapq
import math
import os

import numpy as np
import scipy.sparse as sparse
from opendr.topology import get_vert_connectivity, get_vertices_per_edge
from psbody.mesh import Mesh


def vertex_quadrics(mesh: Mesh):
    """
    Computes a quadric for each vertex in the Mesh.
    See: https://users.csc.calpoly.edu/~zwood/teaching/csc570/final06/jseeba/

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.

    :param mesh: The mesh for which the quadrics should be computed.
    :return: The quadrics for each vertex of the given mesh

    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.v), 4, 4,))

    # Each vertex is the solution of the intersection of a set of planes.
    # Namely, the planes of the triangles/faces that meet at that vertex.
    # Set of planes can be associated with each vertex.
    # Error of vertex with respect to this set can be described as the sum of squared
    # distances to its planes.

    # Plane equation: ax + by + cz + d = 0, a^2+b^2+c^2 = 1

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        #  Fitting plane from 3d points
        # -> https://stackoverflow.com/questions/53591350/plane-fit-of-3d-points-with-singular-value-decomposition
        # -> https://en.wikipedia.org/wiki/Singular_value_decomposition#Applications_Of_The_Svd
        # -> https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
        u, s, v = np.linalg.svd(verts)
        # plane equation
        eq = v[-1, :].reshape(-1, 1)
        # normalize
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def setup_deformation_transfer(source, target):
    """
    Returns the upsampling transofrmation given the source and target mesh.
    Source mesh is less downsampled than target mesh. So that there are vertices in the source mesh which do not have
    corresponding vertices in the target mesh.

    All vertices in the target mesh are projected into the source mesh,
    to find the closest/nearest faces, parts, vertices

    :param source: The source mesh (larger)
    :param target: The target, downsampled mesh.
    """

    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])
    coeffs_n = np.zeros(3 * target.v.shape[0])

    # Search the nearest faces, parts and vertices by looking at aabb_tree. Thanks psbody library
    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree().nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.v.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        # nearest_vertices is a flattened array, that's why there is funny indexing.
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.v[nearest_f[n_id - 1]], source.v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.v[i])[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    matrix = sparse.csc_matrix((coeffs_v, (rows, cols)), shape=(target.v.shape[0], source.v.shape[0]))
    return matrix


def qslim_decimator_transformer(mesh: Mesh, factor=None, n_verts_desired=None):
    """
    Performs mesh simplification.

    A Mesh is an object consisting of vertices and faces:
    Vertices v -> Vx3 array of vertices
    Faces f -> Fx3 array of faces



    Uses a qslime-style approach. (Which is essentially vertex-pair contraction)
    (QSlim: https://www.cs.cmu.edu/~garland/quadrics/qslim.html)
    Uses quadratic error matrices

    :param factor: fraction of the original number of vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new faces: An Fx3 array of faces, mtx: the transformation matrix
    """
    if factor is None and n_verts_desired is None:
        raise Exception("Either the factor or the number of desired vertices to retain must be given.")

    if n_verts_desired is None:
        # Infer the number of vertices that should remain
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    # Get quadric matrix for mesh
    Qv = vertex_quadrics(mesh)

    # Get array of edges, indicates vertex-vertex adjacency nx2, n number of edges

    adj_verts = get_vertices_per_edge(mesh.v, mesh.f)
    #
    # creates upper diaognal of adjecency matrix
    adj_verts = sparse.csc_matrix((np.full(len(adj_verts), 1, dtype=np.int32), (adj_verts[:, 0], adj_verts[:, 1])),
                                  shape=(len(mesh.v), len(mesh.v)))

    # full adjecency matrix
    adj_verts = adj_verts + adj_verts.T
    # -> in COOrdinate format
    # Does not support arithmetic operations and slicing.
    #
    adj_verts = adj_verts.tocoo()

    # queue of edges with cost.
    queue = []

    def collapse_cost(Qv, r, c, v):
        # sum quadrics for r and c
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        # reshape vertex into vector, with additional row, constant 1
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        # calculate cost for each vertex (scalar)
        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)

        # Return all costs including collapse cost
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # nnz -> number of stored values including zeroes
    for k in range(adj_verts.nnz):
        r = adj_verts.row[k]
        c = adj_verts.col[k]

        if r > c:
            # we don't want to count edges twice and we want them ordered
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c,)))

    ### Now we have calculated the removal and collapsing costs for all edges/vertex pairs
    # next step: which ones do we actually remove:

    collapse_list = []
    n_verts_total = len(mesh.v)
    faces = mesh.f.copy()
    while n_verts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        # this can never be true can it? an edge going from/to the same vertex?
        if r == c:
            # Can happen during reduction.
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)

        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            continue
        else:
            # destroy vertex that has the lowest destroy cost.
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # In all faces, replace the index of the vertex that is to be destroyed,
            # by the index of the vertex that is to keep ("contraction"
            np.place(faces, faces == to_destroy, to_keep)

            # also the entries in the queue containing that vertex can be removed
            indices_replace_0 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            indices_replace_1 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]

            for i in indices_replace_0:
                # ! doesnt work: queue[i][1][0] = to_keep, tuple does not support assignment
                queue[i] = (queue[i][0], (to_keep, queue[i][1][1]))
            for i in indices_replace_1:
                # queue[i][1][1] = to_keep
                queue[i] = (queue[i][0], (queue[i][1][0], to_keep))

            # new cost for vertices:
            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            # Check which faces are unique
            # There might be some which yield True, these can be removed
            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            # remaining faces:
            faces = faces[faces_to_keep, :].copy()

        # update number of total vertices that are contained in faces
        n_verts_total = (len(np.unique(faces.flatten())))

    # TODO could store original number of vertices and provide "vertices left" as remaining n_verts_total
    new_faces, mtx = get_sparse_transformation(faces, len(mesh.v))
    return new_faces, mtx


def get_sparse_transformation(faces, num_original_vertices):
    """
    Create sparse transformation matrix that tells us which faces to actually remove

    :param faces: The faces
    :param num_original_vertices: The number of original vertices, before downsampling
    """
    vertices_left = np.unique(faces.flatten())
    IS = np.arange(len(vertices_left))
    JS = vertices_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sparse.csc_matrix((data, ij), shape=(len(vertices_left), num_original_vertices))

    return (new_faces, mtx)


def generate_transformation_matrices(mesh: Mesh, factors):
    """
    Generates a number transformation matrices, which corresponds to the length of the factors.
    The number of factors also describes the number of layers that perform up/downsampling.
    Each transformation matrix is scaled by factors[i] and computes the transformation between them.

    Returns:
         M: a set of meshes downsampled from mesh by a factor specified in factors.
         A: the adjecency matrix for each of the meshes
         D: the downsampling transforms between each of the meshes
         U: the upsampling transforms between each of the meshes
    :param mesh: The mesh for which the up/downsampling transformations should be computed
    :param factors: The factors by which the mesh should be reduced

    :return: The downsampled meshes M, corresponding adjecency matrices A, and up-/downsampling transformations U/D.

    """

    factors = map(lambda x: 1.0 / x, factors)

    M, A, D, U, F, V = [], [], [], [], [], []

    # Set initial adjecency matrix and mesh
    A.append(get_vert_connectivity(mesh_v=mesh.v, mesh_f=mesh.f))
    M.append(mesh)
    F.append(mesh.f)
    V.append(mesh.v)

    for factor in factors:
        # get faces and the transformation
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D)
        # apply transformation to get vertices for next downsampling step
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        F.append(new_mesh.f)
        V.append(new_mesh.v)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f))
        # get upsampling transformation by vertex projection
        U.append(setup_deformation_transfer(M[-1], M[-2]))

    return M, A, D, U, F, V


def get_transformation_matrices(template_mesh, downsampling_factors, computed_folder="../computed/coma/"):
    if not os.path.exists(os.path.join(computed_folder, str(downsampling_factors), "meshes-save.npy")):
        meshes, adjecency_matrices, downsampling_matrices, upsampling_matrices, sampling_faces, sampling_vertices = \
            generate_transformation_matrices(template_mesh, downsampling_factors)

        if not os.path.exists(computed_folder):
            os.makedirs(computed_folder)
            np.save(os.path.join(computed_folder, "meshes-save.npy"), meshes)
            np.save(os.path.join(computed_folder, "adjecency_matrcies-save.npy"), adjecency_matrices)
            np.save(os.path.join(computed_folder, "downsampling_matrices-save.npy"), downsampling_matrices)
            np.save(os.path.join(computed_folder, "upsampling_matrices-save.npy"), upsampling_matrices)
            np.save(os.path.join(computed_folder, "sampling_faces.npy"), sampling_faces)
            np.save(os.path.join(computed_folder, "sampling_vertices.npy"), sampling_vertices)
    else:
        meshes = np.load(os.path.join(computed_folder, "meshes-save.npy"), allow_pickle=True)
        adjecency_matrices = np.load(os.path.join(computed_folder, "adjecency_matrcies-save.npy"),
                                     allow_pickle=True)
        downsampling_matrices = np.load(os.path.join(computed_folder, "downsampling_matrices-save.npy"),
                                        allow_pickle=True)
        upsampling_matrices = np.load(os.path.join(computed_folder, "upsampling_matrices-save.npy"),
                                      allow_pickle=True)
        sampling_faces = np.load(os.path.join(computed_folder, "sampling_faces.npy"), allow_pickle=True)
        sampling_vertices = np.load(os.path.join(computed_folder, "sampling_vertices.npy"), allow_pickle=True)

    return meshes, adjecency_matrices, downsampling_matrices, upsampling_matrices, sampling_faces, sampling_vertices
