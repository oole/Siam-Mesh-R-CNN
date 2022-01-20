import numpy as np
import openmesh as om
from sklearn.neighbors import KDTree


def process_spiral(faces, vertices, spiral_lengths, spiral_dilations):
    spirals = []
    for i in range(len(faces)-1):
        if vertices is not None:
            mesh = om.TriMesh(np.array(vertices[i]), np.array(faces[i]))
        else:
            n_vertices = faces[i].max = 1
            mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(faces[i]))

        spirals.append(extract_spirals(mesh, spiral_lengths[i], spiral_dilations[i]))
    return spirals


def extract_spirals(mesh, spiral_length, spiral_dilation):
    spirals = []
    for origin_vertex in mesh.vertices():
        reference_one_ring = []
        for first_vertex in mesh.vv(origin_vertex):
            reference_one_ring.append(first_vertex.idx())
        spiral = [origin_vertex.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = get_next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while (len(spiral) + len(next_ring) < spiral_length * spiral_dilation):
            if len(next_ring) == 0:
                break

            last_ring = next_ring
            next_ring = get_next_ring(mesh, last_ring, spiral)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]], axis=0), k=spiral_length * spiral_dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(np.asarray(spiral[:spiral_length * spiral_dilation][::spiral_dilation]))
    return np.asarray(spirals)


def get_next_ring(mesh, last_ring, spiral):
    res = []

    for first_vertex in last_ring:
        first_vertex = om.VertexHandle(first_vertex)
        after_last_ring = False
        for second_vertex in mesh.vv(first_vertex):
            if after_last_ring:
                if is_new_vertex(second_vertex, last_ring, spiral, res):
                    res.append(second_vertex.idx())
            if second_vertex.idx() in last_ring:
                after_last_ring = True
        for second_vertex in mesh.vv(first_vertex):
            if second_vertex.idx() in last_ring:
                break
            if is_new_vertex(second_vertex, last_ring, spiral, res):
                res.append(second_vertex.idx())

    return res


def is_new_vertex(vertex, last_ring, spiral, res):
    idx = vertex.idx()
    return idx not in last_ring and idx not in spiral and idx not in res
