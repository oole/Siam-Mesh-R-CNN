from psbody.mesh import MeshViewers, Mesh
import readchar
from util.log_util import date_print, hint_print
import numpy as np


def visualizeSideBySide(original, prediction, number_of_meshes, mesh_data):
    viewer = MeshViewers(shape=(2, number_of_meshes), titlebar="Above:Original, Below: Prediction")
    for i in range(number_of_meshes):
        viewer[1][i].set_dynamic_meshes([mesh_data.vec2mesh(original[i])])
        viewer[0][i].set_dynamic_meshes([mesh_data.vec2mesh(prediction[i])])


def pageThroughMeshes(meshes, mesh_data):
    viewer = MeshViewers(shape=(1, 1), titlebar="Page view")
    index = 0
    num_meshes = meshes.shape[0]
    while (1):

        # date_print("Change latent representation +(1,2,3,4,5,6,7,8,9) -(q,w,e,r,t,y,u,i).")

        input_key = readchar.readchar()
        if input_key == "n":
            index += 1
        elif input_key == "b":
            index -= 1
        elif input_key == "\x1b":
            # escape
            break
        else:
            hint_print("n for next page, p for previous page.")


        if (index >= num_meshes):
            index = num_meshes - 1
            hint_print("End reached")
        if (index <= 0):
            index = 0
            hint_print("Start reached")

        date_print("Viewing index: " + str(index))

        # decode
        # visualize
        viewer[0][0].set_dynamic_meshes([mesh_data.vec2mesh(meshes[index])])

def pageThroughPredictionMeshes(meshes, mesh_data, coma_model, batch_size):
    viewer = MeshViewers(shape=(1, 2), titlebar="Pageing prediction")
    viewer_two = MeshViewers(shape=(1, 1), titlebar="Original")
    index = 0
    num_meshes = meshes.shape[0]
    while (1):

        # date_print("Change latent representation +(1,2,3,4,5,6,7,8,9) -(q,w,e,r,t,y,u,i).")

        input_key = readchar.readchar()
        if input_key == "n":
            index += 1
        elif input_key == "p":
            index -= 1
        elif input_key == "\x1b":
            # escape
            break
        else:
            hint_print("n for next page, p for previous page.")


        if (index >= num_meshes):
            index = num_meshes - 1
            hint_print("End reached")
        if (index <= 0):
            index = 0
            hint_print("Start reached")

        date_print("Viewing index: " + str(index))
        original_vertices = meshes[index]
        predicted_vertices = coma_model.predict(np.full((batch_size, 5023, 3), original_vertices), batch_size=batch_size)[0]

        original_mesh = mesh_data.vec2mesh(original_vertices)
        predicted_mesh = mesh_data.vec2mesh(predicted_vertices)
        # decode
        # visualize
        viewer_two[0][0].set_dynamic_meshes([original_mesh])
        viewer[0][0].set_dynamic_meshes([original_mesh])
        viewer[0][1].set_dynamic_meshes([predicted_mesh])


def euclidean_error(x_test, x_result, mesh_data):
    return np.mean(np.sqrt(np.sum((mesh_data.std * (x_result - mesh_data.vertices_test)) ** 2, axis=2)))