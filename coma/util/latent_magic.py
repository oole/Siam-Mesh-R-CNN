import numpy as np
import readchar
from psbody.mesh import MeshViewers

from coma.model import model


def play_with_latent_space(model: model.CoMa, mesh_data, mesh=None):
    vertices = None
    if mesh is not None:
        # use specified mesh
        vertices = mesh.v
    else:
        # use template mesh
        vertices = mesh_data.reference_mesh.v

    latent_representation = model.encode(np.expand_dims(vertices, axis=0))
    viewer = MeshViewers(shape=(1, 1), titlebar="Interactive Latent Representation")
    viewer[0][0].set_dynamic_meshes([mesh_data.vec2mesh(vertices)])
    print(str(latent_representation[0]))
    while (1):
        # date_print("Change latent representation +(1,2,3,4,5,6,7,8,9) -(q,w,e,r,t,y,u,i).")

        input_key = readchar.readchar()

        if input_key == "1":
            latent_representation[0][0] = 1.01 * latent_representation[0][0]
        elif input_key == "2":
            latent_representation[0][1] = 1.01 * latent_representation[0][1]
        elif input_key == "3":
            latent_representation[0][2] = 1.01 * latent_representation[0][2]
        elif input_key == "4":
            latent_representation[0][3] = 1.01 * latent_representation[0][3]
        elif input_key == "5":
            latent_representation[0][4] = 1.01 * latent_representation[0][4]
        elif input_key == "6":
            latent_representation[0][5] = 1.01 * latent_representation[0][5]
        elif input_key == "7":
            latent_representation[0][6] = 1.01 * latent_representation[0][6]
        elif input_key == "8":
            latent_representation[0][7] = 1.01 * latent_representation[0][7]

        elif input_key == "q":
            latent_representation[0][0] = 0.99 * latent_representation[0][0]
        elif input_key == "w":
            latent_representation[0][1] = 0.99 * latent_representation[0][1]
        elif input_key == "e":
            latent_representation[0][2] = 0.99 * latent_representation[0][2]
        elif input_key == "r":
            latent_representation[0][3] = 0.99 * latent_representation[0][3]
        elif input_key == "t":
            latent_representation[0][4] = 0.99 * latent_representation[0][4]
        elif input_key == "y":
            latent_representation[0][5] = 0.99 * latent_representation[0][5]
        elif input_key == "u":
            latent_representation[0][6] = 0.99 * latent_representation[0][6]
        elif input_key == "i":
            latent_representation[0][7] = 0.99 * latent_representation[0][7]
        elif input_key == "\x1b":
            # escape
            break
        else:
            print("(1,2,3,4,5,6,7,8) to increase values (*1.01), (q,w,e,r,t,y,u,i) to decrease values(*0.99)")
        # decode
        # visualize
        print(str(latent_representation[0]))
        print(latent_representation)
        decoded = model.decode(latent_representation)
        print("Decoded vertices: {}".format(decoded))
        viewer[0][0].set_dynamic_meshes([mesh_data.vec2mesh(decoded)])


def sample_latent_space(model: model.CoMa, mesh_data, batch_size, mesh=None):
    print("SAMPLING LATENT SPACE")
    if mesh is not None:
        mesh = np.full((batch_size, 5023, 3), mesh)
        # use specified mesh
    else:
        # use template mesh
        mesh = np.full((batch_size, 5023, 3), mesh_data.reference_mesh.v)

    latent_representation = model.encode(mesh).numpy()[0]
    samples = []
    for i in [0, 2, 5, 6]:
        component_samples = []
        for j in range(-2, 3, 1):
            new_rep = np.copy(latent_representation)
            new_rep[i] = (1 + 0.5 * j) * latent_representation[i]
            component_samples.append(new_rep)
        component_samples = np.array(component_samples)
        samples.append(component_samples)
    samples = np.array(samples)
    decoded = []
    for lat in samples:
        element = []
        for rep in lat:
            element.append(model.decode(np.full((batch_size, 8), rep)).numpy()[0])
        decoded.append(np.array(element))
    decoded = np.array(decoded)

    # TODO iterate over size of latent space,
    # then iterate over j and visualize the meshes paged.
    viewer = MeshViewers(shape=(len(decoded), len(decoded[0])), titlebar="Sampling")
    for i in range(len(decoded)):
        latent_elem = decoded[i]
        for j in range(len(latent_elem)):
            mesh = mesh_data.vec2mesh(latent_elem[j])
            viewer[i][j].set_dynamic_meshes([mesh])
