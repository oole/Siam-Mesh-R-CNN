import tensorflow._api.v2.compat.v1 as tf
import numpy as np


class RingModel():
    def __init__(self, **kwargs):
        super(RingModel, self).__init__(**kwargs)

        self.load_path = "/mnt/storage/Msc/RingNet/model/ring_6_68641"
        model_path = self.load_path + ".meta"
        self.sess = tf.Session()
        saver = tf.compat.v1.train.import_meta_graph(model_path)
        tf.train.import_meta_graph(model_path)
        # restore checkpoint
        saver.restore(self.sess, self.load_path)
        saver = tf.train.Saver()
        ringnet_graph = self.sess.graph
        self.graph = ringnet_graph
        self.image_input = ringnet_graph.get_tensor_by_name(u'input_images:0')
        self.parameters = ringnet_graph.get_tensor_by_name(u'add_2:0')
        self.vertices = ringnet_graph.get_tensor_by_name(u'Flamenetnormal_2/Add_9:0')
        # self.graph_scope = "r_graph"
        # saver.save(self.sess, self.graph_scope)
        # new_graph = tf.Graph()
        # with new_graph.as_default():
        #     tf.train.import_meta_graph("r_graph.meta")#, import_scope="r_graph")
        #     new_image_input = new_graph.get_tensor_by_name(self.image_input.name)
        #     print(new_image_input.name)
        #
        #     new_vertices = new_graph.get_tensor_by_name(self.vertices.name)
        # from img2mesh2latent.util import img_util
        #
        # input_img, proc_param, img = img_util.preprocess_image(
        #     "/home/oole/git/projects-to-consume/RingNet/input_images/000001.jpg")
        # feed_dict = {new_image_input: np.expand_dims(input_img, axis=0)}
        #
        # fetch_dict = {'vertices': new_vertices}
        # with tf.Session(graph=new_graph) as sess:
        #     sav = tf.train.Saver()
        #     sav.restore(sess, "./r_graph")
        #     result = sess.run(fetch_dict, feed_dict)
        #     print(result)

        # at this point we have a initialized graph



        self.fetch_dict = {
            'vertices': self.vertices,
        }

    def get_checkpoint_path(self):
        return self.load_path
    def get_ringnet_graph(self):
        return self.graph

    def get_image_input(self):
        return self.image_input

    def get_vertices_output(self):
        return self.vertices

    def get_feed_dict_for_ring(self, image_input: str):
        feed_dict = {
            self.image_input: image_input
        }
        return feed_dict

    def test_ringnet(self):
        ## This works:
        from img2mesh2latent.util import img_util

        input_img, proc_param, img = img_util.preprocess_image(
            "/home/oole/git/projects-to-consume/RingNet/input_images/000001.jpg")
        feed_dict = {self.image_input: np.expand_dims(input_img, axis=0)}
        result = self.sess.run(self.fetch_dict, feed_dict)
        print(result)