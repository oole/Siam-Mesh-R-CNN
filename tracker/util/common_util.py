import tensorflow._api.v2.compat.v1 as tf

def get_tf_version_tuple():
    """
    Return TensorFlow version as a 2-element tuple (for comparison).
    """
    # TODO Can be replaced by the tensorpack verison getter.
    return tuple(map(int, tf.__version__.split('.')[:2]))