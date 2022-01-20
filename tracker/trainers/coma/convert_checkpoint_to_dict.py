import argparse
import os

import six
from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', help='The checkpoint dir that should be used for dictionary creation',
                    default='../computed/coma/tensorpack-test-9', required=True)
parser.add_argument('--checkpoint',
                    help="The checkpoint that should be used (default: checkpoint), can be any saved step",
                    default="checkpoint")

args = parser.parse_args()
checkpoint_path = varmanip.get_checkpoint_path(os.path.join(args.ckeckpoint_dir, args.checkpoint))
print(checkpoint_path)
checkpoint_vars_dict = varmanip.load_checkpoint_vars(checkpoint_path)
save_var_dict = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(checkpoint_vars_dict) if
                 "tower0" not in get_op_tensor_name(k)[1] and "global_step" not in get_op_tensor_name(k)[
                     1]}

varmanip.save_chkpt_vars(save_var_dict, args.checkpoint_dir + "-" + args.checkpoint + ".npz")
