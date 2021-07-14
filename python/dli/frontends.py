# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Parser Front End"""
import tvm
from tvm import relay


def load_relay(model_path):
    with open(model_path) as f:
        ir_mod = tvm.parser.fromtext(f.read())
    return ir_mod, None


def load_tensorflow(model_path, shape_dict):
    # pylint: disable=C0415
    import tensorflow as tf
    try:
        # Package "tf.compat.v1" is added from version "r1.13".
        tf_compat_v1 = tf.compat.v1
    except ImportError:
        tf_compat_v1 = tf

    with open(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # Import the graph definition into the default graph. Actually without this
    # line of code the TensorFlow model can be correctly converted to Relay IR,
    # but it can diagnose whether the TensorFlow model is broken or not.
    tf.import_graph_def(graph_def, name="")
    return relay.frontend.from_tensorflow(graph_def, shape=shape_dict)


def parse(framework, model_path, shape_dict):
    if framework == "relay":
        ir_mod, params = load_relay(model_path)
    elif framework == "tensorflow":
        ir_mod, params = load_tensorflow(model_path, shape_dict)

    if params is not None:
        # Use constant nodes(i.e., parameters or weights) to replace their
        # corresponding variable nodes. After importing, all inputs and weights
        # of NN model are converted to parameters of Relay "main" function, with
        # the help of this replacement only inputs of NN model will still be
        # parameters, and the "params" isn't needed, because it is merged into
        # "ir_mod".
        ir_mod["main"] = relay.build_module.bind_params_by_name(ir_mod["main"],
                                                                params)
    passes = [
        relay.transform.RemoveUnusedFunctions(),
    ]
    with tvm.transform.PassContext(opt_level=3):
        ir_mod = tvm.transform.Sequential(passes)(ir_mod)
    return ir_mod
