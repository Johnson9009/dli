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
import sys
from argparse import ArgumentParser

from tvm.driver import tvmc

from dli import frontends


def main():
    parser = ArgumentParser(description="dli.")
    parser.add_argument("-f", "--framework", metavar="BUILD_TYPE",
                        help="The NN model's framework.")
    parser.add_argument("-m", "--model_path",
                        help="The NN model file path.")
    parser.add_argument("-s", "--shape_dict",
                        type=tvmc.common.parse_shape_string,
                        help=("The input shape dictionary, format is "
                              '"input0_name:[dim0,dim1,...,dimN]'
                              ' input1_name:[dim0,dim1]".'))
    parser.add_argument("-o", "--output", help="Relay IR filename.",
                        default="relay.rly")
    cfg = parser.parse_args()

    ir_mod = frontends.parse(cfg.framework, cfg.model_path, cfg.shape_dict)

    # Simplify and optimization.
    passes = [
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.SimplifyExpr(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.CanonicalizeOps(),
    ]
    with tvm.transform.PassContext(opt_level=3):
        ir_mod = tvm.transform.Sequential(passes)(ir_mod)

    with open("{cfg.output}", "w") as f:
        f.write(ir_mod.astext())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Just print a newline character to keep the shell prompt neat.
        sys.stdout.write('\n')
        sys.exit(1)
