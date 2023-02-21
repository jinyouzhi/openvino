# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# one_hot_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import sys
import paddle


def one_hot_v2(name: str, x, num_classes):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape)
        out = paddle.nn.functional.one_hot(node_x, num_classes)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    test_cases = [
        "int32",
        "int64"
    ]


    num_class = 10
    data_x = np.array(
            [np.random.randint(0, num_class - 1) for i in range(6)]
    ).reshape([6, 1])
    for test in test_cases:
        one_hot_v2("one_hot_v2_" + test, paddle.to_tensor(data_x), num_class)


if __name__ == "__main__":
    main()
