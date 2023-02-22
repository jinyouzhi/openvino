# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# grid_sampler paddle model generator
#
import numpy as np
import paddle
from save_model import saveModel
import sys


def grid_sample(name: str, x, grid, mode="bilinear", padding_mode="zeros", align_corners=True):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        grid_node = paddle.static.data(name="grid", shape=grid.shape, dtype=grid.dtype)
        out = paddle.nn.functional.grid_sample(x_node, grid_node, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x,
                  'grid': grid},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x', 'grid'], fetchlist=[out], inputs=[
                  x, grid], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    for x_shape, grid_shape in zip(
        [[1, 1, 3, 3], [1, 2, 3, 4, 5]],
        [[1, 3, 3, 2], [1, 3, 4, 5, 3]]):
        for idx, data_type, mode, padding_mode, align_corners in zip(
            [1, 2, 3],
            ['float32', 'float64', 'float32'],
            ['bilinear', 'nearest', 'bilinear'],
            ['zeros', 'refleaction', 'border'],
            [True, False, True]):
            x = np.random.randn(*(x_shape)).astype(data_type)
            grid = np.random.uniform(-1, 1, grid_shape).astype(data_type)
            grid_sample(name='grid_sampler_' + str(len(x_shape)) + '-D_' + str(idx),
                        x=x, grid=grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


if __name__ == "__main__":
    main()