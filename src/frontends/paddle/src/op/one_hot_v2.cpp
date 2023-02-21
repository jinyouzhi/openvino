// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs one_hot_v2(const NodeContext& node) {
    auto x = node.get_input("X");
    auto num_classes = node.get_attribute<int32_t>("num_classes");

    return node.default_single_output_mapping({std::make_shared<ov::opset8::OneHot>(x, num_classes)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
