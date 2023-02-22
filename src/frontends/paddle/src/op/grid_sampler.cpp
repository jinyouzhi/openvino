// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset9.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs grid_sampler(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto grid = node.get_input("Grid");
    ov::opset9::GridSample::Attributes attributes{};
    attributes.align_corners = node.get_attribute<bool>("align_corners", true);
    attributes.mode = ov::EnumNames<ov::opset9::GridSample::InterpolationMode>::as_enum(
        node.get_attribute<std::string>("mode", "bilinear"));
    attributes.padding_mode = ov::EnumNames<ov::opset9::GridSample::PaddingMode>::as_enum(
        node.get_attribute<std::string>("padding_mode", "zeros"));

    return node.default_single_output_mapping({std::make_shared<ov::opset9::GridSample>(x, grid, attributes)}, {"Output"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov