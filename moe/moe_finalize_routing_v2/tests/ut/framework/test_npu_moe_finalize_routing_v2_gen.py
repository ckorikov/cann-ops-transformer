import onnx
import os
from onnx import helper
from onnx import TensorProto


def make_npu_moe_finalize_routing_v2_onnx():
    """v22"""
    e = 3
    c = 10
    h = 7
    n = 12
    k = 2
    expanded_x = helper.make_tensor_value_info('expanded_x', TensorProto.FLOAT, [e, c, h])
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [n, h])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [n, h])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [e, h])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [n, k])
    expanded_row_idx = helper.make_tensor_value_info('expanded_row_idx', TensorProto.INT32, [n * k])
    expanded_expert_idx = helper.make_tensor_value_info('expanded_expert_idx', TensorProto.INT32, [n, k])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [n, h])
    node_def = helper.make_node(
        'NPUMoeFinalizeRoutingV2',
        inputs=['expanded_x', 'expanded_row_idx', 'x1', 'x2', 'bias', 'scales', 'expanded_expert_idx'],
        drop_pad_mode = 1,
        outputs=['y']
    )

    graph = helper.make_graph(
        [node_def],
        'NPUMoeFinalizeRoutingV2_Graph',
        [expanded_x, expanded_row_idx, x1, x2, bias, scales, expanded_expert_idx],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 1
    model.opset_import[0].domain = "npu"
    current_patch = os.path.abspath(__file__)
    idx = current_patch.rfind('/')
    current_patch = current_patch[:idx]
    onnx.save(model, current_patch + "/test_npu_moe_finalize_routing_v2_case1_v11.onnx")

if __name__ == '__main__':
    make_npu_moe_finalize_routing_v2_onnx()