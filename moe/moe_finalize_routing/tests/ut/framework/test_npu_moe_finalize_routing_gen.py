import onnx
import os
from onnx import helper
from onnx import TensorProto


def make_npu_moe_finalize_routing_onnx():
    """v11"""
    expanded_x = helper.make_tensor_value_info('expanded_x', TensorProto.FLOAT, [24, 7])
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [12, 7])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [12, 7])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [3, 7])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [12, 2])
    expanded_row_idx = helper.make_tensor_value_info('expanded_row_idx', TensorProto.INT32, [24])
    expanded_expert_idx = helper.make_tensor_value_info('expanded_expert_idx', TensorProto.INT32, [12, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [12, 7])
    node_def = helper.make_node(
        'NPUMoeFinalizeRouting',
        inputs=['expanded_x', 'x1', 'x2', 'bias', 'scales', 'expanded_row_idx', 'expanded_expert_idx'],
        outputs=['y']
    )

    graph = helper.make_graph(
        [node_def],
        'NPUMoeFinalizeRouting_Graph',
        [expanded_x, x1, x2, bias, scales, expanded_row_idx, expanded_expert_idx],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 1
    model.opset_import[0].domain = "npu"
    current_patch = os.path.abspath(__file__)
    idx = current_patch.rfind('/')
    current_patch = current_patch[:idx]
    onnx.save(model, current_patch + "/test_npu_moe_finalize_routing_case1_v11.onnx")

if __name__ == '__main__':
    make_npu_moe_finalize_routing_onnx()