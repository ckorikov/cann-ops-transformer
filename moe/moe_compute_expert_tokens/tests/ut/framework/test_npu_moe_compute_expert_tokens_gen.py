import onnx
import os
from onnx import helper
from onnx import TensorProto


def make_npu_moe_compute_expert_tokens_onnx():
    """v11"""
    sorted_experts = helper.make_tensor_value_info('sorted_experts', TensorProto.INT32, [10])
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [5])
    node_def = helper.make_node(
        'NPUMoeComputeExpertTokens',
        inputs=['sorted_experts'],
        outputs=['y'],
        num_experts = 5
    )

    graph = helper.make_graph(
        [node_def],
        'NPUMoeComputeExpertTokens_Graph',
        [sorted_experts],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 1
    model.opset_import[0].domain = "npu"
    current_patch = os.path.abspath(__file__)
    idx = current_patch.rfind('/')
    current_patch = current_patch[:idx]
    onnx.save(model, current_patch + "/test_npu_moe_compute_expert_tokens_case1_v11.onnx")

if __name__ == '__main__':
    make_npu_moe_compute_expert_tokens_onnx()