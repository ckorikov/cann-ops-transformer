import onnx
from onnx import helper
import os


def make_moe_init_routing():
    print("start gen moe_init_routing onnx file")
    node = helper.make_node('NPUMoeInitRouting',
                            inputs=["x", "row_idx", "expert_idx"],
                            outputs=["expanded_x", "expanded_row_idx", "expanded_expert_idx"],
                            active_num=99)
    graph = helper.make_graph(
        nodes=[node],
        name="test_MoeInitRouting",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4]),
                helper.make_tensor_value_info("row_idx", onnx.TensorProto.INT32, [3, 2]),
                helper.make_tensor_value_info("expert_idx", onnx.TensorProto.INT32, [3, 2])],
        outputs=[helper.make_tensor_value_info("expanded_x", onnx.TensorProto.FLOAT, [6, 4]),
                 helper.make_tensor_value_info("expanded_row_idx", onnx.TensorProto.INT32, [6]),
                 helper.make_tensor_value_info("expanded_expert_idx", onnx.TensorProto.INT32, [6])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    current_patch = os.path.abspath(__file__)
    idx = current_patch.rfind('/')
    current_patch = current_patch[:idx]

    onnx.save(model, current_patch + "/model_npu_moe_init_routing.onnx")
    print("end gen moe_init_routing onnx file")

if __name__ == '__main__':
    make_moe_init_routing()
