import onnx
from onnx import helper
import os


def make_rotary_position_embedding():
    print("start gen rotary_position_embedding onnx file")
    node = helper.make_node('NPURotaryPositionEmbedding',
                            inputs=["x", "r1", "r2"],
                            outputs=["y"],
                            mode=1)
    graph = helper.make_graph(
        nodes=[node],
        name="test_RotaryPositionEmbedding",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 1, 64]),
                helper.make_tensor_value_info("r1", onnx.TensorProto.FLOAT, [1, 1, 1, 64]),
                helper.make_tensor_value_info("r2", onnx.TensorProto.FLOAT, [1, 1, 1, 64])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 1, 64])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    current_patch = os.path.abspath(__file__)
    idx = current_patch.rfind('/')
    current_patch = current_patch[:idx]

    onnx.save(model, current_patch + "/model_npu_rotary_position_embedding.onnx")
    print("end gen rotary_position_embedding onnx file")

if __name__ == '__main__':
    make_rotary_position_embedding()
