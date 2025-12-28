import onnx
from onnx import helper


def make_moe_gating_top_k_softmax_1():
    node = helper.make_node('NPUMoeGatingTopKSoftmax',
                            inputs=['gating', 'finished'],
                            outputs=['out', 'expert_idx', 'row_idx'],
                            k=2)
    graph = helper.make_graph(
        nodes=[node],
        name="test_MoeGatingTopKSoftmax",
        inputs=[helper.make_tensor_value_info("gating", onnx.TensorProto.FLOAT, [2, 16]),
                helper.make_tensor_value_info("finished", onnx.TensorProto.BOOL, [2])],
        outputs=[helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [2, 2]),
                 helper.make_tensor_value_info("expert_idx", onnx.TensorProto.INT32, [2, 2]),
                 helper.make_tensor_value_info("row_idx", onnx.TensorProto.INT32, [2, 2])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./model_npu_moe_gating_top_k_softmax.onnx")


def make_moe_gating_top_k_softmax_2():
    node = helper.make_node('NPUMoeGatingTopKSoftmax',
                            inputs=['gating', 'finished'],
                            outputs=['out', 'expert_idx', 'row_idx'])
    graph = helper.make_graph(
        nodes=[node],
        name="test_MoeGatingTopKSoftmax",
        inputs=[helper.make_tensor_value_info("gating", onnx.TensorProto.FLOAT, [2, 16]),
                helper.make_tensor_value_info("finished", onnx.TensorProto.BOOL, [2])],
        outputs=[helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [2, 2]),
                 helper.make_tensor_value_info("expert_idx", onnx.TensorProto.INT32, [2, 2]),
                 helper.make_tensor_value_info("row_idx", onnx.TensorProto.INT32, [2, 2])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./model_npu_moe_gating_top_k_softmax_no_k.onnx")


if __name__ == '__main__':
    make_moe_gating_top_k_softmax_1()
    make_moe_gating_top_k_softmax_2()
