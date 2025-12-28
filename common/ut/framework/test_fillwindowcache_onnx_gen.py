import onnx
import os
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def export_fillwindowcache():
    # Create two input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 8, 1, 1])
    clean_cache = helper.make_tensor_value_info('clean_cache', TensorProto.BOOL, [1,])
  
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 8, 2, 1])

    node = helper.make_node(
        'FillWindowCache',
        inputs=['x', 'clean_cache'],
        outputs=['y'],
        axis=2,
        cache_depth=2)

    graph_def = helper.make_graph(
        [node],
        'test_fillwindowcache',
        [x, clean_cache],
        [y])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-parser_test')
    model_def.opset_import[0].version = 11
    current_patch = os.path.abspath(__file__)
    idx = current_patch.rfind('/')
    current_patch = current_patch[:idx]
    onnx.save(model_def, current_patch + "/test_fillwindowcache_onnx_case1.onnx")

if __name__ == '__main__':
    export_fillwindowcache()

