import onnx

onnx.utils.extract_model('whole_model.onnx', 'partial_model.onnx', ['input.4'], ['input.20'])
onnx.utils.extract_model('whole_model.onnx', 'submodel_1.onnx', ['input.4'], ['onnx::Add_25', '31'])
onnx.utils.extract_model('whole_model.onnx', 'submodel_2.onnx', ['input.4', 'input.1'], ['input.20'])
# Error
# onnx.utils.extract_model('whole_model.onnx', 'submodel_3.onnx', ['input.12'], ['input.20'])
onnx.utils.extract_model('whole_model.onnx', 'more_output_model.onnx', ['input.1'], ['31', 'input.4', 'onnx::Add_25', 'onnx::Add_27'])

# =========================== 拆分模型 ===========================
onnx.utils.extract_model('whole_model.onnx', 'debug_model_1.onnx', ['input.1'], ['input.8'])
onnx.utils.extract_model('whole_model.onnx', 'debug_model_2.onnx', ['input.8'], ['onnx::Add_25'])
onnx.utils.extract_model('whole_model.onnx', 'debug_model_3.onnx', ['input.8'], ['onnx::Add_27'])
onnx.utils.extract_model('whole_model.onnx', 'debug_model_4.onnx', ['onnx::Add_25', 'onnx::Add_27'], ['31'])
