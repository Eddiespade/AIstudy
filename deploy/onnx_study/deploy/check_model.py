import onnx

onnx_model = onnx.load("../deploy_04/asinh.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")