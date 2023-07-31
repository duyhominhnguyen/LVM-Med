import onnx 
"""
Unit test for onnx model
"""
onnx_model_rn50 = onnx.load("onnx_model/lvmmed_rn50.onnx")
try:
    onnx.checker.check_model(onnx_model_rn50)
    print("[Unit test] RN50 onnx passed!")
except:
    print("[Unit test] RN50 onnx failed!")

onnx_model_vit = onnx.load("onnx_model/lvmmed_vit.onnx")
try:
    onnx.checker.check_model(onnx_model_vit)
    print("[Unit test] ViT onnx passed!")
except:
    print("[Unit test] ViT onnx failed!")