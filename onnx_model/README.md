<div align='center'>
<h1> Onnx support for LVM-Med </h1>
</div>

- Open Neural Network Exchange ([ONNX](https://github.com/onnx/onnx)) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Currently ONNX concentrates on the capabilities needed for inferencing (scoring).     

- ONNX is widely supported and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. 

- Here, we release 2 versions of onnx models for LVM-Med which are based on Resnet-50 backbone ([`lvmmed_rn50.onnx`](./lvmmed_rn50.onnx)) and ViT backbone ([`lvmmed_vit.onnx`](./lvmmed_vit.onnx)). Also, we also release our code to transform any other LVM-Med based models into ONNX models [`torch2onnx.py`](./torch2onnx.py).

### Onnx in LVM-Med 
- To load onnx LVM-Med models:
```python
"""
Load ONNX model
"""
onnx_model_rn50 = onnx.load("onnx_model/lvmmed_rn50.onnx") # If ResNet-50 backbone
onnx_model_vit = onnx.load("onnx_model/lvmmed_vit.onnx")   # If ViT backbone

"""
Check ONNX
"""
try:
    onnx.checker.check_model(onnx_model_rn50)
    print("RN50 onnx passed!")
except:
    print("RN50 onnx failed!")
```