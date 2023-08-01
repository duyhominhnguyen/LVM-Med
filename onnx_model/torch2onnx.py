import torch.onnx
import torch 
from torch import nn 
from torch.nn import functional as F 

import onnx
from transformers import AutoModel


def export_onnx(example_input: torch.Tensor, 
                model,
                onnx_model_name) -> None:
    torch.onnx.export(
        model, 
        example_input,
        onnx_model_name,
        export_params=False,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {
                0 : 'batch_size'
            },
            'output' : {
                0 : 'batch_size'
            }
        }
    )

if __name__ == "__main__":
    """
    Export LVM-Med (RN50 version)
    """
    example_input_rn50 = torch.ones(1, 3, 1024, 1024)
    lvmmed_rn50 = AutoModel.from_pretrained('ngctnnnn/lvmmed_rn50')
    example_output_rn50 = lvmmed_rn50(example_input_rn50)['pooler_output']
    print(f"Example output for LVM-Med (RN50)'s shape: {example_output_rn50.shape}")

    export_onnx(example_input_rn50, lvmmed_rn50, onnx_model_name="onnx_model/lvmmed_rn50.onnx")

    """
    Export LVM-Med (ViT)
    """
    example_input_vit = torch.ones(1, 3, 224, 224)
    lvmmed_vit = AutoModel.from_pretrained('ngctnnnn/lvmmed_vit')
    example_output_vit = lvmmed_vit(example_input_vit)['pooler_output']
    print(f"Example output for LVM-Med (RN50)'s shape: {example_output_vit.shape}")
    
    export_onnx(example_input_vit, lvmmed_vit, onnx_model_name="onnx_model/lvmmed_vit.onnx")