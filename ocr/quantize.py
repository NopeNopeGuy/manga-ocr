from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx", 
    "quantized_model.onnx",
     weight_type=QuantType.QInt4, 
     nodes_to_exclude=['/encoder/embeddings/patch_embeddings/projection/Conv']
)
