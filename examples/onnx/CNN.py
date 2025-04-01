import torch
import torch.nn as nn
import torch.onnx
import numpy as np

print("Quantizing PyTorch CNN model (simplified approach)...")

# Define a simple CNN model with 3 layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: Convolutional layer with 1 input channel, 6 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Layer 2: Convolutional layer with 6 input channels, 12 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Layer 3: Fully connected layer with 12*28*28 input features and 10 output features
        self.fc = nn.Linear(12 * 28 * 28, 10)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)

        # Flatten the output - use reshape instead of view
        x = x.reshape(x.size(0), -1)

        # Layer 3
        x = self.fc(x)

        return x

# 整数値入力処理のためのラッパークラス
class QuantizedModelWrapper:
    def __init__(self, model, input_scale=1/255.0, input_zero_point=0):
        self.model = model
        self.input_scale = input_scale
        self.input_zero_point = input_zero_point

    def __call__(self, integer_input):
        """整数値入力を受け取り、前処理してから推論を実行"""
        # 整数入力を浮動小数点に変換
        if isinstance(integer_input, np.ndarray):
            float_input = (integer_input.astype(np.float32) - self.input_zero_point) * self.input_scale
            float_input = torch.from_numpy(float_input)
        elif isinstance(integer_input, torch.Tensor):
            if integer_input.dtype == torch.uint8 or integer_input.dtype == torch.int8:
                float_input = (integer_input.float() - self.input_zero_point) * self.input_scale
            else:
                float_input = integer_input  # すでに浮動小数点の場合
        else:
            raise TypeError("Input must be numpy array or torch tensor")

        # モデルで推論実行
        with torch.no_grad():
            output = self.model(float_input)

        return output

# 量子化関数
def quantize_model_to_int8():
    # モデルのインスタンス化
    model = SimpleCNN()
    model.eval()  # 評価モードに設定

    # 量子化パラメータ
    input_scale = 1/255.0
    input_zero_point = 0

    # ONNXエクスポート用の量子化ヒントを埋め込む（オプション）
    # これにより、ONNXランタイムが自動的に量子化を実行できる場合がある

    # ダミー入力（0-255の範囲の整数値）
    dummy_input_int = np.random.randint(0, 256, (1, 1, 28, 28), dtype=np.uint8)

    # 整数入力を浮動小数点に変換
    dummy_input_float = (dummy_input_int.astype(np.float32) - input_zero_point) * input_scale
    dummy_input = torch.from_numpy(dummy_input_float)

    # 入力と出力のスケール情報を設定
    input_dynamic_range = {
        'input': {
            'data_type': 'int8',
            'dynamic_range': [-128, 127],
            'input_scale': input_scale,
            'input_zero_point': input_zero_point
        }
    }

    # ONNXエクスポート
    output_path = 'examples/onnx/data/int8_cnn.onnx'
    torch.onnx.export(
        model,               # PyTorchモデル
        dummy_input,         # ダミー入力テンソル（浮動小数点）
        output_path,         # 出力ファイルパス
        export_params=True,  # モデルパラメータをエクスポート
        opset_version=12,    # ONNX opsetバージョン
        do_constant_folding=True,  # 定数畳み込みを最適化
        input_names=['input'],  # 入力の名前
        output_names=['output'],  # 出力の名前
        dynamic_axes={
            'input': {0: 'batch_size'},  # 可変バッチサイズ
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model saved to {output_path}")

    # 整数入力ラッパーを作成
    wrapped_model = QuantizedModelWrapper(model, input_scale, input_zero_point)

    return wrapped_model, model

# テスト関数
def test_with_integer_input(wrapped_model):
    # 整数値のテストデータ (0-255範囲)
    test_data = np.random.randint(0, 256, (3, 1, 28, 28), dtype=np.uint8)

    # テスト実行
    results = []
    for i in range(test_data.shape[0]):
        sample = test_data[i:i+1]  # バッチ次元を保持

        # 推論実行 (ラッパーが内部で整数→浮動小数点変換を行う)
        output = wrapped_model(sample)

        # 結果の取得
        _, predicted_class = torch.max(output, 1)
        results.append(predicted_class.item())

    print("Test predictions with integer input:")
    for i, pred in enumerate(results):
        print(f"  Sample {i+1}: Class {pred}")

# ONNXモデルを量子化する方法（後処理）
def quantize_onnx_model():
    print("\nTo further quantize the ONNX model after export, you can use onnxruntime:")
    print("""
    # Install required packages: pip install onnx onnxruntime onnxruntime-tools

    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # Load the model
    model_fp32 = 'examples/onnx/data/int8_cnn.onnx'
    model_quant = 'examples/onnx/data/int8_cnn_quantized.onnx'

    # Quantize the model
    quantize_dynamic(
        model_fp32,
        model_quant,
        weight_type=QuantType.QInt8
    )

    print(f"Quantized ONNX model saved to {model_quant}")
    """)

# メイン実行部分
if __name__ == "__main__":
    # モデルを量子化
    wrapped_model, original_model = quantize_model_to_int8()

    # 整数入力でテスト
    test_with_integer_input(wrapped_model)

    # ONNXモデルの量子化手順を表示
    quantize_onnx_model()

    # 使用例
    print("\nExample of how to use the exported ONNX model with integer input:")
    print("""
    # Python with ONNX Runtime
    import onnxruntime as ort
    import numpy as np

    # 0-255の整数値入力
    integer_input = np.random.randint(0, 256, (1, 1, 28, 28), dtype=np.uint8)

    # 入力を変換（整数→浮動小数点）
    input_scale = 1/255.0
    input_zero_point = 0
    float_input = (integer_input.astype(np.float32) - input_zero_point) * input_scale

    # ONNXランタイムセッションの作成
    session = ort.InferenceSession("examples/onnx/data/int8_cnn.onnx")

    # 入出力名を取得
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 推論実行
    results = session.run([output_name], {input_name: float_input})

    # 結果の解析
    prediction = np.argmax(results[0], axis=1)
    print(f"Predicted class: {prediction[0]}")
    """)