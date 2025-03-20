import torch
import torch.nn as nn
import torch.onnx
import numpy as np

print("Generating PyTorch CNN ONNX model...")

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

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Layer 3
        x = self.fc(x)

        return x

# Create an instance of the model
model = SimpleCNN()
model.eval()  # Set the model to evaluation mode

# Create a dummy input for the model (batch_size=1, channels=1, height=28, width=28)
dummy_input = torch.randn(1, 1, 28, 28)

# Define output path
output_path = 'examples/onnx/data/cnn.onnx'

# Export the model to ONNX format
torch.onnx.export(
    model,               # PyTorch model
    dummy_input,         # Dummy input tensor
    output_path,         # Output file path
    export_params=True,  # Export model parameters
    opset_version=12,    # ONNX opset version
    do_constant_folding=True,  # Optimize constant folding
    input_names=['input'],  # Name of the input
    output_names=['output'],  # Name of the output
    dynamic_axes={
        'input': {0: 'batch_size'},  # Variable batch size
        'output': {0: 'batch_size'}
    }
)

print(f"Model saved to {output_path}")

# Generate some test data for inference
test_data = torch.randn(3, 1, 28, 28)  # 3 samples of 28x28 images with 1 channel

# Test the model on the test data
with torch.no_grad():
    predictions = model(test_data)
    _, predicted_classes = torch.max(predictions, 1)

print("Test predictions with PyTorch model:")
for i, pred in enumerate(predicted_classes):
    print(f"  Sample {i+1}: Class {pred.item()}")
