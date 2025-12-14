import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io

# Define your model (simplified version of your CNN)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv2_drop(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# Load model
model = CNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

def preprocess_image(image):
    """Convert uploaded image to model input format"""
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(image)
    # Normalize (MNIST has black background, white digits)
    # Invert if necessary
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    # Add batch and channel dimensions
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor

def predict_digit(image):
    """Predict digit from uploaded image"""
    if image is None:
        return "Please upload an image"
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_digit].item()
    
    # Create confidence bars
    confidences = {str(i): float(probabilities[i]) for i in range(10)}
    
    return f"Predicted: {predicted_digit} (Confidence: {confidence:.2%})", confidences

# Create Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Upload Digit Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=3, label="Confidences")
    ],
    title="MNIST Digit Classifier",
    description="Upload an image of a handwritten digit (0-9) and the model will predict what digit it is.",
    examples=[
        ["sample_0.png"],
        ["sample_3.png"],
        ["sample_8.png"]
    ]
)

if __name__ == "__main__":
    interface.launch()