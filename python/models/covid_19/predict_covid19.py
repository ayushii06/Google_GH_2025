import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# class for the CNN model
class CovidCNN(nn.Module):
    def __init__(self):
        super(CovidCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 classes: COVID, Normal, Pneumonia

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = CovidCNN()
model.load_state_dict(torch.load("models/covid_19/covid_cnn.pth"))
model.eval()

# Function to predict COVID-19
def predict_covid19(image_path):
    
    # Define the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()  # Get predicted class index
        print(predicted_class)
        
    class_names = ["COVID-19", "NORMAL", "PNEUMONIA"]
    print("class", class_names[predicted_class])
    
    # The predicted class
    result = class_names[predicted_class]
    
    #RETURN THE PREDICTED CLASS
    return result

