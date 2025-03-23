# app.py: FastAPI REST API for Sketch-to-Colorful Image Generation
import torch
from torchvision import transforms
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import io

# Define Generator class (same as in your Colab Cell 5)
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 4, 2, 1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize FastAPI app
app = FastAPI()

# Load the trained generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
model_path = "generator.pth"  # Model file will be in the same directory on Render
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
generator.load_state_dict(torch.load(model_path))
generator.eval()

# Transform for input sketch
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/generate/")
async def generate_image(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    sketch = Image.open(io.BytesIO(contents)).convert('L')  # Convert to grayscale

    # Apply transforms
    sketch = transform(sketch).unsqueeze(0).to(device)

    # Generate colorful image
    with torch.no_grad():
        fake_photo = generator(sketch)
        fake_photo = fake_photo * 0.5 + 0.5  # Denormalize
        fake_photo = fake_photo.squeeze(0).cpu().permute(1, 2, 0).numpy()
        fake_photo = (fake_photo * 255).astype('uint8')
        fake_photo = cv2.cvtColor(fake_photo, cv2.COLOR_RGB2BGR)

    # Save the generated image temporarily
    output_path = "generated_photo.png"
    cv2.imwrite(output_path, fake_photo)

    # Return the image as a file response
    return FileResponse(output_path, media_type="image/png", filename="generated_photo.png")
