# Example of a simple multi-modal generative AI using pre-trained models

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import torchvision.transforms as transforms

# Load pre-trained GPT-2 model and tokenizer for text generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load pre-trained image classification model (e.g., ResNet)
image_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
image_model.eval()

def generate_text(prompt):
    # Tokenize input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate text based on input prompt
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def generate_image_caption(image_path):
    # Load and preprocess image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)
    # Classify image and get prediction
    with torch.no_grad():
        output = image_model(input_batch)
    # Get predicted label (caption)
    _, predicted_idx = output.max(1)
    predicted_caption = str(predicted_idx.item())
    return predicted_caption

def generate_response(prompt, image_path):
    # Generate text response
    text_response = generate_text(prompt)
    # Generate image caption
    image_caption = generate_image_caption(image_path)
    # Combine text and image captions into a single response
    response = f"Text Response: {text_response}\nImage Caption: {image_caption}"
    return response

# Example usage
prompt = "A cat is sitting on"
image_path = "cat_image.jpg"
response = generate_response(prompt, image_path)
print(response)
