import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import torchattacks

# Load pre-trained ResNet18 model with ImageNet weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Get class names
weights = ResNet18_Weights.IMAGENET1K_V1
class_names = weights.meta["categories"]

# Define preprocessing



preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image (assuming cat.jpg is in the current directory)
image = Image.open('cat.jpg').convert('RGB')

# Preprocess the image
input_tensor = preprocess(image).unsqueeze(0)
input_tensor.requires_grad = True  


# For gradient computation

# Original prediction
with torch.no_grad():
    output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    predicted_class = class_names[predicted_idx.item()]
    print(f'Original prediction: {predicted_class} (index: {predicted_idx.item()})')

# Target class index for "dog" (using Golden Retriever as an example, index 207)
target_idx = 207  # Golden Retriever
target_label = torch.tensor([target_idx], dtype=torch.long)

# Create PGD attack (targeted by passing target)
attack = torchattacks.PGD(model, eps=2, alpha=0.01, steps=20)

# Generate adversarial example
adversarial_input = attack(input_tensor, target_label)

# Prediction on adversarial example
with torch.no_grad():
    adv_output = model(adversarial_input)
    _, adv_predicted_idx = torch.max(adv_output, 1)
    adv_predicted_class = class_names[adv_predicted_idx.item()]
    print(f'Adversarial prediction: {adv_predicted_class} (index: {adv_predicted_idx.item()})')

# Save the adversarial image (denormalize first)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

adversarial_image = inv_normalize(adversarial_input.squeeze(0))
adversarial_image = torch.clamp(adversarial_image, 0, 1)
to_pil = transforms.ToPILImage()
adv_pil = to_pil(adversarial_image)
adv_pil.save('adversarial_cat.jpg')

print("Adversarial image saved as 'adversarial_cat.jpg'")