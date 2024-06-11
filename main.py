import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F

from model import CSD_CLIP, convert_state_dict

from PIL import Image

# init model
model = CSD_CLIP("vit_large", "default")

# load model
model_path = "models/checkpoint.pth"
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(state_dict, strict=False)
model = model.cuda()

# normalization
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

# style image
image = preprocess(Image.open("1.jpg")).unsqueeze(0).to("cuda")
_, content_output, style_output = model(image)

# another style image
image1 = preprocess(Image.open("4.jpg")).unsqueeze(0).to("cuda")
_, content_output1, style_output1 = model(image1)

sim = style_output@style_output1.T
print(sim)
