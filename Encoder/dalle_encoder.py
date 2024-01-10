import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torchvision import transforms
from dall_e import map_pixels, load_model

class DalleEncoder:
    def __init__(self, args):
        self.encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", args.device)
        self.codebook_len = self.encoder.vocab_size
        self.device = args.device
        self.image_size = args.image_size
        
        self.transform = transforms.Compose([
            transforms.Lambda(self.resize_and_crop),
            transforms.ToTensor(),
            transforms.Lambda(self.modified_map_pixels)
        ])
        
    def encode(self, image):
        image = image.to(self.device)
        z_logits = self.encoder(image)
        z = torch.argmax(z_logits, axis=1)
        # z_test = z.reshape(z.size(0), -1)
        z_ = F.one_hot(z, num_classes=self.codebook_len).permute(0, 3, 1, 2).float()
        return z_
    
    def resize_and_crop(self, img):
        # Resize while maintaining aspect ratio and center crop
        s = min(img.size)
        r = self.image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [self.image_size])
        return img

    def modified_map_pixels(self, img):
        # Add a batch dimension, apply map_pixels, and then remove the batch dimension
        img = img.unsqueeze(0)
        img = map_pixels(img)
        return img.squeeze(0)