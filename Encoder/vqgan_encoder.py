import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torchvision import transforms

from dall_e import map_pixels
from vqgan_jax.modeling_flax_vqgan import VQModel

from .base import InputEncoder

class VQGanEncoder(InputEncoder):
    def __init__(self, args):
        super(VQGanEncoder, self).__init__(args)
        checkpoint = "dalle-mini/vqgan_imagenet_f16_16384"
        self.vqmodel = VQModel.from_pretrained(checkpoint)
        self.vocab_len = self.vqmodel.config.n_embed
        self.image_size = args.image_size
        
        self.transform = transforms.Compose([
            transforms.Lambda(self.resize_and_crop),
            transforms.ToTensor(),
            transforms.Lambda(self.modified_map_pixels)
        ])

    def get_ground_truth(self, images):
        images = images.permute(0,2,3,1)
        _, ground_truth_tokens = self.vqmodel.encode(images)
        ground_truth_tokens = torch.from_dlpack(ground_truth_tokens)
        ground_truth_tokens = ground_truth_tokens.to(torch.int64).to(self.device)
        
        return ground_truth_tokens
    
    def get_onehot(self, ground_truth_tokens):
        one_hot_image_tokens = F.one_hot(ground_truth_tokens, num_classes=self.vocab_len).float()
        return one_hot_image_tokens
    
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