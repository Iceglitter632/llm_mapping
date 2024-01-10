

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class Dataset:
    def __init__(self, args, image_encoder):
        
        if(args.exp_type=="image"):
            print(f"\tDataset at: {args.dataset_path}")
            self.dataset = datasets.LSUN(root=args.dataset_path, classes=args.dataset_classes, transform=image_encoder.transform)
            
        