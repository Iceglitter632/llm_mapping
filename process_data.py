

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class Dataset:
    def __init__(self, args, image_encoder):
        
        # if(args.exp_type=="image"):
        print(f"\tDataset at: {args.dataset_path}")
        self.dataset = datasets.LSUN(root=args.dataset_path, classes=args.dataset_classes, transform=image_encoder.transform)
        self.dataset_len = len(self.dataset)
        self.train_ratio = args.train_val_split
        self.split_dataset()
        
    def split_dataset(self):
        train_len = int(self.dataset_len * self.train_ratio)
        val_len = self.dataset_len - train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])
        

class MusicDataset:
    def __init__(self, args, encoder):
        pass
    
    def split_dataset(self):
        pass
        
