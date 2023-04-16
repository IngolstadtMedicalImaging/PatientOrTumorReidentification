from ast import Str
import random
import torch
import torchvision.transforms.functional as tf
from torchvision import transforms as t


class CustomCompose(object):
    def __init__(self, transforms):
        self.operations = transforms
        
    def __call__(self, image):
        for op in self.operations:
            image = op(image)
        return image


class Resize(object):
    def __init__(self, width: int, height: int):
        self.resize_image = t.Resize((width, height))
        
    def __call__(self, image: torch.Tensor, target):
        return self.resize_image(image), target


class RandomCrop(object):
    def __init__(self, minimum_size: float = .3):
        self.min = minimum_size

    def __call__(self, image: torch.Tensor):
        c, h, w = image.size()
        height, width = random.randint(int(h*self.min), h), random.randint(int(w*self.min), w)
        top, left = random.randint(0, h-height), random.randint(0, w-width)
        image = tf.crop(image, top, left, height, width)
        return image


class ToTensor(object):
    def __init__(self):
        self.operation = t.ToTensor()
        
    def __call__(self, image):
        image = self.operation(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self):
        self.operation = t.RandomVerticalFlip(p=0.5)

    def __call__(self, image):
        image = self.operation(image)
        return image
    
    
class RandomHorizontalFlip(object):
    def __init__(self):
        self.operation = t.RandomHorizontalFlip(p=0.5)

    def __call__(self, image):
        image = self.operation(image)
        return image


class ColorJitter(object):
    def __init__(self):
        self.operation = t.ColorJitter(brightness=.2, contrast=.2, hue=.2, saturation=.2)

    def __call__(self, image):
        image = self.operation(image)
        return image
    

class Greyscale(object):
    def __init__(self):
        self.operation = t.Grayscale(num_output_channels=3)

    def __call__(self, image):
        image = self.operation(image)
        return image
    

class GaussianBlur(object):
    def __init__(self):
        self.operation = t.GaussianBlur(kernel_size=(7, 13), sigma=(0.001, 2))

    def __call__(self, image):
        image = self.operation(image)
        return image


class Normalize(object):
    def __init__(self):
        self.operation = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, image):
        image = self.operation(image)
        return image


class PreProcess(object):
    def __init__(self):
        self.width, self.height = 400, 400
        self.operations = t.Compose([t.ToTensor(), t.Resize((self.width, self.height)),
                                     t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, image):
        image = self.operations(image)
        return image.unsqueeze(0)


class InvertNormalization(object):
    def __init__(self):
        self.invert = t.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    def __call__(self, image):
        return self.invert(image)
    

class ToPILImage(object):
    def __init__(self):
        self.operation = t.ToPILImage()
    
    def __call__(self, image):
        image = self.operation(image)
        return image

    
def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    x = torch.rand(3, 640, 480)
    y = torch.rand(3, 640, 480)
    crop = RandomCrop()
    print(crop(x, y)[0].size())