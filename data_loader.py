from torchvision import datasets, transforms
nltk.download('punkt')
from imageio import imread
from PIL import Image
import torch.utils.data as data


class ImagecaptionDataset(data.Dataset):
  def __init__(self,root,vocab, image_ids, text, transform = None):
    self.image_ids = image_ids
    self.text = text
    self.vocab = vocab
    self.transform = transform
    self.root = root
  def __getitem__(self, index):
    img = self.image_ids[index]
    caption = self.text[img][0]
    image = Image.open(self.root +img).convert('RGB')
    image = image.resize((256, 256))

    if self.transform is not None:
      image = self.transform(image)
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    caption = []
    caption.append(self.vocab('<start>'))
    caption.extend([self.vocab(cap) for cap in tokens])
    caption.append(self.vocab('<end>'))

    target = torch.Tensor(caption)

    return image, target
  
  def __len__(self):
    return len(self.image_ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
      end = lengths[i]
      targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, vocab,  image_ids,text, batch_size, shuffle, num_workers, transform):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    flickr = ImagecaptionDataset(root,vocab, image_ids, text, transform)
    data_loader = torch.utils.data.DataLoader(dataset=flickr, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader