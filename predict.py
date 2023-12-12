# Imports here
import argparse
import json
import torch
from torchvision import transforms, models
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('input', type = str, help = 'path to the input image')
parser.add_argument('checkpoint', type = str, help = 'checkpoint file')
parser.add_argument('--top_k', type = int, default = 5, help = 'set the top K probability class')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'category to names file')
parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU or not')

args = parser.parse_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


def load(filepath):
    """loads model from a checkpoint"""

    checkpoint = torch.load(filepath)
    model = eval('models.' + checkpoint['arch'] +'(pretrained=True)')
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
    '''
    
    pil_image = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    img_tensor = transform(pil_image)
    return img_tensor.unsqueeze(0)

def predict(image_path, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = load(model)
    model.to(device)
    image = image_path.to(device)
    model.eval()
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    with torch.no_grad():
        ps = torch.exp(model.forward(image))
    top_p, top_idx = ps.topk(topk, dim=1)
    top_class = [model.idx_to_class[idx] for idx in top_idx.tolist()[0]]
    top_p = top_p.tolist()[0]
    return top_p, top_class

def display(top_p, top_class):
    """Displays image and top class plobabilities"""
    
    top_cat = [cat_to_name[cls] for cls in top_class]
    
    if args.top_k == 1:
        print(f"Most likely category and it's probability is: {tuple(zip(top_cat, top_p))[0]}")
    else:
        print(f"Top {args.top_k} most likely categories and their proabilities are:")
        for i, j in zip(top_cat, top_p):
            print(f"{i}, {j}")

model = args.checkpoint
image = args.input
top_p, top_class = predict(process_image(image), model)
display(top_p, top_class)
