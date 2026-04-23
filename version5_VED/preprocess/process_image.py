import torch
import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import open_clip
import torch.nn as nn
import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BlurringPipeline:
    def __init__(self,blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(img_blur)
    
class Make_dataset(nn.Module):
    def __init__(self,):
        super().__init__()

        self.vlmodel, _, _ = open_clip.create_model_and_transforms(  
            'RN50',  
            # pretrained= 'openai'
            pretrained= r"./data/open_clip_weights/RN50/open_clip_pytorch_model.bin" 
        ) 

        self.vlmodel = self.vlmodel
        # self.freeze()
        self.vlmodel.eval()

        self.blur_transform = {}
        for kernel,tag in zip([1,3,9,15,21,27,33,39,45,51,57,63],['1','3','9','15','21','27','33','39','45','51','57','63']):
            self.blur_transform[tag] = BlurringPipeline(kernel)
        process_term = [transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))] 
        self.process_transform = transforms.Compose(process_term)

    def freeze(self):
        for param in self.vlmodel.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def ImageEncoder(self,images,blur_transform=None):
        if blur_transform == None:
            blur_transform = self.blur_transform
        self.vlmodel.eval()

        set_images = images
        # set_images.sort()
        print(len(set_images))
        batch_size = 128
        image_features_list = []
        for i in tqdm.tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]

            device = next(self.vlmodel.parameters()).device
            # print(batch_images)
            ele = []
            for img in batch_images:
                p_img = self.process_transform(blur_transform(Image.open(os.path.join(base_path,img)).convert("RGB").resize((224,224))))
                # print(p_img.shape)
                ele.append(p_img)
              
            image_inputs = torch.stack(ele).to(device)
            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features/batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]:image_features[i].float().cpu() for i in range(len(set_images))}

        return image_features_dict


if __name__ == '__main__':
    model = Make_dataset().to(device)
    base_path = r'./data/things-eeg/Image_set'
    train_features_save_path = './data/things-eeg/Image_feature/MultiBlur_RN50_train.pt'
    test_features_save_path  = './data/things-eeg/Image_feature/MultiBlur_RN50_test.pt'


    # Load the image paths
    # train_paths = np.load(f'./preprocess/things_eeg_img_paths.npz', allow_pickle=True)['train_paths']
    # test_paths  = np.load(f'./preprocess/things_eeg_img_paths.npz', allow_pickle=True)['test_paths']

    train_saved_features = {}
    train_paths = []
    for root, dirs, files in os.walk(os.path.join(base_path,'train_images')):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                train_paths.append(os.path.join(root.split('Image_set\\')[-1], file).replace('\\','/'))

    for keys in ['1','3','9','15','21','27','33','39','45','51','57','63']:
        train_saved_features[keys] = model.ImageEncoder(train_paths,blur_transform=model.blur_transform[keys])
    

    save_dir = os.path.dirname(train_features_save_path)
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    torch.save(train_saved_features, train_features_save_path)


    test_paths = []
    for root, dirs, files in os.walk(os.path.join(base_path,'test_images')):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                test_paths.append(os.path.join(root.split('Image_set\\')[-1], file).replace('\\','/'))

    test_saved_features = {}
    for keys in ['1','3','9','15','21','27','33','39','45','51','57','63']:
        test_saved_features[keys] = model.ImageEncoder(test_paths,blur_transform=model.blur_transform[keys])
    torch.save(test_saved_features, test_features_save_path)