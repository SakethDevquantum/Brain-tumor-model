'''this file is for validation only, choose the image path from your local device and paste it in line
63, download this file along with the pth file to validate this code
'''

import torch
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import torch.nn as nn
from PIL import Image

device="cuda" if torch.cuda.is_available() else "cpu"

class Vit(nn.Module):
    def __init__(self, img_size=128, patch_size=8, depth=6, num_classes=4,
                in_channels=1, dim=128, mlp=512, nheads=4, dropout=0.1):
        super().__init__()
        assert(img_size%patch_size==0), "image size should be divisible by patch size"
        self.patch_size=patch_size
        self.dim=dim
        num_patches=(img_size//patch_size)**2
        self.patch_embed=nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token=nn.Parameter(torch.randn(1,1,dim))
        self.pos_embed=nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.dropout=nn.Dropout(dropout)
        encoder_layer=nn.TransformerEncoderLayer(d_model=dim, nhead=nheads, dim_feedforward=mlp,
                                                      dropout=dropout, activation='gelu', batch_first=True)
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm=nn.LayerNorm(dim)
        self.mlp_head=nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B=x.size(0)
        x=self.patch_embed(x)
        x=x.flatten(2).transpose(1,2)
        cls_token=self.cls_token.expand(B, -1, -1)
        x=torch.cat((cls_token,x), dim=1)
        x=x+self.pos_embed
        x=self.dropout(x)
        x=self.transformer(x)
        cls_out=x[:,0]
        avg_out=x[:,1:].mean(dim=1)
        x=self.norm(cls_out+avg_out)
        return self.mlp_head(x)

if __name__ == "__main__":
    model=Vit().to(device)
    optimizer=torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss()
    saving_path="brain-tumour.pth"
    start_epoch=0
    epochs=100
    torch.cuda.empty_cache()
    if os.path.exists(saving_path):
        checkpoint=torch.load(saving_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch=checkpoint["epoch"]+1
        print("pth successfully loaded")
        model.eval()
        img_path=""# paste image path from your local files to test the model with a custom image after training
        if os.path.exists(img_path): 
            transformen=transforms.Compose([
                transforms.Resize((128,128)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
            img=Image.open(img_path)
            img=transformen(img)
            img=img.unsqueeze(0)
            img=img.to(device)

            with torch.inference_mode():
                output=model(img)
                _,preds=torch.max(output,1)
            if(preds.item()==0):
                print("This is glioma")
            elif(preds.item()==1):
                print("this is meningioma")
            elif(preds.item()==3):
                print("this is pituitary")
            else:
                print("there seems to be no tumor here")
        else:
            print("choose a valid image path")
    else:
        print("Error retrieving pth")
