# Brain-tumor-model
Basic brain tumor classifying Vision Transformer model

# Info and purpose:
This is a basic model that I built to just showcase the skills I have on image classification and transformers,this model is able to detect brain tumour in 4 classifications: Glioma, Meningioma, Pituitary and No Tumour if nothing is detected to an accuracy of 94-95%. It takes in MRI grayscale images

## features
optimizer: Adam,
loss function: cross entropy loss,
epochs: 100,
architecture: transformer architecture (Vision Transformer),
images: MRI scanned gray-scale images of brain

## how to check:
Make sure you have pytorch including torchvision is installed in your device or else use this command in cmd to install it for latest version in cpu:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

There are 2 files brain-tumor(full) and brain-tumour. If you want to validate the models' performace by testing with a single image to see what this model detects, you need to download the brain-tumour.py and brain-tumour.pth to your local device and make sure both of these files are in same directory. For retraining this model from scratch with your own datasets or want to see the full architecture of how I trained you just need to download the brain-tumour(full).py .

To test the model with singular image, follow this procedure:
->Go to your local files to the image you want to select, then right click the image, then click on "copy as path" or ctrl+Shift+c
->Paste the path in line 63 of brain-tumour.py which is "img_path" variable and make sure that your path is in quotes and replace single backslashes (\) with double backslashes (\\) or use a raw string like r"your_path_here" to avoid path issues on Windows.
