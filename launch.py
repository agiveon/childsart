import gradio as gr
import torch
from torchvision import transforms
import requests
from PIL import Image

model = torch.load('ChildsArtModelMOBILENETV2.pt')
model.eval()

labels = ['ABSTRACT ART', 'CONTEMPORARY ART', 'CUBISM', 'FANTASY ART', 'GRAFFITI', 'IMPRESSIONISM', 'POP ART', 'SURREALISM']

def classify_dino(img):

  img = transforms.Resize((224,224))(img)
  img = transforms.ToTensor()(img)
  img = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(img)

  with torch.no_grad():
    new_pred = model(img.view(1,3,224,224)).argmax()
  return labels[new_pred.item()]

image = gr.inputs.Image(type='pil',image_mode="RGB")
label = gr.outputs.Label()
title = 'What type of artist is your child?'
<<<<<<< HEAD
description = 'Is your child more of an impressionist or contemporary art kinda artist? A surrialist maybe? Snap a picture of your child\'s art and find out what genre their art is!'

sample_images = [["ex1.JPG"],['ex2.JPG'],['ex3.JPG'],['ex4.JPG']]

=======
description = 'Is your child more of an impressionist or contemporary? Surrialist maybe? Snap a picture of your child\'s art and find out what genre is their art!'
>>>>>>> 24d39d2491a99e541f7e9efadb25791f7ec84717

gr.Interface(fn=classify_dino, inputs=image, outputs=label, capture_session=True, title=title, description=description).launch(share=True)
