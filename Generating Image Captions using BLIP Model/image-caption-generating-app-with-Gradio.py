#Import Libraries/Tools
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#Function for Captioning Images
def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the image
    inputs = processor(raw_image, return_tensors="pt")

    # Generate a caption for the image
    out = model.generate(**inputs,max_length=50)

    # Decode the generated tokens to text
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

#Gradio Interface
iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

#Launch App
iface.launch()
