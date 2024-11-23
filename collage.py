import os
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch


# Define the directories
input_folder = 'F:/kalyani/lol_dataset/eval15/low'
ground_truth_folder = 'F:/kalyani/lol_dataset/eval15/high'
result_folder = 'C:/Users/Admin/Desktop/restormer/result/inpaint'
output_folder = 'C:/Users/Admin/Desktop/restormer/outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define constants
image_size = (512, 512)
spacing = 20
margin =  75 # Margin for serial number
label_height = 40  # Height for the label space
collage_width = 3 * image_size[0] + 2 * spacing + 2 * margin
collage_height = image_size[1] + 2 * margin + label_height

# Font settings (adjust font size and bold if necessary)
try:
    font = ImageFont.truetype("arialbd.ttf", 24)  # Bold Arial font on Windows
except IOError:
    font = ImageFont.load_default()  # Default font if Arial Bold is unavailable

# Function to center text
def draw_centered_text(draw, text, image_width, y_position, margin):
    text_width, _ = draw.textsize(text, font=font)
    x_position = margin + (image_width - text_width) // 2
    draw.text((x_position, y_position), text, font=font, fill="black")

# Function to create a collage with labels
def create_collage(input_img, gt_img, result_img, save_path, collage_number):
    # Create a white background image
    collage = Image.new('RGB', (collage_width, collage_height), 'white')
    
    # Create a drawing object for adding text
    draw = ImageDraw.Draw(collage)
    
    # Add serial number at the top
    serial_text = f"Image {collage_number}"
    draw_centered_text(draw, serial_text, collage_width, margin // 2 - 20, 0)
    
    # Open and resize the images
    input_img = Image.open(input_img).resize(image_size)
    gt_img = Image.open(gt_img).resize(image_size)
    result_img = Image.open(result_img).resize(image_size)
    
    # Calculate image positions
    x_input = margin
    x_gt = margin + image_size[0] + spacing
    x_result = margin + 2 * (image_size[0] + spacing)
    y_image = margin + label_height // 2
    
    # Paste images onto the collage
    collage.paste(input_img, (x_input, y_image))
    collage.paste(gt_img, (x_gt, y_image))
    collage.paste(result_img, (x_result, y_image))
    
    # Add labels below the images, centered
    draw_centered_text(draw, "Input", image_size[0], y_image + image_size[1] + 5, x_input)
    draw_centered_text(draw, "Ground Truth", image_size[0], y_image + image_size[1] + 5, x_gt)
    draw_centered_text(draw, "Result", image_size[0], y_image + image_size[1] + 5, x_result)
    
    # Save the collage
    collage.save(save_path)

# Get the list of images in each folder
input_images = sorted(os.listdir(input_folder))
ground_truth_images = sorted(os.listdir(ground_truth_folder))
result_images = sorted(os.listdir(result_folder))

# Check if the number of images matches in all folders
assert len(input_images) == len(ground_truth_images) == len(result_images), "Mismatch in number of images!"

# Create collages
for i, (input_img, gt_img, result_img) in enumerate(zip(input_images, ground_truth_images, result_images)):
    input_img_path = os.path.join(input_folder, input_img)
    gt_img_path = os.path.join(ground_truth_folder, gt_img)
    result_img_path = os.path.join(result_folder, result_img)
    
    save_path = os.path.join(output_folder, f'{i+1}.jpg')
    
    create_collage(input_img_path, gt_img_path, result_img_path, save_path, i + 1)

print(f"Collages saved in {output_folder}")