import os
from PIL import Image

def concatenate_images(low_dir, high_dir, output_dir):
    """
    Concatenate images from two directories side by side and save them with sequential names (1, 2, 3, ...) to an output directory.

    Args:
        low_dir (str): Path to the directory containing low-light images.
        high_dir (str): Path to the directory containing high-light images.
        output_dir (str): Path to the directory where concatenated images will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of images from both directories
    low_images = sorted([f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    high_images = sorted([f for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Ensure both directories have the same number of images
    if len(low_images) != len(high_images):
        print("Error: The number of images in the two directories does not match.")
        return

    # Loop through and concatenate images
    for i, (low_img_name, high_img_name) in enumerate(zip(low_images, high_images), start=1):
        # Open images
        low_img = Image.open(os.path.join(low_dir, low_img_name))
        high_img = Image.open(os.path.join(high_dir, high_img_name))

        # Check if images are of the same height, resize if needed
        if low_img.height != high_img.height:
            high_img = high_img.resize((high_img.width, low_img.height))

        # Create a new image by concatenating both images side by side
        new_img = Image.new('RGB', (low_img.width + high_img.width, low_img.height))
        new_img.paste(low_img, (0, 0))
        new_img.paste(high_img, (low_img.width, 0))

        # Save the concatenated image with a sequential name (e.g., 1.jpg, 2.jpg, 3.jpg)
        new_img_name = f'{i}.jpg'
        new_img.save(os.path.join(output_dir, new_img_name))

    print("Concatenation complete!")

if __name__ == "__main__":
    # Define your directories here
    low_light_directory = 'F:/kalyani/lol_dataset/eval15/low'   # Replace with your low-light images directory
    high_light_directory = 'F:/kalyani/lol_dataset/eval15/high' # Replace with your high-light images directory
    output_directory = 'F:/kalyani/lol_dataset/test'   # Replace with your desired output directory

    # Run the concatenation
    concatenate_images(low_light_directory, high_light_directory, output_directory)
