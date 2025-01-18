import os
from PIL import Image

# Input and output paths
input_folder = "./images/"  # Path to the original image folder
output_folder = "./cropped_datasets"  # Path to the output folder
sizes = [(256, 256), (512, 512), (1024, 1024), (128, 128), (640, 640), (320, 320)]  # Six cropping sizes

def crop_and_save_images(input_folder, output_folder, sizes):
    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through all image files
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.JPEG', '.png')):  # Filter image files
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)
            
            for size in sizes:
                # Calculate the cropping center position
                width, height = img.size
                crop_width, crop_height = size
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                
                # Resize the image if the cropping size exceeds the original image size
                if crop_width > width or crop_height > height:
                    img_resized = img.resize(size, Image.Resampling.LANCZOS)
                else:
                    img_resized = img.crop((left, top, right, bottom))
                
                # Output file path and subfolder
                size_folder = os.path.join(output_folder, f"{size[0]}x{size[1]}")
                if not os.path.exists(size_folder):
                    os.makedirs(size_folder)
                
                output_path = os.path.join(size_folder, filename)
                img_resized.save(output_path)
                print(f"Saved: {output_path}")

# Run the function
crop_and_save_images(input_folder, output_folder, sizes)