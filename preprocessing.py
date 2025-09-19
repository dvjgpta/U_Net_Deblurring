import os
import random
from PIL import Image


from pathlib import Path
def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

image_path=Path("/mnt/DATA/EE22B013/Btech_project/U_NET/patches")
#print(walk_through_dir(image_path))

train_dir_sharp=image_path / "DIV2K_train_HR"
train_dir_blurred=image_path / "DIV2K_train_HR_blurred"

test_dir_sharp= image_path/"val"/"DIV2K_valid_HR_256"
test_dir_blurred= image_path/"val"/"DIV2K_valid_HR_blurred_256"

#print(train_dir_sharp)

random.seed(13)

# getting all image path in any .png format
image_path_list=list(image_path.glob("*/*/*.png")) 

# getting a random image path in any .png format
random_image_path=random.choice(image_path_list)

#open image
img=Image.open(random_image_path)

#printmetadata

# print(f"Random Image Path:{random_image_path}")
# print(f"Img Height:{img.height}")
# print(f"Image Width:{img.width}")
# img.show()

