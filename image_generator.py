import requests
from PIL import Image
import os.path
import cv2 

save_path = 'C:/wamp64/www/CS424-G1/data/image_classification_smu/validation/non-smu'

def img_requests(txt, save_folder):
     response=requests.get("https://source.unsplash.com/random{0}".format(txt))
    # Ensure the save folder exists, create it if not
     if not os.path.exists(save_folder):
          print(f'Creating folder {save_folder}')
          os.makedirs(save_folder)

     img_path = os.path.join(save_folder, f'image_{i}.jpg')  # Save images with a unique name
     # with open(img_path, 'wb') as file:
     #      file.write(response.content)
     #      print("writing file")
          
     img = Image.open(img_path)

     # return img_path

print("""Please provide an option for Image
     1. HD Random Picture
     2. FHD Random Picture
     3. 2K Random Picture
     4. 4k Random Picture
     5. Picture with User Provided Keyword """)

ans=input()
image_list = []
# completeName = os.path.join(save_path, name_of_file+".txt")         

# file1 = open(completeName, "w")
if 'one' in ans or '1' in ans:
     print("Please wait while we fetch the images from our database.")
     for i in range(100):
        img = img_requests('/1280x720')
        image_list.append(img)

elif 'two' in ans or '2' in ans:
     print("Please wait while we fetch the images from our database.")
     for i in range(100):
        img = img_requests('/1920x1080')
        image_list.append(img)
elif 'three' in ans or '3' in ans:
     print("Please wait while we fetch the images from our database.")
     for i in range(100):
        img = img_requests('/2048x1080')
        image_list.append(img)
elif 'four' in ans or '4' in ans:
     print("Please wait while we fetch the images from our database.")
     for i in range(100):
        img = img_requests('/4096x2160')
        image_list.append(img)
elif 'five' in ans or '5' in ans:
     print("Please enter the keyword you want to get a random image of.")
     st=input()
     st="?"+st
     print("Please wait while we fetch the images from our database.")
     for i in range(500):
        img = img_requests(st, save_path)
        image_list.append(img)

else:
     print("Please provide a valid input.")

print("Done", len(image_list))
print(image_list)
 