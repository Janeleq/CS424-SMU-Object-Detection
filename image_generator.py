import requests
from PIL import Image

def img_requests(txt):
    response=requests.get("https://source.unsplash.com/random{0}".format(txt))
    file=open('image.jpg','wb')
    file.write(response.content)
    img=Image.open(r"image.jpg")
    img.show()
    file.close()

print("""Please provide an option for Image
     1. HD Random Picture
     2. FHD Random Picture
     3. 2K Random Picture
     4. 4k Random Picture
     5. Picture with User Provided Keyword """)

ans=input()
image_list = []

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
     for i in range(100):
        img_requests(st)
else:
     print("Please provide a valid input.")

print("Done", len(image_list))
print(image_list)
 