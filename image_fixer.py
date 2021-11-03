from PIL import Image
img = Image.open('starry.jpg')

img = img.resize((img.size[0]//2, img.size[1]//2))
img.save('starry.jpeg')