import imageio
import os

images = []
image_list = os.listdir('./anime_3d')
image_list.sort(key=lambda x: int(x.split('.')[0][:]))
# image_list = sorted(image_list)
# image_list =
for filename in image_list:
    print(filename)
    image_path = os.path.join('./anime_3d',filename)
    images.append(imageio.imread(image_path))
imageio.mimsave('3d.gif',images,duration=0.5)