import pandas as pd
import os
import random
import PIL.Image
import PIL.ImageDraw

df = pd.read_csv('./Brixia_for_bbox.csv')

def show_img(index, path_to_image):
    """show a random CXR image from BrixIA dataset.

    Args:
        index: index of image in dataset
        path_to_image: path where the images are saved
    """
    path = os.path.join(path_to_image,df.loc[index,'Filename'])
    im = PIL.Image.open(path)
    im = im.convert('RGB')
    draw = PIL.ImageDraw.Draw(im)

    if (df.loc[index,'A']>1):
        draw.rectangle([40,20,80,60],outline="#FF0000",width=2)

    if (df.loc[index,'B']>1):
        draw.rectangle([40,95,80,135],outline="#FF0000",width=2)

    if (df.loc[index,'C']>1):
        draw.rectangle([40,170,80,210],outline="#FF0000",width=2)

    if (df.loc[index,'D']>1):
        draw.rectangle([152,20,192,60],outline="#FF0000",width=2)

    if (df.loc[index,'E']>1):
        draw.rectangle([152,95,192,135],outline="#FF0000",width=2)

    if (df.loc[index,'F']>1):
        draw.rectangle([152,170,192,210],outline="#FF0000",width=2)

    im.show()

#the path to be changed
path_to_image = r'D:\MLMI\osfstorage\data\brixia\images'

if __name__ == '__main__':
    index = random.randint(0,df.shape[0])
    show_img(index,path_to_image)
