import os
from skimage import io
from skimage.transform import resize
from functions import *
from skimage import draw
from tqdm import trange, tqdm

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

imagesize = 640

#to save image after drawing boxes.
savepath = './boxes/'

#imagelist = os.listdir('./decoder/images/')
imagelist = ['200022050-00005_1.jpg']

for file in tqdm(imagelist):
    #Just give image file and csv file location here-----------------------
    #imagepath = './testdata/200021853/images/'
    imagepath = "./traindata/200022050/images/"
    imagefile = file
    
    #Either GT or prediction csv. It will detect by itself.
    
    csvfile = './traindata/200022050/200022050_coordinate.csv'
    #csvfile = './decoder/cluster/' + file[0:-4] + ".csv"
    
    
    
    #Open file and resize
    im = Image.open(imagepath + imagefile)
    width, height = im.size
    rx = width/imagesize
    ry = height/imagesize
    #im = Image.open("./boxes/200021853-00010_2gt.jpg")
    #im = im.resize((imagesize, imagesize), Image.ANTIALIAS)
    
    df = pd.read_csv(csvfile)
    coordinate = pd.DataFrame()
    
    #checking if csv is GT or Prediction
    
    
    if 'Width' in df.columns:
        csv = df[df['Image'] == imagefile[:-4]] 
        coordinate['x'] = csv['X']
        coordinate['y'] = csv['Y']
        coordinate['w'] = csv['Width']
        coordinate['h'] = csv['Height']
    '''
    else:
        coordinate['x'] = df['x']
        coordinate['y'] = df['y']
        coordinate['w'] = [5] * len(df['x'])
        coordinate['h'] = [5] * len(df['x'])
    '''
    
    
    drawbox(im, coordinate)
    
    im.show()
    im.save(savepath + imagefile)

   
        
        
        
        
        
        
        
        



