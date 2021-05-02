import numpy
import copy
import math
from statistics import mean
import random
from PIL import Image

# returns a 2D array of RGB given a filename of a picture
# 2D array is a list of columns
def image_creation(fp):
    im = Image.open(fp, "r").convert("RGB")
    length = im.width
    height = im.height

    # arr is a 2D array of tuples containing the RGB value for every pixel
    arr = []
    for i in range(length):
        r = []
        for j in range(height):
            r.append(im.getpixel((i, j)))
        arr.append(r)
    
    return arr

# saves a 2D array as a picture file to the current directory
def save_img(arr, name, format):
    length = len(arr)
    height = len(arr[0])
    im = Image.new(format, (length, height))
    for i in range(length):
        for j in range(height):
            im.putpixel((i, j), arr[i][j])
    im.save(name)


# returns a 2D L array with the grayscale version of the original array
# dim refers to which column to start copying, set to 0 by default
# this optional argument is used for generating testing data
def grayscale(arr):
    length = len(arr)
    height = len(arr[0])

    gray = []
    for i in range(length):
        r = []
        for j in range(height):
            pix = arr[i][j]
            g_pix = int(.21 * pix[0] + .72 * pix[1] + .07 * pix[2])
            r.append(g_pix)
        gray.append(r)
    
    return gray


def basic_agent(fp):
    arr = image_creation(fp)
    gray = grayscale(arr)
    left_rgb = arr[:len(arr) // 2]
    left_l = gray[:len(gray) // 2]
    right_rgb = arr[len(arr) // 2:]
    right_l = gray[len(gray) // 2:]

# saves an image of all the center colors
def center_img(centers, name):
    arr = []
    for c in centers:
        col = [c] * 100
        for x in range(50):
            arr.append(col)
    
    save_img(arr, name, "RGB")
# distance between two rgb tuples
def dist(c, d):
    cr, cg, cb = c[0], c[1], c[2]
    dr, dg, db = d[0], d[1], d[2]
    return math.sqrt(2 * ((cr - dr) ** 2) + 4 * ((cg - dg) ** 2) + 3 * ((cb - db) ** 2))

def cluster(arr, k):
    # generate random colors as centers
    centers = sorted([tuple((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) for x in range(k)])
    clusters = [[] for x in range(k)]
    for i in range(50):
        prevcent = copy.deepcopy(centers)
        print("running lloyd")
        print(centers)
        center_img(sorted(centers), f"center{i}.png")
        for c in clusters:
            c.clear()
        # add rgb tuples to clusters
        for x in range(len(arr)):
            for y in range(len(arr[0])):
                pix = arr[x][y]
                closest = 0
                for i in range(1, len(centers)):
                    if dist(pix, centers[i]) < dist(pix, centers[closest]):
                        closest = i
                clusters[closest].append(pix)
        print([len(c) for c in clusters])
        # compute new centers by averaging clusters
        for i in range(len(clusters)):
            c = clusters[i]
            if len(c) == 0:
                centers[i] = tuple((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                continue
            rs = [pix[0] for pix in c]
            gs = [pix[1] for pix in c]
            bs = [pix[2] for pix in c]
            centers[i] = tuple((int(mean(rs)), int(mean(gs)), int(mean(bs))))
        if centers == prevcent:
            break
    return centers
    

arr = image_creation("fruit_salad.png")
#gray = grayscale(arr, dim=250)
#save_img(gray, "copy.png", "L")
basic_agent("fruit_salad.png")
cluster(arr, 5)