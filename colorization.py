import numpy
import copy
import math
from matplotlib import pyplot as plt
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

# recolor the image with rep colors
def recolor(arr, reps):
    recol = copy.deepcopy(arr)
    for i in range(len(recol)):
        for j in range(len(recol[0])):
            pix = recol[i][j]
            closest = 0
            for k in range(1, len(reps)):
                if dist_rgb(pix, reps[k]) < dist_rgb(pix, reps[closest]):
                    closest = k
            recol[i][j] = copy.deepcopy(reps[closest])
    return recol

def basic_agent(fp):
    arr = image_creation(fp)
    gray = grayscale(arr)
    left_rgb = arr[:len(arr) // 2]
    left_l = gray[:len(gray) // 2]
    right_rgb = arr[len(arr) // 2:]
    right_l = gray[len(gray) // 2:]

    precision = 15
    comps = precision + 1
    # return the most representative colors of the left half
    reps = cluster(left_rgb, precision)

    # recolor the left rgb with rep colors
    left_recol = recolor(left_rgb, reps)
    
    right_recol = copy.deepcopy(right_rgb)

    # now create a list of all patches in the test data
    test_patches = []
    size = 3
    rad = size // 2
    for i in range(rad, len(right_l) - rad):
        for j in range(rad, len(right_l[0]) - rad):
            test_patches.append(patch(right_l, (i, j), size))

    # then create a list of all patches in the training data
    train_patches = []
    for i in range(rad, len(left_l) - rad):
        for j in range(rad, len(left_l[0]) - rad):
            train_patches.append(patch(left_l, (i, j), size))
    for i in range(len(test_patches)):
        p = test_patches[i][0]
        center = test_patches[i][1]
        most_similar = []
        for j in range(len(train_patches)):
            n = patch_similarity(p, train_patches[j][0])
            if len(most_similar) < comps:
                most_similar.append((n, j))
                continue
            most_similar.sort()
            closest = most_similar[-1][0]
            if n < closest:
                most_similar[-1] = (n, j)
        best_indices = [s[1] for s in most_similar]
        best_center_coords = [train_patches[i][1] for i in best_indices]
        best_colors = [left_recol[center[0]][center[1]] for center in best_center_coords]

        # convert list of best_colors to a dictionary
        counts = dict()
        for c in best_colors:
            counts[c] = counts.get(c, 0) + 1

        # find most representative color
        largest = list(counts.keys())[0]
        for c in counts.keys():
            # majority representation
            if counts[c] > counts[largest]:
                largest = c

        # if there is tie, identify all ties
        ties = []
        for c in counts.keys():
            if counts[c] == counts[largest]:
                ties.append(c)
        
        # check which color in ties is the best
        for c in best_colors:
            if c in ties:
                largest = c
                break
        
        right_recol[center[0]][center[1]] = largest
    
    save_img(right_recol, f"{precision}-colorized_{fp}", "RGB")

# returns a tuple: first index is a list of the values in a patch with center and size given, second index is the tuple coordinates of the center
# size must be an odd number
# works on both grayscale and rgb arrays
def patch(arr, center, size):
    if size % 2 == 0:
        return None
    x, y = center[0], center[1]
    rad = size // 2
    # if center is out of bounds, don't return a list
    if x - rad < 0 or y - rad < 0 or x + rad > len(arr) or y + rad > len(arr[0]):
        return None
    a = x - rad
    b = y - rad
    vals = []
    for i in range(size):
        for j in range(size):
            vals.append(arr[a + i][b + j])
    return (vals, center)
    
# saves an image of all the center colors
def center_img(centers, name):
    arr = []
    for c in centers:
        col = [c] * 100
        for x in range(50):
            arr.append(col)
    save_img(arr, name, "RGB")

# euclidean distance between two rgb tuples
def dist_rgb(c, d):
    cr, cg, cb = c[0], c[1], c[2]
    dr, dg, db = d[0], d[1], d[2]
    return 2 * ((cr - dr) ** 2) + 4 * ((cg - dg) ** 2) + 3 * ((cb - db) ** 2)

# quantifies the similarity of two patches
# currently only supports grayscale
def patch_similarity(p, q):
    diff = 0
    for i in range(len(p)):
        diff += (p[i] - q[i]) ** 2
    return diff

# run k-means clustering on the arr to determine the most representative colors
def cluster(arr, k):
    # generate random colors as centers
    centers = sorted([tuple((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) for x in range(k)])
    clusters = [[] for x in range(k)]
    for i in range(10):
        prevcent = copy.deepcopy(centers)
        for c in clusters:
            c.clear()
        # add rgb tuples to clusters
        for x in range(len(arr)):
            for y in range(len(arr[0])):
                pix = arr[x][y]
                closest = 0
                for i in range(1, len(centers)):
                    if dist_rgb(pix, centers[i]) < dist_rgb(pix, centers[closest]):
                        closest = i
                clusters[closest].append(pix)
        # compute new centers by averaging clusters
        for i in range(len(clusters)):
            c = clusters[i]
            if len(c) == 0:
                while True:
                    a = tuple((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                    if a not in centers:
                        centers[i] = a
                        break
                continue
            rs = [pix[0] for pix in c]
            gs = [pix[1] for pix in c]
            bs = [pix[2] for pix in c]
            centers[i] = tuple((int(mean(rs)), int(mean(gs)), int(mean(bs))))

        # if there is no improvement in centers, return
        if centers == prevcent:
            break
    return centers

# compares two image arrays and finds the euclidean distance between them
def img_similarity(a, b):
    dist = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            dist += dist_rgb(a[i][j], b[i][j])
    return dist

fp = "landscape.png"
arr = image_creation(fp)
print(f"{len(arr)} x {len(arr[0])}")
#save_img(grayscale(arr), f"gray_{fp}", "L")
#save_img(recolor(arr, cluster(arr, 29)), "recol_dua.png", "RGB")
#save_img(recolor(arr, cluster(arr, 5)), f"recol_{fp}", "RGB")
basic_agent(fp)
#center_img(cluster(arr, 5), f"reps{fp}")
#driver()