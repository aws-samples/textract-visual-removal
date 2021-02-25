from pdf2image import convert_from_bytes
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def show_page(page, color=False):
    plt.figure(figsize=(15,10))
    if color:
        plt.imshow(page)
    else:
        plt.imshow(page, cmap=plt.get_cmap('gray'))
    plt.show()
    
    
def plot_aggregation(y):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(y)), y)
    plt.xlabel('axis index')
    plt.ylabel('average pixel value')
    plt.show()


def rgb2gray(rgb):
    """
    Converts color image to gray image. 
    
    Arguemnts:
        rgb (np.array): color image
        
    Return:
        (np.array): gray image
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def max_sliding_window(x, w_size=5, fun=max):
    """
    Smoothing Extract segments with no text across one of the axis. 
    
    Arguemnts:
        black_dist (np.array): 1-D series with the black pixels distribution for one of the axis.
        w_size (int): Size of the rolling window to smooth concentrations of black text.
        fun (float): Smoothing function.
        
    Return:
        (list): 1-D series with smoothed black pixels distribution for one of the axis.
    """
    if w_size>1:
        N = len(x)
        w = w_size//2
        out = np.array([fun(x[max(0,i-w):i+w]) for i in range(len(x))])
    else:
        out = x
    return out


def get_candidate_segments(black_dist):
    """
    Get distinct segments based on black pixels distribution.
    
    Arguemnts:
        black_dist (float): Array with the black pixels in an axis.
        
    Return:
        (list): Distinct segments indices.
    """
    
    # differentiate chunks 
    changes_in_dit = [0] +  [1 
                             if (black_dist[i]==0. and black_dist[i+1]!=0.) or (black_dist[i-1]!=0. and black_dist[i]==0.) 
                             else 0 
                             for i in range(1, len(black_dist)-1)
                            ] + [0]
    
    # get indices
    idxs = [0] + np.where(np.array(changes_in_dit)==1.)[0].tolist() + [len(changes_in_dit)-1]
    segments = [
        (idxs[i],idxs[i+1]) for i in range(len(idxs)-1)
    ]
    return segments


def filter_segments(segments, black_dist, black_min_thr, black_max_thr):
    """
    Filter segment and return segments that can potentially contain images. 
    
    Arguemnts:
        segments (list): List of segments. 
        black_dist (float): Array with the smoothed distribution of black pixels in an axis.
        black_min_thr (int): Lower threshold of the average pixel for text regions.
        black_max_thr (int): Upper threshold of the average pixel for text regions.
        
    Return:
        (list): filtered axis segments.
    """
    candidates = []
    for m, M in segments:
        if np.mean(black_dist[m:M]) < black_min_thr:
            candidates.append((m, M))
        elif np.mean(black_dist[m:M]) > black_max_thr:
            candidates.append((m, M))
    return candidates


def get_segments(img, orientation, black_min_thr, black_max_thr, window_size):
    """
    Get segments on one of the axis (orientation) that can contain images based on black pixel distribution. 
    
    Arguemnts:
        img (np.array): gray scale image.
        black_min_thr (int): Lower threshold of the average pixel for text regions.
        black_max_thr (int): Upper threshold of the average pixel for text regions.
        window_size (int): Size of the rolling window to smooth concentrations of black text.
        
    Return:
        (list): List of candidate axis segments where images can be found.
    """
    if orientation=="vertical":
        y1 = 1 - img.mean(axis=1)
    elif orientation=="horizontal":
        y1 = 1 - img.mean(axis=0)
    else:
        raise ValueError(f"Orientation {orientation} must be one of: vertical, horizontal.")
    y2 = max_sliding_window(y1, window_size)
    segments = get_candidate_segments(y2)
    segments = filter_segments(segments, y2, black_min_thr, black_max_thr)
    return segments

def remove_images_from_full_pdf(pdf_path, x_min_thr, x_max_thr, x_window, 
                                y_min_thr, y_max_thr, y_window, lower_area_thr, plot=False):
    """
    Removes 
    
    Arguemnts:
        pdf_path (str): Path to the pdf document.
        plot (bool): Whether or not to plot the pages before and after images have been removed.
        x_min_thr (float): Lower threshold of the average pixel for text regions through horizontal axis.
        x_max_thr (float): Upper threshold of the average pixel for text regions through horizontal axis.
        x_window (int): Size of the rolling window to smooth concentrations of black text through horizontal axis.
        y_min_thr (int): Lower threshold of the average pixel for text regions through vertical axis.
        y_max_thr (int): Upper threshold of the average pixel for text regions through vertical axis.
        y_window (int): Size of the rolling window to smooth concentrations of black text through vertical axis.
        
    Return:
        (list): List of pdf pages (as arrays) with images removed.
    
    """
    # load
    pdf = convert_from_bytes(open('regular.pdf', 'rb').read())
    
    pages_wo_images = []
    
    for i, image in enumerate(pdf, 1):
        
        pix = np.array(image)

        # convert to gray
        gray = rgb2gray(pix)
        orig_img = deepcopy(gray)

        # remove non-black areas
        gray[gray > 1] = 1

        y_segments = get_segments(img=gray, orientation="vertical", 
                              black_thr=y_thr, window_size=y_window)

        coordinates = []
        for ymin, ymax in y_segments:
            xs_segments = get_segments(img=gray[ymin:ymax, :], orientation="horizontal", 
                                       black_thr=x_thr, window_size=x_window)
            coordinates.extend([((ymin, ymax),(xmin, xmax)) for xmin, xmax in xs_segments if (ymax-ymin)*(xmax-xmin)>lower_area_thr**2])
            
        if plot:
            
            f, axarr = plt.subplots(1, 2, figsize=(15,10))
            f.suptitle(f'page {i}', fontsize=16)
            axarr[0].title.set_text('Before')
            axarr[0].imshow(orig_img, cmap=plt.get_cmap('gray'))

            for (ymin,ymax),(xmin,xmax) in coordinates:
                orig_img[ymin:ymax,xmin:xmax] = 255
            
            axarr[1].title.set_text('After')
            axarr[1].imshow(orig_img, cmap=plt.get_cmap('gray'))
            plt.show()
        
        pages_wo_images.append(orig_img)
    
    return pages_wo_images
