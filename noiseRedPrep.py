import os
from pdf2image import convert_from_path
from skimage.filters import threshold_sauvola
from imageUtils import read_grayscale
import cv2
import sys, pymupdf
import fitz
import numpy as np

def deNoiseSubFolders(folder, sTh = 0.85, ws = 271):
    """
        Receive a folder with
        sakuma books one in each
        subfolder, denoise
        every subfolder in turn
    """
    for dirpath, dnames, fnames in os.walk(folder):
        for dirN in dnames:
            print("processing "+str(dirN))
            if "sakuma" in dirN:
                deNoiseFolder(os.path.join(folder, dirN), sTh , ws )

def deNoiseFolder(folder, sTh = 0.85, ws = 271):
    """
        Receive a folder corresponding
        to a scanned book
        Get all its image files
        Denoise them
    """
    for dirpath, dnames, fnames in os.walk(folder):
        for f in fnames:
            if "denoised" not in str(f):
                # read the image as a color image
                print("      ***** processing "+str(f))
                im = cv2.imread(os.path.join(folder,f))
                # Transform to grayscale
                img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                g2 = img_gray.copy()
                newName = f[:-4]+"denoised"+".png"

                ws = 271
                sauv = sauvolaThreshold(img_gray,hardness = sTh, window_size= ws)
                cv2.imwrite(os.path.join(folder,newName),sauv)

def deNoiseFolderADV(folder):
    """
        Receive a folder corresponding
        to a single sakuma data folder with all the pages from all the books 
        and denoise them if they have annotations
    """
    for dirpath, dnames, fnames in os.walk(folder):
        for f in fnames:
            if "KP" in str(f):
                # image name KPsakuma-0447_Page_03AN
                imName = f[2:f.find("AN")]+".png"
                newName = imName[:-4]+"denoised"+".png"
                # read the corresponding image as a grayscale image
                print("      ***** processing "+str(f))
                im = read_grayscale(os.path.join(folder,imName))

                noiseRed = reduceNoiseSakuma2(im)
                cv2.imwrite(os.path.join(folder,newName),noiseRed)


def pdfToPNG(pdfPATH,outPath = ""):
  """
    Receive a one page pdf file
    Convert it to png format
  """
  caseCode = pdfPATH.split(os.sep)[-1][:-4]
  #filePath = outPath if outPath != "" else os.path.dirname(pdfPATH)
  #convert_from_path(pdfPATH, output_folder = filePath,fmt='png',output_file = caseCode)
  #os.rename(os.path.join(filePath,caseCode+"0001-1.png"),os.path.join(filePath,caseCode+".png"))

  pdf_file = fitz.open(pdfPATH)
  num_of_pics = 0

  for page in pdf_file:
      images = page.get_images()
      if not len(images) == 0:
          for image in images:
                num_of_pics += 1
                xref = image[0]
                img = pdf_file.extract_image(xref)
                with open(pdfPATH[:-4]+".png".format(num_of_pics), "wb") as f:
                    f.write(img["image"])

  pdf_file.close()

def listOfPdfstoPNG(caseCodes):

    # put all of our image names in a list
    originalPDFPATHs = [caseCode+".pdf" for caseCode in caseCodes]
    origImagePATHs = [caseCode+".png" for caseCode in caseCodes]

    print(caseCodes)
    print(originalPDFPATHs)
    print(origImagePATHs)

    # Run the previous function with all the files in the list to convert all files to png
    list(map(pdfToPNG,originalPDFPATHs))

    return origImagePATHs

def globalBinarization(im,th=125):
  """
  Simple binarization function based on global threshold
  Using opencv
  """
  outTH , ret = cv2.threshold(im,th,255,cv2.THRESH_BINARY)
  return ret

def otsu(img):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

def sauvolaThreshold(im,hardness = 0.85, window_size=25):
  """
  Function to apply Sauvola local threshold
  """
  th = threshold_sauvola(im, window_size=window_size)
  return (im > hardness*th).astype(int)*255

#source https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
def cleanIsolatedPoints(img, it = 5):
    """
        Function to clean small isolated noise 
        regions with square kernel
    """
    kernel = np.ones((5, 5), np.uint8)
    first=cv2.erode(cv2.bitwise_not(img.astype('uint8') ),kernel,iterations = it)
    return cv2.bitwise_not(cv2.dilate(first,kernel,iterations = it))


def findLinesCanny(gray_image):

    v = np.median(gray_image)
    sigma=0.70
    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray_image, lower, upper)

def enhanceBlackCharacters(im,size = 3,it =1):
  """
    Function that receives a grayscale image 
    and erodes using a square kernel
    In Wasan documents, it enhances the black parts of the document
  """
  kernel = np.ones((size,size),np.uint8)
  return cv2.erode(im,kernel,iterations = it)   

def maskImage(im,mask):
    """
        receives an Image and a masks (black info on white bck)
        return the pixels in the image that are 
        white in the mask
    """
    aux = im.copy()
    aux[mask==255] = 255 
    return aux

def findBackgroundClusters(img):
    size1=5
    kernel=np.ones((size1,size1),np.uint8)
    return cv2.morphologyEx(255-img, cv2.MORPH_OPEN, kernel)

def eraseSmall(maskI, areaTH = 250, its = 5, ws = 501):
    """
    Receive a binary image
    erase regions that are small than the threshold
    """
    # copy not to destroy input
    mask = maskI.copy()
    mask = sauvolaThreshold(mask,hardness = 1, window_size = ws)

    mask = cleanIsolatedPoints(mask, its)
    numLabels, labelIm, stats, centroids = cv2.connectedComponentsWithStats(255-mask)

    for j in range(1,len(np.unique(labelIm))):
        if stats[j][4] < areaTH :
            mask[labelIm == j] = 255
    return mask

def detectLines(gray, kWidth = 1, ratioLines = 50):
    """
        Function to extract vertical and horizontal lines 
        using morphological operators
    """
    bw = cv2.adaptiveThreshold(255-gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # Create the images that will use to extract the horizontal and vertical lines
    processingV = np.copy(bw)
    processingH = np.copy(bw)
    
    # Specify sizes 
    rows, columns = processingV.shape
    vSize = rows // ratioLines
    hSize = columns // ratioLines
    
    # structure elements for extracting vertical and horizontal lines through morphology operations
    vStruct = cv2.getStructuringElement(cv2.MORPH_RECT, (kWidth, vSize))
    hStruct = cv2.getStructuringElement(cv2.MORPH_RECT, (hSize, kWidth))

    # Apply morphology operations
    #vertical lines
    processingV = cv2.erode(processingV, vStruct)
    processingV = cv2.dilate(processingV, vStruct)
    #horizontal lines
    processingH = cv2.erode(processingH, hStruct)
    processingH = cv2.dilate(processingH, hStruct)
        
    # Invert resutl images and gather results in a single image
    processing = 255 - processingV
    processing[processingH == 255] = 0

    # Step 1
    edges = cv2.adaptiveThreshold(processing, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    # Step 2
    kernel = np.ones((35, 35), np.uint8)
    edges = cv2.dilate(edges, kernel)

    # Show final result
    retVal = 255 - gray
    retVal[edges== 255] = 255
    return retVal

def reduceNoiseSakuma2(im, eroSize = 3, eroIts = 1, kW =1, rL = 50, areaTH = 250, smallRIts = 3, grayTH = 150):
    """
    Receives a grayscale image and reduces 
    noise according to the procedure described in
    https://www.mdpi.com/2076-3417/11/17/8050
    modified to include sauvola thresholding
    """
    # step 1, bring up the kanji with erosion 
    size = eroSize
    its = eroIts
    clearerKanji = enhanceBlackCharacters(im,size,its)
    #step 2, function to erase linear structures
    noLines = detectLines(clearerKanji, kWidth = kW, ratioLines = rL)
    # step 3, mask original image with the line removed image
    masked = maskImage(im,noLines)
    # step 4, bring up the kanji again with second erosion 
    clearerKanji2 = enhanceBlackCharacters(masked,size+2,its+2)
    # Step 5, clean up small noise regions, includes binarization
    smallRegionErased = eraseSmall(clearerKanji2, areaTH = areaTH, its = smallRIts)
    # step 6, mask again
    masked2 = maskImage(masked, smallRegionErased)
    # step 7 HARD global thresholding to erase gray
    final = masked2.copy()
    final[masked2<grayTH] = 0 
    final[masked2>=grayTH] = 255
    return final


if __name__ == '__main__':

    deNoiseFolderADV(sys.argv[1])

    sys.exit()
    ##OLD CODE
    deNoiseSubFolders(sys.argv[1])
    files = [x for x in sys.argv[1:]]
    #older CODE
    # now do local threshold processing for all images
    for imFile in files:
        # read the image as a color image
        print("processing "+str(imFile))
        im = cv2.imread(imFile)
        # Transform to grayscale
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g2 = img_gray.copy()
        newName = imFile[:-4]+"localThreshold"+".png"

        #ws = 101 if "Ex" in imFile else 31
        ws = 271
        sauv = sauvolaThreshold(img_gray,ws)
        cv2.imwrite(newName,sauv)


        # do also otsu ?
        #oats = otsu(g2)
        #cv2.imwrite(imFile[:-4]+"otsu"+".png",oats)

        # erase small regions
        areaTH = 300
        newName = imFile[:-4]+"NoSmall"+".png"
        noS = eraseSmall(sauv.astype("uint8"),areaTH = areaTH)
        cv2.imwrite(newName, noS)

        sys.exit()


        # erode
        sizeI = 3
        it = 5
        newName = imFile[:-4]+"IsolatedPoints"+".png"
        isolated = cleanIsolatedPoints(noS,np.ones((sizeI,sizeI),np.uint8), it = it)
        cv2.imwrite(newName, isolated)

        # erase small regions
        areaTH = 500
        newName = imFile[:-4]+"NoSmall2"+".png"
        noS = eraseSmall(isolated,areaTH = areaTH)
        cv2.imwrite(newName, noS)


        # enhance the remaining regions

        # mask out with the original grayscale image
