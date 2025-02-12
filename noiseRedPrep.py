import os
from pdf2image import convert_from_path
from skimage.filters import threshold_sauvola
import cv2
import sys, pymupdf
import fitz

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

def sauvolaThreshold(im,window_size=25):
  """
  Function to apply Sauvola local threshold
  """
  th = threshold_sauvola(im, window_size=window_size)

  return (im > th).astype(int)*255

#source https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
def cleanIsolatedPoints(img,kernel):
    """
        to clean small isolated noise regions
        possible kernel
        np.ones((size1,size1),np.uint8)
    """
    it=3
    first=cv2.erode(cv2.bitwise_not(img),kernel,iterations = it)
    #return 250-first
    return cv2.bitwise_not(cv2.dilate(first,kernel,iterations = it))

def findLinesCanny(gray_image):

    v = np.median(gray_image)
    sigma=0.70
    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray_image, lower, upper)

def findBackgroundClusters(img):
    size1=5
    kernel=np.ones((size1,size1),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def detectVerticalLines(gray):
    """
        possible kernel
        cv2.getStructuringElement(cv2.MORPH_RECT,(8,30))
        does this work????
    """
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # Create the images that will use to extract the horizontal and vertical lines
    vertical = np.copy(bw)
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 50

    print(verticalsize)
    # CreImageProcessingUtils.ate structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical)
    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''
    # Step 1
    edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    # Step 2
    kernel = np.ones((25, 25), np.uint8)
    edges = cv2.dilate(edges, kernel)
    # Step 3
    #smooth = np.copy(vertical)
    # Step 4
    #smooth = cv2.blur(smooth, (2, 2))
    # Step 5
    #(rows, cols) = np.where(edges != 0)
    #vertical[rows, cols] = smooth[rows, cols]
    # Show final result
    return cv2.bitwise_not(edges)



if __name__ == '__main__':
    """
    files = [os.path.join("noiseRedData","2024-05-31-ExcellentA"),
    os.path.join("noiseRedData","2024-05-31-FineA"),
    os.path.join("noiseRedData","2024-05-31-SuperFineA"),
    os.path.join("noiseRedData","2024-05-31-ExcellentB"),
    os.path.join("noiseRedData","2024-05-31-FineB"),
    os.path.join("noiseRedData","2024-05-31-SuperFineB")]
    pngIMpaths = listOfPdfstoPNG(files)
    # now do local threshold processing for all images
    for imFile in pngIMpaths:
        # read the image as a binary image
        im = cv2.imread(imFile,0)
        newName = imFile[:-4]+"localThreshold"+".png"
        ws = 101 if "Ex" in imFile else 31
        cv2.imwrite(newName,sauvolaThreshold(im,ws))

    """
    files = [x for x in sys.argv[1:]]
    # now do local threshold processing for all images
    for imFile in files:
        # read the image as a binary image
        print("processing "+str(imFile))
        im = cv2.imread(imFile)
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g2 = img_gray.copy()
        newName = imFile[:-4]+"localThreshold"+".png"
        #ws = 101 if "Ex" in imFile else 31
        ws = 271
        cv2.imwrite(newName,sauvolaThreshold(img_gray,ws))

        # do also otsu
        cv2.imwrite(imFile[:-4]+"otsu"+".png",otsu(g2))
