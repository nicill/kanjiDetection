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


#def pdfToPNG(pdfPATH,outPath = ""):
#  """
#    Receive a one page pdf file
#    Convert it to png format
#  """
#  caseCode = pdfPATH.split(os.sep)[-1][:-4]
#  filePath = outPath if outPath != "" else os.path.dirname(pdfPATH)
#  convert_from_path(pdfPATH, output_folder = filePath,fmt='png',output_file = caseCode)
#  os.rename(os.path.join(filePath,caseCode+"0001-1.png"),os.path.join(filePath,caseCode+".png"))

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

def sauvolaThreshold(im,window_size=25):
  """
  Function to apply Sauvola local threshold
  """
  th = threshold_sauvola(im, window_size=window_size)

  return (im > th).astype(int)*255

if __name__ == '__main__':
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
