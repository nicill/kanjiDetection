import numpy as np
import cv2
import sys
from math import sqrt
import os
from pathlib import Path
import shutil

def gatherCharacters(source,dest):
    """
        Function to traverse all the "full"
        koutenshiki data base, traverse
        "characters" folders in the book
        subfolders and copy them while merging
        the same unicode character subfolders
    """
    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)
        if len(path) > 1 and path[-2] == "characters":
            shutil.copytree(root, os.path.join(dest,path[-1]),dirs_exist_ok=True)

def selectKanjiFromFile(inFile,inFolder,outFolder,etl = False):
    """
        First, read a file written in kanji and 
        store all characters in a dictionary
        if the kanji come from the etl database the names 
        of the folders are slightly different and need
        to be changed
    """
    charDict = {}
    countYes = 0
    countNo = 0
    with open(inFile) as f:
        while True:    
            c = f.read(1)
            if c != os.linesep and c!= " " and c and not c in charDict:
                code = "U+"+str(hex(ord(c)))[2:].upper()
                charDict[c] = code
                codeetl = "U+"+str(hex(ord(c)))[2:].upper() if not etl else "0x"+str(hex(ord(c)))[2:]
                dirName = os.path.join(inFolder,codeetl)
                if Path(dirName).is_dir():
                    countYes+=1
                    #print(str(dirName)+"EXISTS!!!!!!!!!!!!!!!!!")
                    print("copy to "+str(os.path.join(outFolder,code)))
                    shutil.copytree(dirName, os.path.join(outFolder,code), dirs_exist_ok=True)  # Fine
                else:
                    countNo+=1
                    print(str(dirName)+" does not exist "+str(c))
            if not c:
                break
    print(countYes)
    print(countNo)
    return charDict




if __name__ == '__main__':
    print(selectKanjiFromFile(sys.argv[1],sys.argv[2],sys.argv[3],etl = True))