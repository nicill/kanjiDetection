import numpy as np
import cv2
import sys
from math import sqrt
import os
from pathlib import Path
import shutil
from collections import defaultdict

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


def makeKoutenshiki(source,dest,th = 9):
    """
        Copy folders not containing hiragana and 
        with at least th files 
    """
    codeDict = defaultdict(lambda:0)
    # create dict with codes
    avoid_list = [
                    'U+{:04X}'.format(i)
                    for i in range(0x3040, 0x30A0)
                ]

    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)

        for f in files:
            if len(path) > 1 and path[-2] == "characters":
                code = path[-1]
                if code not in avoid_list:
                    codeDict[code]+=1
                #else:
                #    print("avoiding "+code)
            #if int(3040   int(code) != "30" and more files than th                
            #    shutil.copytree(root, os.path.join(dest,path[-1]),dirs_exist_ok=True)
    #print(codeDict)
    # now only copy those with at least th repetitions as shown in codeDict
    for k,v in codeDict.items():
        if v > th:
            print("copy "+k+" to "+str(os.path.join(dest,k))+ "because of having "+str(v))    
            shutil.copytree(os.path.join(source,k), os.path.join(dest,k),dirs_exist_ok=True)



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
                    #print("copy to "+str(os.path.join(outFolder,code)))
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
    #print(selectKanjiFromFile(sys.argv[1],sys.argv[2],sys.argv[3],etl = True))
    #gatherCharacters(sys.argv[1],sys.argv[2])
    makeKoutenshiki(sys.argv[1],sys.argv[2])