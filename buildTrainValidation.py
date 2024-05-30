"""
File to build training and Validation datasets 
For Kanji Detection using deep learning
"""

def buildTrainValid(imageFolder,maskFolder,perc):
    """
        Receives a folder with images 
        And another of masks
        CAREFUL! name correspondences betwee
        mask and image files
    """
    for 