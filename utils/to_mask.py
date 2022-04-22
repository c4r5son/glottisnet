import os
import glob
import argparse
import json
import numpy as np
import cv2 as cv

def load_json(f): 
    '''Loads a JSON file to read. Takes the file to open as an argument.
        @params:
        f   - Required : JSON file to open
    '''
    with open(f, 'r') as fp:
        return json.load(fp)

def parse_args() -> argparse.Namespace:

    """Parse arguments."""
    parser = argparse.ArgumentParser(description="bdd100k to a black and white bitmask")
    parser.add_argument(
        "-i",
        "--input",
        help=(
            "path to a label JSON file"
        ),
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save the bitmasks",
    )
    parser.add_argument(
        "--image-directory",
        help="path to an image directory for proper sizing (jpgs)",
    )
    return parser.parse_args()

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if iteration>0:
        #clear previous line
        print ("\033[A                             \033[A")
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def listToString(s) -> str: 
    '''Converts a passed list to a string with '/' in between strings'''
    # initialize an empty string
    str1 = "" 
    # traverse in the string  
    for ele in s: 
        str1 += ele  
        str1 += "/"
    
    # return string  
    return str1

def main():
    '''The main function.'''
    args = parse_args()

    #set the file path to the user defined value
    file_path = args.input
    print("Loading annotations...")
    jsonFile = load_json(args.input)

    if args.image_directory != None:
        os.chdir(args.image_directory)
        images = glob.glob("*.jpg")
        img = cv.imread(images[0], cv.IMREAD_UNCHANGED)
        shape = [img.shape[0],img.shape[1],3]
    else:
        shape = [434,503,3]

    #run over every frame in the jsonFile
    counter = 0
    print("Converting JSON to bitmasks...")
    print("Converting JSON to bitmasks...")
    for frame in jsonFile["frames"]:

        #create a black image
        img = np.zeros(shape, np.uint8)

        #check to see if there is a labeled polygon in the frame
        for label in frame["labels"]:
            if "poly2d" in label:
                vertices = np.asarray(label["poly2d"][0]["vertices"],dtype=np.int32)
                vertices = vertices.reshape((-1,1,2))

                #draw the polygon
                cv.fillPoly(img,pts= [vertices],color= (255,255,255))

        #save the image
        splitFilePath = file_path.rsplit('/')
        splitName = frame["name"].rsplit('/')
        saveFilePath = listToString(splitFilePath[:len(splitFilePath)-1])+"masks/"

        if args.output != None:
            saveFilePath = args.output + '/'
        else:
            #check to make sure the masks directory exists
            try:
                os.mkdir(saveFilePath)
            except FileExistsError:
                pass

        #save the image
        saveFilePath = saveFilePath+listToString(splitName[-1:])[:-1]
        cv.imwrite(saveFilePath,img)
        
        #update counter for progress bar
        counter += 1
        printProgressBar(counter,len(jsonFile["frames"]))

    #clear last line and print done
    print ("\033[A                             \033[A")
    print("Done!")
        

    

if __name__ == "__main__":
    main()