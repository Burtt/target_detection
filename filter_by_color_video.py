import numpy as np
import cv2

def checkIfContourMatches(cntr):
    #creates an approximation of the contours
    peri = cv2.arcLength(cntr, True)
    approx = cv2.approxPolyDP(cntr, 0.01 * peri, True)
    #checks how many points in each contour, four points => rectangle
    if len(approx) >= 4 and len(approx) <= 6:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)

        #compute the solidity of the original contours
        area = cv2.contourArea(cntr)
        hullArea = cv2.contourArea(cv2.convexHull(cntr))
        solidity = area / float(hullArea)

        #check to see if the width, height, and solidity and aspect ratio
        #fall within the appropriate bounds
        keepDims = w > 25 and h > 25
        keepSolidity = solidity > 0.9
        keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

        #check to see if all the tests were passed
        if keepDims and keepSolidity and keepAspectRatio:
            #draw an outline around the target and update the status
            cv2.drawContours(blurredMaskImage, [approx], -1, (255, 255, 0), 4)
            #return center of target to determine orientation of targets
            return ((x+x+w)/2,(y+y+h)/2)
        else:
            return False


cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    sourceImage = frame
    #sourceImage = cv2.imread("RBY_Targets.jpg")
    #convert pixel ratio to resize image
    #ratio = sourceImage.shape[0] / 300.0
    #orig = sourceImage.copy()
    #sourceImage = imutils.resize(sourceImage, height = 300)
    #edgesOnlyImage = cv2.Canny(sourceImage, 150,250)

    #show image with edges only to test if it is working
    #cv2.imshow("edgish", edgesOnlyImage)
    #cv2.waitKey(0)

    #create an array containing the three desired GBR color ranges: blue,yellow, red
    colorBoundaries =[
                    ([45, 0, 0], [145, 95, 55]),
                    ([25, 160, 180], [95, 230, 255]),
                    ([50, 50, 175], [125, 115, 250]),
                    ]
    #make empty list to put centers of targets in
    targets = []
    #loop through the ranges
    for (lowerBound, upperBound) in colorBoundaries:
        #openCV needs numpy arrays, cannot omit this
        lower = np.array(lowerBound, dtype = "uint8")
        upper = np.array(upperBound, dtype = "uint8")
        #turns pixels in range white, else black
        colorMask = cv2.inRange(sourceImage, lower, upper)

        #finds union of image and mask
        maskedSourceImage = cv2.bitwise_and(sourceImage, sourceImage, mask = colorMask)
        #blurs photo slightly to reduce false positives
        blurredMaskImage = cv2.bilateralFilter(maskedSourceImage, 11, 17, 17)
        edgyPhoto = cv2.Canny(blurredMaskImage, 100, 200)

        #now we will find the actual edges in the photo by looking for contours
        (_, contours, _) = cv2.findContours(edgyPhoto.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
        #contours = sorted(contours, key = cv2.contourArea, reverse = True)
        screenContour = None #initialize variable (center of target)
        
        for cntr in contours:
            screenContour = checkIfContourMatches(cntr)
            if(screenContour):
                break
        if not screenContour:
            print("target not found")
        else:
            print("target found")
            targets.append(screenContour)
            
        #show the filtered images
        cv2.imshow("images", np.hstack([sourceImage, blurredMaskImage]))
        #cv2.imshow("Aquired Target", maskedSourceImage)
    #determine orientation of targets if all are found
    if len(targets) == 3:
        print("all targets found")
        a, b, c = targets
        #formula derived from slopes between points to determine orientation
        val = (b[1] - a[1])*(b[0] - a[0]) - (c[1] - b[1])*(c[0] - b[0])
        if val > 0:
            print("clockwise")
        else:
            print("counterclockwise")
    else:
        print("some targets not found")
    if cv2.waitKey(25) and 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
