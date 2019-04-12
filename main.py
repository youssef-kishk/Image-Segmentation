from scipy.spatial import distance_matrix
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import scipy.io
from PIL import Image
import math
from skimage import  color
from skimage.future import graph
from random import randint

def readImagesPath(path):
    dataName = path
    dataDir = "" + dataName
    trainData = []
    for ImageName in os.listdir(dataDir):
        ImagePath = os.path.join(dataDir, ImageName)
        trainData += [ImagePath]
    # return all images paths
    return trainData
#########################################################################################################################################################
def imgRGBread(images):
    rgbImages = []  # 3D images
    vectorizedImages = []
    for i in range(0, len(images)):
        image = mpimg.imread(images[i])
        rgbImages.append(image)
        # Blur image to reduce the edge content and makes the transition form one color to the other very smooth.
        # Check video:https://www.youtube.com/watch?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&v=sARklx6sgDk
        image = cv2.GaussianBlur(rgbImages[i], (7, 7), 0)

        # convert image to (M*N)*3 Matrix
        vectorizedImages.append(image.reshape(-1, 3))
    return rgbImages, vectorizedImages
#########################################################################################################################################################
def imgRGBreadOneImage(imagePath):
    rgbImages = []  # 3D images
    vectorizedImage = []
    image = mpimg.imread(imagePath)
    rgbImages.append(image)
    # Blur image to reduce the edge content and makes the transition form one color to the other very smooth.
    # Check video:https://www.youtube.com/watch?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&v=sARklx6sgDk
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # convert image to (M*N)*3 Matrix
    vectorizedImage.append(image.reshape(-1, 3))
    return rgbImages, vectorizedImage
#########################################################################################################################################################
def kmeans(dataSet, k):
    # I/p one of the images of vectorized Images list
    numOfPoints = len(dataSet)
    # k random initial points
    randomIndeces = np.random.choice(numOfPoints, k, replace=False)
    centers = []
    for i in range(0, len(randomIndeces)):
        centers.append(dataSet[randomIndeces[i]])
    centersOld = [0] * k
    clusterAssignment = [0] * len(dataSet)
    start = 0
    while (1):
        flag = 0
        if start != 0:
            for i in range(0, k):
                if centersOld[i] != centers[i]:
                    centersOld[i] = centers[i]
                else:
                    flag += 1
        start = 1
        if flag == k:
            return (centers, clusterAssignment)
            # distance between points and centers matrix
        distMatrix = distance_matrix(dataSet, centers, p=2)

        for i in range(0, numOfPoints):
            # closest center
            d = distMatrix[i]
            closestCenter = (np.where(d == np.min(d)))[0][0]
            # associate point to closest center
            clusterAssignment[i] = closestCenter

        # new centers
        for i in range(0, k):
            sumX = 0
            sumY = 0
            sumZ = 0
            count = 0
            for j in range(0, numOfPoints):
                if (clusterAssignment[j] == i):
                    sumX += (dataSet[j])[0]
                    sumY += (dataSet[j])[1]
                    sumZ += (dataSet[j])[2]
                    count += 1
            centers[i] = (sumX / count, sumY / count, sumZ / count)
    return (centers, clusterAssignment)
#########################################################################################################################################################
def __extractGrondTruthMatrix(mat):
    _groundTruthLabelVectorList = []
    _groundTruthMatrixes = []
    for _groundTruthMatrix in mat["groundTruth"][0]:
        _groundTruthMatrix = _groundTruthMatrix[0][0][0]
        _groundTruthMatrixes.append(_groundTruthMatrix)
        tempList = []
        for row in _groundTruthMatrix:
            tempList.extend(row.tolist())
        _groundTruthLabelVectorList.append(tempList)

    return _groundTruthMatrixes,_groundTruthLabelVectorList
#########################################################################################################################################################
def __getGroundTruthLabels(groundTruthMatrix,image):
    _labelsDict = {}
    i = -1
    for row in groundTruthMatrix:
        i += 1
        j = -1
        for key in row:
            j += 1
            if key not in _labelsDict:
                ima = image[i][j]
                _labelsDict.update({key:[ima[2],ima[1],ima[0]]})
    return _labelsDict
#########################################################################################################################################################
def getGroundTruthLabelsAndGenerateImage(matPath,imagePath):

    image = cv2.imread(imagePath)
    mat = scipy.io.loadmat(matPath)
    _groundTruthMatrixes,_groundTruthLabelVectorList = __extractGrondTruthMatrix(mat)
    for z in range(len(_groundTruthMatrixes)):
        groundTruthMatrix = _groundTruthMatrixes[z]
        labelsDict = __getGroundTruthLabels(groundTruthMatrix,image)
        rowsNumber  = len(groundTruthMatrix)
        colsNumber = len(groundTruthMatrix[0])
        rgbArray = np.zeros((rowsNumber, colsNumber, 3), 'uint8')
        for i in range(rowsNumber):
            for j in range(colsNumber):
                rgbArray[i][j] = labelsDict[groundTruthMatrix[i][j]]
        img = Image.fromarray(rgbArray)
        img.save('groundTruth#'+str(z)+'.jpg')

    return _groundTruthLabelVectorList
#########################################################################################################################################################
def purityOfEachClass(labels, groundTruth2, k=3, sorted=True):
    groundTruthLabesNumber = 0
    for i in range(len(groundTruth2)):
        if groundTruthLabesNumber < groundTruth2[i]:
            groundTruthLabesNumber = groundTruth2[i]
    groundTruthLabesNumber += 1
    dataInClusterindexies = []
    for i in range(k):
        dataInClusterindexies.append([])
    for i in range(len(groundTruth2)):
        dataInClusterindexies[labels[i]].append(i)
    listNij = []
    for i in range(k):
        list = [0] * (groundTruthLabesNumber)
        listNij.append(list)

    for i in range(k):
        for j in range(len(dataInClusterindexies[i])):
            listNij[i][groundTruth2[dataInClusterindexies[i][j]]] += 1
    finalListNij = []
    for i in range(k):
        list = [0] * (groundTruthLabesNumber)
        finalListNij.append(list)

    for i in range(k):
        for j in range(groundTruthLabesNumber):
            finalListNij[i][j] = (listNij[i][j], j + 1)

    groundtruthList = [0] * (groundTruthLabesNumber)

    for j in range(groundTruthLabesNumber):
        sum = 0
        for i in range(k):
            sum += finalListNij[i][j][0]
        groundtruthList[j] = sum

    if sorted == True:
        for i in range(k):
            finalListNij[i].sort(reverse=True)

    return finalListNij, groundtruthList, groundTruthLabesNumber
#########################################################################################################################################################
def calculatePurity(labels, groundTruth, k=3):
    listNij, groundtruthList, groundTruthLabesNumber = purityOfEachClass(labels, groundTruth, k)
    sum = 0
    for i in range(k):
        sum += listNij[i][0]
    purity = sum / len(labels)
    return purity
#########################################################################################################################################################
def calculateF_Measure(labels, groundTruth, k=3):
    listNij, groundtruthList, groundTruthLabesNumber = purityOfEachClass(labels, groundTruth, k)
    NumberOfElementsInEachCluster = [0] * k
    for i in range(k):
        for j in range(len(listNij[i])):
            NumberOfElementsInEachCluster[i] += listNij[i][j][0]
    listF_measure = [0]*k
    for i in range(k):
        if NumberOfElementsInEachCluster[i] == 0:
            listF_measure[i] = 0
        else:
            if i > (len(groundtruthList) - 1):
                listF_measure[i] = ((2 * listNij[i][0][0]) / (NumberOfElementsInEachCluster[i]))
            else:
                listF_measure[i] = ((2 * listNij[i][0][0]) / (NumberOfElementsInEachCluster[i] + groundtruthList[listNij[i][0][1]-1]))
    sum = 0
    for i in range(k):
        sum += listF_measure[i]
    f_Measure = sum / k
    return f_Measure
#########################################################################################################################################################
def calculateConditionalEntropy(labels, groundTruth, k=3):
    listNij, groundtruthList, groundTruthLabesNumber = purityOfEachClass(labels, groundTruth, k, sorted=False)
    sizeOfData = len(groundTruth)
    numberOfElementsInEachCluster = [0] * k
    entropyOfEachCluster = [0] * k
    for i in range(k):
        for j in range(groundTruthLabesNumber):
            numberOfElementsInEachCluster[i] += listNij[i][j][0]
    for i in range(k):
        for j in range(groundTruthLabesNumber):
            if numberOfElementsInEachCluster[i] != 0:
                tempValue = listNij[i][j][0] / numberOfElementsInEachCluster[i]
            if tempValue != 0:
                entropyOfEachCluster[i] += (-tempValue) * math.log2((tempValue))

    entropy = 0
    for i in range(k):
        entropy += (numberOfElementsInEachCluster[i] / sizeOfData) * entropyOfEachCluster[i]
    return entropy
#########################################################################################################################################################
def calculateEntropyAndF_measure(clustersLabels,groundTruthLabelsVectorList , k=3):
    entropySum = 0
    f_measureSum = 0
    i = -1
    maxEntropy = 99999
    maxF_measure = 0
    bestGroundTruthLabels = 0
    for groundTruthLabelVector in groundTruthLabelsVectorList:
        i += 1
        condEntropy = calculateConditionalEntropy(clustersLabels, groundTruthLabelVector, k=k)
        fMeasure = calculateF_Measure(clustersLabels, groundTruthLabelVector, k=k)
        entropySum += condEntropy
        f_measureSum += fMeasure
        if condEntropy < maxEntropy and fMeasure > maxF_measure:
            bestGroundTruthLabels = groundTruthLabelVector
    avgEntropy = entropySum/i
    avgf_measure = f_measureSum/i
    return avgEntropy,(1-avgf_measure),bestGroundTruthLabels
#########################################################################################################################################################
def normalizedCut(testRGBImage,imagePath,clustersLabels,groundTruthLabelVector,k):
    image = mpimg.imread(imagePath)   
    
    #Compute the Region Adjacency Graph    
    g = graph.rag_mean_color(testRGBImage, np.reshape(clustersLabels,(nrows,ncols)), mode='similarity')
    #Perform Normalized Graph cut on the Region Adjacency Graph.
    labels = graph.cut_normalized(np.reshape(clustersLabels,(nrows,ncols)), g)
    #return labels
    return labels
#########################################################################################################################################################
def resShape_2D_ListTo_1D(inputlist):
    listToReturn = []
    for row in inputlist:
        listToReturn.extend(row)
    return listToReturn
#########################################################################################################################################################
#Bounus
def eklidianDistance(x,y,xCor,yCor):
    result = 0
    for i in range(len(x)):
        result += (x[i]-y[i])*(x[i]-y[i])
    for i in range(len(xCor)):
        result += (xCor[i]-yCor[i])*(xCor[i]-yCor[i])
    return math.sqrt(result)
#########################################################################################################################################################
def k_meanAlgUsingRBGAndPixelPosition(dataMatrix,imageWidth,k=1,prevCenters=[]):
    if len(dataMatrix) < k:
        return "error k must be equal number of clusters"
    thereIsChange = True
    centers = []
    centersCoordinate = []
    for i in range(k):
        index = randint(0, (len(dataMatrix)-1))
        newcenter = dataMatrix[index].tolist()
        centerCoordinate = [(index//imageWidth),(index%imageWidth)]
        while newcenter in centers or centerCoordinate in centersCoordinate :
            index = randint(0, (len(dataMatrix) - 1))
            newcenter = dataMatrix[index].tolist()
            centerCoordinate = [(index // imageWidth), (index % imageWidth)]
        centersCoordinate.append(centerCoordinate)
        centers.append(newcenter)
    loops = 0
    while thereIsChange and loops < 20:
        # print(loops)
        loops +=1
        labels = []
        distances = []
        prevCentersInLoop = []
        prevCentersCoordiantesInLoop = []
        for i in range(k):
            list = []
            for j in range(len(dataMatrix)):
                dataCoordinate = [(j // imageWidth), (j % imageWidth)]
                list.append((eklidianDistance(centers[i],dataMatrix[j],centersCoordinate[i],dataCoordinate),j))
            distances.append(list)

        #calculate new centers
        for i in range(len(centers)):
            prevCentersInLoop.append(centers[i])
            prevCentersCoordiantesInLoop.append(centersCoordinate[i])
        for i in range(len(distances[0])):
            min = 99999999
            minLabel = 1
            for j in range(len(centers)):
                if min > distances[j][i][0]:
                    min = distances[j][i][0]
                    minLabel =  j
            labels.append(minLabel)

        for i in range(0,k):
            counter =0
            calculateNewCenters = []
            calculateNewCentersCoordinate = []
            for j in range(len(centers[0])):
                calculateNewCenters.append(0)
            for j in range(len(centersCoordinate[0])):
                calculateNewCentersCoordinate.append(0)
            for j in range(len(labels)):
                if labels[j] == i :
                    counter +=1
                    for z in range(len(centers[0])):
                        calculateNewCenters[z] += dataMatrix[j][z]

                    calculateNewCentersCoordinate[0] = (j // imageWidth)
                    calculateNewCentersCoordinate[1] = (j % imageWidth)
            if counter > 0:
                for z in range(len(centers[0])):
                    calculateNewCenters[z] = calculateNewCenters[z]/counter
                for z in range(len(centersCoordinate[0])):
                    calculateNewCentersCoordinate[z] = calculateNewCentersCoordinate[z]/counter
                del centers[i]
                centers.insert(i,calculateNewCenters)
                del centersCoordinate[i]
                centersCoordinate.insert(i, calculateNewCentersCoordinate)

        thereIsChange = False
        for i in range(len(centers)):
            for j in range(len(centers[0])):
                if centers[j] != prevCentersInLoop[j]:
                    thereIsChange = True
    return centers,labels
#########################################################################################################################################################

if __name__ == '__main__':
    trainImages = readImagesPath("data/images/train")
    graundTruthImages = readImagesPath("data/groundTruth/train") 

    ######################################    
    #The only values to change in the code
    kValues = [3,5,7,9]
    imageIndex=79
    ######################################   
      
    matPath = graundTruthImages[imageIndex]
    imagePath=trainImages[imageIndex]
    
    groundTruthLabelsVectorList = getGroundTruthLabelsAndGenerateImage (matPath,imagePath)
    rgbImages, vectorizedImages = imgRGBread(trainImages)
    nrows = len(rgbImages[imageIndex])
    ncols = len(rgbImages[imageIndex][0])
    testRGBImage = rgbImages[imageIndex]
    testImage = vectorizedImages[imageIndex]
    i =-1
    outList = []
    normalizedOutList = []
    specitalRGBOutList = []
    for k in kValues:
        bestTruthLabel=0
        i += 1

        print('K value = ',k)
        finalCenters, clustersLabels = kmeans(testImage, k)
        out = color.label2rgb(np.reshape(clustersLabels,(nrows,ncols)), testRGBImage, kind='avg')
        outList.append(out)
        print("Manually Implemented Kmeans")
        entropy, f_measure,bestTruthLabel = calculateEntropyAndF_measure(clustersLabels, groundTruthLabelsVectorList, k=k)
        print("ConditionalEntropy of k = ", k, " = ", entropy)
        print("F_Measure of k =", k, " = ", f_measure, "\n")

        print('*****************************************************************')
        print('Sickit learn KMeans:')
        #usage for sickit learn Kmeans
        testKMeansSickitLearn = KMeans(n_clusters=k).fit(testImage)
        entropy, f_measure, neglect = calculateEntropyAndF_measure(testKMeansSickitLearn.labels_, groundTruthLabelsVectorList, k=k)
        print("ConditionalEntropy of k = ", k, " = ", entropy)
        print("F_Measure of k =", k, " = ", f_measure, "\n")

        print('*****************************************************************')
        print("Normalized Cut:")
        labels = normalizedCut(testRGBImage,imagePath,clustersLabels,bestTruthLabel,k)
        clustersLabels = resShape_2D_ListTo_1D(labels)
        entropy, f_measure, neglect = calculateEntropyAndF_measure(clustersLabels,groundTruthLabelsVectorList, k=k)
        print("ConditionalEntropy of k = ", k, " = ", entropy)
        print("F_Measure of k =", k, " = ", f_measure, "\n")

        out = color.label2rgb(labels, testRGBImage, kind='avg')
        normalizedOutList.append(out)
        print('*****************************************************************')

        print("specital RGB Kmean Cut:")
        finalCenters, clustersLabels = k_meanAlgUsingRBGAndPixelPosition(testImage, ncols, k)
        entropy, f_measure,bestTruthLabel = calculateEntropyAndF_measure(clustersLabels, groundTruthLabelsVectorList, k=k)
        print("ConditionalEntropy of k = ", k, " = ", entropy)
        print("F_Measure of k =", k, " = ", f_measure, "\n")
        out = color.label2rgb(np.reshape(clustersLabels, (nrows, ncols)), testRGBImage, kind='avg')
        specitalRGBOutList.append(out)
        print('*****************************************************************')

    nrows = 3
    ncols = math.ceil((len(specitalRGBOutList)+1)/2)
    fig, ax = plt.subplots(nrows= nrows,ncols= ncols, figsize=(6, 8))
    ax[0][0].imshow(mpimg.imread(imagePath))

    for i in range(nrows):
        for j in range(ncols):
            if i == 0:
                j = 1
            ax[i][j].imshow(outList[i])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows= nrows,ncols= ncols , figsize=(6, 8))
    ax[0][0].imshow(mpimg.imread(imagePath))
    for i in range(ncols):
        for j in range(ncols):
            if i == 0:
                j = 1
            ax[i][j].imshow(normalizedOutList[i])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))
    ax[0][0].imshow(mpimg.imread(imagePath))
    for i in range(nrows):
        for j in range(ncols):
            if i == 0:
                j = 1
            ax[i][j].imshow(specitalRGBOutList[i])
    plt.tight_layout()
    plt.show()
    #..........................................................................................