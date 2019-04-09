from sklearn import datasets
import random
import numpy
import matplotlib.pyplot as plt

K = 3
belongCluster = []
sourceData = None
centerData = None

def main():
      global sourceData, centerData

      
      iris = datasets.load_iris()
      sourceData = iris.data[:, :4]  
      centerData = initCluster()  #shape[0]表資料數，shape[0]表資料維度
      for i in range(30):
            Clustering()
            updateCenter()
      drawPlot()

def initCluster():
      global sourceData, centerData

      for i in range(sourceData.shape[0]):
            belongCluster.append(-1)

      index = random.sample(range(sourceData.shape[0]), K)
      centerData = []
      for i in index:
            centerData.append(list(sourceData[i]))
      return centerData

def Clustering():
      global sourceData, centerData

      for index, data in enumerate(sourceData):
            minDistance = -1
            for k in range(K):
                  dis = calDistance(data, centerData[k], sourceData.shape[1])
                  if minDistance == -1 or dis < minDistance:
                        minDistance = dis
                        belongCluster[index] = k

def updateCenter():
      global sourceData, centerData

      for k in range(K):
            temp = [0,0,0,0]
            index = [i for i,x in enumerate(belongCluster) if x == k]
            for dim in range(sourceData.shape[1]):
                  for i in index:
                        temp[dim] += sourceData[i][dim]
                  centerData[k][dim] = temp[dim] / len(index)

def calDistance(data, center, dim):
      sum = 0
      for i in range(dim):
           sum += numpy.square(data[i]-center[i])
      return sum

def drawPlot():
      plt.scatter(sourceData[:, 0], sourceData[:, 1], c=belongCluster)
      plt.show()

if __name__ == "__main__":
      main()