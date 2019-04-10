from sklearn import datasets
import random
import numpy
import time
import csv

import matplotlib.pyplot as plt

K = 3
belongCluster = []
sourceData = None
centerData = None
sum_of_square_error = 0

def main():
      global sourceData, centerData, K

      dataset = input('Choose dataset, 1) Iris, 2) Abalone：')
      K = int(input('Define number of K：'))
      if dataset == '1':
            iris = datasets.load_iris()
            sourceData = iris.data[:, :4]
      elif dataset == '2':  
            with open('abalone.csv', 'r') as f:
                  reader = csv.reader(f)
                  sourceData = list(reader)
            for i in range(len(sourceData)):
                  sourceData[i] = [str(ord(x)) if x.isalpha() else x for x in sourceData[i]]
                  sourceData[i] = [float(x) if True else x for x in sourceData[i]]
      try:  
            start = time.time()

            sourceData = numpy.asarray(sourceData)
            centerData = initCluster()  #shape[0]表資料數，shape[0]表資料維度
            iter = 0
            while True:
                  Clustering()
                  isDone = updateCenter()
                  print(sum_of_square_error)
                  if isDone and iter > 20:
                        break
                  iter+=1

            end = time.time()
            print('Spent time:', end-start, 's')

            drawPlot()
      except BaseException:
            print('Error')

def initCluster():
      global sourceData, centerData, K

      for i in range(sourceData.shape[0]):
            belongCluster.append(-1)

      index = random.sample(range(sourceData.shape[0]), K)
      centerData = []
      for i in index:
            centerData.append(list(sourceData[i]))
      return centerData

def Clustering():
      global sourceData, centerData, K

      for index, data in enumerate(sourceData):
            minDistance = -1
            for k in range(K):
                  dis = calDistance(data, centerData[k], sourceData.shape[1])
                  if minDistance == -1 or dis < minDistance:
                        minDistance = dis
                        belongCluster[index] = k

def updateCenter():
      global sourceData, centerData, sum_of_square_error, K

      sse = 0
      for k in range(K):
            tempSum = [0]*sourceData.shape[1]
            index = [i for i,x in enumerate(belongCluster) if x == k]
            for dim in range(sourceData.shape[1]):
                  for i in index:
                        tempSum[dim] += sourceData[i][dim]
                        sse += numpy.square(sourceData[i][dim]-centerData[k][dim])
                  centerData[k][dim] = tempSum[dim] / len(index)
      if sse == sum_of_square_error:
            return True
      else:
            sum_of_square_error = sse
            return False

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