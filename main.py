#############################################################################
#Step 1. 隨機取K個點為群中心
#Step 2. 計算每個點到K個群中心距離，並取出最小的將其點歸入此群
#Step 3. 算出該群中所有點的新中心
#Step 4. 若群中所有點不再變動或達到最大迭代次數則為收斂，否則回到Step 2.狀態
#############################################################################
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

      inputDataset()
      try:  
            start = time.time()

            centerData = initCluster()  #Step 1.
            iter = 0
            while True:
                  countChangingCluster = Clustering() #Step 2.
                  isDone = updateCenter() #Step 3.
                  print(sum_of_square_error)
                  if isDone or iter > 50 or countChangingCluster == 0:
                        break
                  iter+=1

            end = time.time()
            print('Spent time:', end-start, 's')

            drawPlot()
      except BaseException:
            print('Error')

def inputDataset():
      global sourceData

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
                  sourceData[i] = [str(ord(x)) if x.isalpha() else x for x in sourceData[i]]    #shape[0]:資料數，shape[0]:資料維度
                  sourceData[i] = [float(x) if True else x for x in sourceData[i]]
            sourceData = numpy.asarray(sourceData)

def initCluster():
      global sourceData, centerData, K

      for i in range(sourceData.shape[0]):      # Initialize所有data歸屬群
            belongCluster.append(-1)

      # Randomly select cluster center
      index = random.sample(range(sourceData.shape[0]), K)
      centerData = []
      for i in index:
            centerData.append(list(sourceData[i]))

      return centerData

def Clustering():
      global sourceData, centerData, K
      
      latestBelongCluster = belongCluster.copy()
      # 判斷點歸屬cluster, by distance
      for index, data in enumerate(sourceData):
            minDistance = -1
            for k in range(K):
                  dis = calDistance(data, centerData[k], sourceData.shape[1])
                  if minDistance == -1 or dis < minDistance:
                        minDistance = dis
                        belongCluster[index] = k

      countChangingCluster = countListDiff(latestBelongCluster,belongCluster) #計算ata的變動情況
      return countChangingCluster

def updateCenter():
      global sourceData, centerData, sum_of_square_error, K

      sse = 0
      for k in range(K):
            tempSum = [0]*sourceData.shape[1]
            index = [i for i,x in enumerate(belongCluster) if x == k]   # 找到在belongCluster中所有值為k的元素index
            for dim in range(sourceData.shape[1]):    # 計算所有屬於cluster k的總和，藉此取平均找出新cluster center
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
      retSum = 0
      for i in range(dim):
           retSum += numpy.square(data[i]-center[i])
      return retSum

def countListDiff(list1, list2):
      retSum = 0
      for i in range(len(list1)):
            if list1[i] != list2[i]:
                  retSum += 1
      return retSum

def drawPlot():
      plt.scatter(sourceData[:, 0], sourceData[:, 1], c=belongCluster)
      plt.show()

if __name__ == "__main__":
      main()