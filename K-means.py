#############################################################################
#Step 1. 隨機取K個點為群中心
#Step 2. 計算每個點到K個群中心距離，並取出最小的將其點歸入此群
#Step 3. 算出該群中所有點的新中心
#Step 4. 若群中所有點不再變動或達到最大迭代次數則為收斂，否則回到Step 2.狀態
#############################################################################
from sklearn import datasets, metrics
import random
import numpy
import copy
import time
import csv
import matplotlib.pyplot as plt

N = 0
K = 0
belongCluster = []
sum_of_square_error = []
outputData = []
sourceData = None
centerData = None

def main():
      global sourceData, centerData, K, N

      inputDataset()
      try:  
            for n_th in range(N):
                  for K in range(2,11,1):
                        initVariable()
                        start = time.time()

                        initCentroid()  #Step 1.
                        iter = 0
                        while True:
                              countChangingCluster = fitting() #Step 2.
                              isDone = updateCentroid() #Step 3.
                              if isDone or iter > 50 or countChangingCluster == 0:
                                    break
                              iter+=1

                        end = time.time()

                        outputData.append('K: '+str(K))
                        outputData.append('Cost '+str(iter)+ ' iterations.')
                        outputData.append('Time: '+str(end-start)+' s')
                        # 輪廓係數
                        # 表樣本與同類別距離相近，不同類別距離遠離的程度，範圍[-1,1]
                        outputData.append('Silhouette_score: '+str(metrics.silhouette_score(sourceData, belongCluster)))
                        outputData.append(copy.deepcopy(sum_of_square_error))
                        outputData.append('*--------------------------*')
            outputResult(outputData)
      except BaseException:
            print('Error')

def inputDataset():
      global sourceData, N

      dataset = input('Choose dataset, 1) Iris, 2) Abalone：')
      N = int(input('Determine running time：'))
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

def initVariable():
      belongCluster.clear()
      for i in range(sourceData.shape[0]):      # Initialize所有data歸屬群
            belongCluster.append(-1)
      sum_of_square_error.clear()

def initCentroid():
      global sourceData, centerData, K

      # Randomly select cluster center
      index = random.sample(range(sourceData.shape[0]), K)
      centerData = []
      for i in index:
            centerData.append(copy.deepcopy(list(sourceData[i])))

def fitting():
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

def updateCentroid():
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

      sum_of_square_error.append(sse)
      if len(sum_of_square_error) != 1:   # Sum_of_square_error 停止條件
            if sum_of_square_error[-1] == sum_of_square_error[-2]:
                  return True
      else:
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

def outputResult(outputData):
      with open('result_K-means.txt', 'w') as file:
            for data in outputData:
                  if isinstance(data, list):
                        for item in data:
                              file.write('%s\n' % item)
                  else:
                        file.write('%s\n' % data)
            
            file.close()

def drawPlot():
      plt.scatter(sourceData[:, 0], sourceData[:, 1], c=belongCluster)
      plt.show()

if __name__ == "__main__":
      main()