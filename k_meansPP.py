#############################################################################
# Step 1. 隨機取K個點為群中心
# Step 2. 計算每個點到K個群中心距離，並取出最小的將其點歸入此群
# Step 3. 算出該群中所有點的新中心
# Step 4. 若群中所有點不再變動或達到最大迭代次數則為收斂，否則回到Step 2.狀態
#############################################################################
import random
import numpy
import copy

class K_meansPP():
      def __init__(self, sourceData, K):
            self.sourceData = sourceData
            self.K = K
            self.dim = sourceData.shape[1]
            self.centerData = []
            self.belongCluster = [-1]*sourceData.shape[0]
            self.sum_of_square_error = []

      def initCentroid(self):
            firstMean = random.randint(0,self.sourceData.shape[0])
            self.centerData.append(self.sourceData[firstMean])
            for k in range(self.K-1):
                  dataDistance = []
                  for center in self.centerData:     # 找出資料點與群中心之距離
                        dataDistance.append([self.calDistance(data, center, self.dim) for data in self.sourceData])
                  dataDistance = numpy.array(dataDistance).min(0) # 找出所有點對其所屬群中心點的最近距離,min(0)為所有列中最小值

                  # 根據距離給選中機率
                  probability = numpy.array([item/sum(dataDistance) for item in dataDistance])
                  index = numpy.random.choice(list(range(self.sourceData.shape[0])), p = probability.ravel())
                  self.centerData.append(copy.deepcopy(self.sourceData[index])) 

      def fitting(self):
            latestBelongCluster = self.belongCluster.copy()
            # 判斷點歸屬cluster, by distance
            for index, data in enumerate(self.sourceData):
                  listDistance = [self.calDistance(data, self.centerData[k], self.dim) for k in range(self.K)]
                  self.belongCluster[index] = listDistance.index(min(listDistance))

            countChangingCluster = self.countListDiff(latestBelongCluster, self.belongCluster) #計算ata的變動情況
            return countChangingCluster

      def updateCentroid(self):
            sse = 0
            for k in range(self.K):
                  index = [i for i,x in enumerate(self.belongCluster) if x == k]   # 找到在belongCluster中所有值為k的元素index
                  sse += sum([self.calDistance(element, self.centerData[k], self.dim) for element in self.sourceData[index,:]])
                  self.centerData[k] = numpy.mean(self.sourceData[index,:], 0)      # 找在此list index數據中的均值
            
            self.sum_of_square_error.append(sse)

      def calDistance(self, data, center, dim):
            retSum = 0
            retSum += sum([numpy.square(data[i]-center[i]) for i in range(len(data))])
            return retSum

      def countListDiff(self, list1, list2):
            retSum = 0
            for i in range(len(list1)):
                  if list1[i] != list2[i]:
                        retSum += 1
            return retSum
