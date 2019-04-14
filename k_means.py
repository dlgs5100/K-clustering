#############################################################################
# Step 1. 隨機取K個點為群中心
# Step 2. 計算每個點到K個群中心距離，並取出最小的將其點歸入此群
# Step 3. 算出該群中所有點的新中心
# Step 4. 若群中所有點不再變動或達到最大迭代次數則為收斂，否則回到Step 2.狀態
#############################################################################
import random
import numpy
import copy

class K_means():
      def __init__(self, sourceData, K):
            self.sourceData = sourceData
            self.K = K
            self.centerData = []
            self.belongCluster = [-1]*sourceData.shape[0]
            self.sum_of_square_error = []

      def initCentroid(self):
            # Randomly select cluster center
            index = random.sample(range(self.sourceData.shape[0]), self.K)
            for i in index:
                  self.centerData.append(copy.deepcopy(list(self.sourceData[i])))

      def fitting(self):
            latestBelongCluster = self.belongCluster.copy()
            # 判斷點歸屬cluster, by distance
            for index, data in enumerate(self.sourceData):
                  minDistance = -1
                  for k in range(self.K):
                        dis = self.calDistance(data, self.centerData[k], self.sourceData.shape[1])
                        if minDistance == -1 or dis < minDistance:
                              minDistance = dis
                              self.belongCluster[index] = k

            countChangingCluster = self.countListDiff(latestBelongCluster, self.belongCluster) #計算ata的變動情況
            return countChangingCluster

      def updateCentroid(self):
            sse = 0
            for k in range(self.K):
                  tempSum = [0]*self.sourceData.shape[1]
                  index = [i for i,x in enumerate(self.belongCluster) if x == k]   # 找到在belongCluster中所有值為k的元素index
                  for dim in range(self.sourceData.shape[1]):    # 計算所有屬於cluster k的總和，藉此取平均找出新cluster center
                        for i in index:
                              tempSum[dim] += self.sourceData[i][dim]
                              sse += numpy.square(self.sourceData[i][dim]-self.centerData[k][dim])
                        self.centerData[k][dim] = tempSum[dim] / len(index)

            self.sum_of_square_error.append(sse)

      def calDistance(self, data, center, dim):
            retSum = 0
            for i in range(dim):
                  retSum += numpy.square(data[i]-center[i])
            return retSum

      def countListDiff(self, list1, list2):
            retSum = 0
            for i in range(len(list1)):
                  if list1[i] != list2[i]:
                        retSum += 1
            return retSum
