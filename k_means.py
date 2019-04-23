import random
import numpy
import copy

class K_means():
      def __init__(self, sourceData, K):
            self.sourceData = sourceData
            self.K = K
            self.dim = sourceData.shape[1]
            self.centerData = []
            self.belongCluster = [-1]*sourceData.shape[0]
            self.sum_of_square_error = []

      def initCentroid(self):
            # Randomly select cluster center
            index = random.sample(range(self.sourceData.shape[0]), self.K)
            self.centerData = self.sourceData[index,:]

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
