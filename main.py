from sklearn import datasets, metrics
from k_meansPP import K_meansPP
from k_means import K_means
import matplotlib.pyplot as plt
import numpy
import time
import csv
import sys
import os

def main():
      timeData = []
      silhouette_scoreData = []
      sum_of_square_errorData = []

      sourceData = inputDataset()
      choose = int(input('Determine the clustering algorithm, 1) K-means, 2) K-means++：'))
      N = int(input('Determine running time：'))
      K1, K2 = map(int,input('Determine K range, (ex1:2 10, K=2~10; ex2:3 3, K=3)：').split())
      try:  
            deletePreviousOutputFile()
            for n_th in range(N):
                  timeData.append([])
                  silhouette_scoreData.append([])
                  sum_of_square_errorData.append([])
                  for K in range(K1,K2+1,1):
                        if choose == 1:
                              k_cluster = K_means(sourceData, K)
                        elif choose == 2:
                              k_cluster = K_meansPP(sourceData, K)
                        
                        start = time.time()
                        k_cluster.initCentroid()  # Step 1.
                        iter = 0
                        while True:
                              countChangingCluster = k_cluster.fitting() # Step 2.
                              k_cluster.updateCentroid() # Step 3.
                              if iter > 50 or countChangingCluster == 0:
                                    break
                              iter+=1

                        end = time.time()

                        print('Round: '+str(n_th+1)+', K: '+str(K)+' done.')
                        
                        timeData[n_th].append(end-start)
                        silhouette_scoreData[n_th].append(metrics.silhouette_score(sourceData, k_cluster.belongCluster))
                        sum_of_square_errorData[n_th].append(k_cluster.sum_of_square_error[-1])
                        outputResult(K, iter, end-start, metrics.silhouette_score(sourceData, k_cluster.belongCluster), k_cluster.sum_of_square_error)
            outputPlot(K1, K2, numpy.array(timeData), numpy.array(silhouette_scoreData), numpy.array(sum_of_square_errorData))
            
      except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


def inputDataset():
      dataset = input('Choose dataset, 1) Iris, 2) Abalone：')
      if dataset == '1':
            #   iris = datasets.load_iris()
            #   sourceData = iris.data[:, :4]     # shape[0]:資料數，shape[1]:資料維度
            with open('iris.csv', 'r') as f:
                  reader = csv.reader(f)
                  sourceData = list(reader)
            sourceData = [sourceData[i][:4] for i in range(len(sourceData)-1)]
            for i in range(len(sourceData)):
                  sourceData[i] = [float(x) if True else x for x in sourceData[i]]
            sourceData = numpy.asarray(sourceData)
      elif dataset == '2':
            with open('abalone.csv', 'r') as f:
                  reader = csv.reader(f)
                  sourceData = list(reader)
            for i in range(len(sourceData)):
                  sourceData[i] = [str(ord(x)) if x.isalpha()else x for x in sourceData[i]]
                  sourceData[i] = [float(x) if True else x for x in sourceData[i]]
            sourceData = numpy.asarray(sourceData)
        
      return sourceData

def deletePreviousOutputFile():
      try:
            os.remove('result.txt')
      except OSError as e:
            None

def outputResult(K, iter, time, silhouette_score, sum_of_square_error):
      with open('result.txt', 'a') as file:
            file.write('K: %s\n' % str(K))
            file.write('Cost: %s iterations.\n' % str(iter))
            file.write('Time: %s s\n' % str(time))
            file.write('Silhouette_score: %s\n' % str(silhouette_score))
            file.write('Sum_of_square_error:\n')
            for data in sum_of_square_error:
                  file.write('%s\n' % str(data))
            file.write('*--------------------------*\n')
            file.close()

def outputPlot(K1, K2, timeData, silhouette_scoreData, sum_of_square_errorData):

      plt.xlabel("number of clusters")
      plt.ylabel("Time")
      plt.xticks(numpy.arange(K1, K2+1, 1))
      plt.bar(numpy.arange(K1, K2+1, 1), numpy.mean(timeData, 0)) 
      fig = plt.gcf()
      fig.savefig('time.png', dpi=100)

      plt.clf()
      plt.xlabel("number of clusters")
      plt.ylabel("Silhouette_score")
      plt.xticks(numpy.arange(K1, K2+1, 1))
      plt.bar(numpy.arange(K1, K2+1, 1), numpy.mean(silhouette_scoreData, 0)) 
      fig = plt.gcf()
      fig.savefig('silhouette_score.png', dpi=100)

      plt.clf()
      plt.xlabel("number of clusters")
      plt.ylabel("Sum_of_square_error")
      plt.xticks(numpy.arange(K1, K2+1, 1))
      plt.plot(numpy.arange(K1, K2+1, 1), numpy.mean(sum_of_square_errorData, 0), marker = 'o') 
      fig = plt.gcf()
      fig.savefig('sum_of_square_error.png', dpi=100)

if __name__ == "__main__":
      main()