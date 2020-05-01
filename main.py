import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2
from sklearn import tree

D=np.empty([400,10304])
DTrain=np.empty([200,10304])
DTest=np.empty([200,10304])
L=np.empty([400,1])
L=np.empty([200,1])
L=np.empty([200,1])

for i in range(1,41):
    for j in range(1,11):
        #print("File number : "+str(i))
        #print("Photo number : "+str(j))
        s="C:/Users/M3MO/Desktop/term 8/pattern/orl_faces/s"+str(i)+"/"+str(j)+".pgm"
        img1=cv2.imread(s,0)  
        img1=np.concatenate(img1)
        
        if  (i==1 and j==2):
            D=np.vstack((img2,img1)) 

        img2=cv2.imread(s,0)  
        img2=np.concatenate(img2)

        if not (i==1 and (j==1 or j==2)):
            D=np.vstack((D,img2))
        #print(img2)
        #cv2.imshow('image',img1)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()      
#print ("D concatenated array")        
#print(D) 
#print (np.shape(D))
DTest=np.vstack((D[0],D[2]))
L=np.vstack((1,1))
DTrain=np.vstack((D[1],D[3]))
for i in range(1,41) :
    if (i==1):
        for j in range(1,9):
            L=np.vstack((L,i))  
    else:       
        for j in range(1,11):
            L=np.vstack((L,i))
            
LTrain=np.vstack((1,1)) 
LTest=np.vstack((1,1)) 


count =0
for i in range(4,400) :
    if (i%2==0):    
        DTest=np.vstack((DTest,D[i]))
        LTest=np.vstack((LTest,L[i]))
       
    else :
        DTrain=np.vstack((DTrain,D[i]))
        LTrain=np.vstack((LTrain,L[i]))


print ("DTrain array")        
print(DTrain[0:5,:])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(DTrain,LTrain)
a=clf.predict(DTest)
count=0
for i in range (0,len(LTest)):
    if (LTest[i]== a[i]):
        count = count+1
    
    
print(a)



#print (np.shape(DTrain))
#print ("DTest array")        
#print(DTest) 
#print (np.shape(DTest)[0])          
#for ev in eig_vecs:
    #np.testing.assert_array_almost_equal(np.linalg.norm(ev),1.0, err_msg='errrrrrrrrrorrrr')
#print('Everything ok!')
 
   
#mid=[10304][10303]

def GetMidPoints (variables,numOfVariables):
    mid=np.empty([numOfVariables,1])
    for i in range (0,len(variables)-1):
        #print(features[i])
        #print(features[i+1])
        
        #print(features)
        mid[i] = (int(variables[i])+int(variables[i+1]))/2.0
        #print(i)
    
    return mid 

#def BestMean (mid,Train,numFeature,numOfClasses,labels):
    #c1=np.empty([numOfClasses,1])
    #c2=np.empty([numOfClasses,1])
    #infoGain=np.empty([len(mid),1])
    #for i in range (0,len(mid)):
        #for j in range (0,Train.shape[0]):   
            #if (Train[j,numFeature]>=temp):
                #c1[label[j],1]=c1[label[j],1]+1
                
            #else:
                #c2[label[j],1]=c2[label[j],1]+1
                #infoGain[i,1]=informationGain(c1,c2);
                
    #max(infoGain)
        
        
#return    
        
                    


#mid=np.empty([10304,10303])
#for i in range (0,10304):
    
   #temp=GetMidPoints(DTrain[:,i])
   #if(i==0):
       #print(DTrain[:,i])
       #print(temp.T)       
   
   #mid[i]=temp.T
   
#for i in range(0,10304):
    #print("mid ")
    #print(mid[i])
    
    
    ##from here
def PCA (data , alpha):   
    
    mean_vec = np.mean(data, axis=0) 
    print(data.shape)
    print(mean_vec.shape)
    print("mean")
    print(mean_vec)
    print("diff")
    print(data- mean_vec)
    
    #cov_mat = (data- mean_vec).T.dot((data - mean_vec)) / (data.shape[0]-1)
    #print('Covariance matrix \n%s' %cov_mat)
    cov_mat = np.cov(data.T)
    print('NumPy covariance matrix: \n%s' %cov_mat)  
    print("cov shape" , np.shape(cov_mat) )
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    print('Eigenvectors \n%s' %eig_vecs)
    print("eig_vecs shape" , np.shape(eig_vecs))
    print('\nEigenvalues \n%s' %eig_vals) 
    print("eig_vals shape" , np.shape(eig_vals) )
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    eig_pairs.sort()
    eig_pairs.reverse()   
    
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    for i in range(0,cum_var_exp.shape[0]):
       # print (i," ",cum_var_exp[i])
        if (cum_var_exp[i]>=alpha):
            j=i
            break

    print ("j= " , j)
    matrix_w = np.hstack((eig_pairs[0][1].reshape(10304,1), 
                          eig_pairs[1][1].reshape(10304,1)))
    for i in range(2,j):
        matrix_w = np.hstack((matrix_w , eig_pairs[i][1].reshape(10304,1)))
    print('Matrix W:\n', matrix_w)
    return matrix_w
        
        
    
matrix_w=PCA(DTrain,95)
WTrain= DTrain.dot(matrix_w)
print("shape wtrain" ,np.shape(WTrain))
WTest= DTest.dot(matrix_w)
print("shape wtest" ,np.shape(WTest))
#print("new data ", NewData)

##LTrain=LTrain.reshape((1,200))
##print (LTrain.reshape((1,200)) )
##knn=KNeighborsClassifier(n_neighbors=1)
##knn.fit(WTrain,LTrain)
##predictions= knn.predict(WTest) 
##count=0
##for i in range (len(predictions)):
    ##if predictions[i]==LTest[i]:
        ##count=count+1
##print('first KNN')
##print(predictions)
##print('Accuracy:')
##print(count/len(predictions))

k_range = [1,3,5,7]
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(WTrain,LTrain)
    predictions= knn.predict(WTest)   
    scores.append(metrics.accuracy_score(LTest, predictions))

##%matplotlib inline

##plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()
print ( "Predictions ")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(WTrain,LTrain)
predictions= knn.predict(WTest)  
print (predictions )
print ("Accuracies for k = 1 , 3 , 5 , 7 ")
print (scores)

###till here 

## from here
def LDA (Data):
    mean_vectors = []
    suum=0
    for i in range(0,40):
        mean_vectors.append(np.mean(Data[5*i:(5*i)+5,:], axis=0))
        #print('Mean Vector class %s: %s\n' %(i, mean_vectors[i]))  
        
  
        ##The within-class scatter matrix SW
    S_W = np.zeros((10304,10304))
    for cl,mv in zip(range(1,40), mean_vectors):
    #for rowm in mean_vectors:    
        class_sc_mat = np.zeros((10304,10304)) 
       # for i in range (0,40):
        for row in Data[5*(cl-1):(5*(cl-1))+5,:]:
                suum=suum+1
                #print ("row shape " , np.shape(row)[0])
                #print(row)
                row, mv = row.reshape(10304,1), mv.reshape(10304,1) 
                class_sc_mat += ((row-mv)).dot((row-mv).T)
        S_W += class_sc_mat                             
    print('within-class Scatter Matrix:\n', S_W)          
        
     
   ##between class scatter matrix B
    
    overall_mean = np.mean(Data[:,:], axis=0)
    print(overall_mean.size)
    print("overall mean " , overall_mean)
    S_B = np.zeros((10304,10304))
    for i,mean_vec in enumerate(mean_vectors):
        n = 5
        mean_vec = mean_vec.reshape(10304,1) # make column vector
        overall_mean = overall_mean.reshape(10304,1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
   
    print('between-class Scatter Matrix:\n', S_B)   
   
    #S_W = np.linalg.pinv(S_W)
    eig_vals, eig_vecs = np.linalg.eigh(np.linalg.inv(S_W).dot(S_B))

    
    print('Eigenvectors \n%s' %eig_vecs)
    print("eig_vecs shape" , np.shape(eig_vecs))
    print('\nEigenvalues \n%s' %eig_vals) 
    print("eig_vals shape" , np.shape(eig_vals) )
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    eig_pairs.sort()
    eig_pairs.reverse()   
    
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
   
    matrix_w = np.hstack((eig_pairs[0][1].reshape(10304,1), 
                          eig_pairs[1][1].reshape(10304,1)))
    for i in range(2,39):
        matrix_w = np.hstack((matrix_w , eig_pairs[i][1].reshape(10304,1)))
    print('Matrix W:\n', matrix_w)
    return matrix_w 

def NaiveBayes (Data):
    mean_vectors = []
    suum=0
    for i in range(0,40):
        mean_vectors.append(np.mean(Data[5*i:(5*i)+5,:], axis=0))  


  

def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)        
    
     
    


 
matrix_w=LDA(DTrain)
WTrain= DTrain.dot(matrix_w)
print("shape wtrain" ,np.shape(WTrain))
WTest= DTest.dot(matrix_w)
print("shape wtest" ,np.shape(WTest))



k_range = [1,3,5,7]
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(WTrain,LTrain)
    predictions= knn.predict(WTest)   
    scores.append(metrics.accuracy_score(LTest, predictions))


print ( scores )

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()
print ( "Predictions ")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(WTrain,LTrain)
predictions= knn.predict(WTest)  
print (predictions )
print ("Accuracies for k = 1 , 3 , 5 , 7 ")
print (scores)





