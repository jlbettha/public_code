
import numpy as np

""" Betthauser - 2016 --  Compute k-means based on classwise Gaussians from data.  
% INPUTS: Atrain - training data (numF x N)
%         trainlabels - training data labels (1 x N)  
%         normalize data or not (0,1)
%         
% OUTPUT: [Atrain2, trainlabels2, totalcount] - new training data, labels,
%                                               and membership changes
"""
def kGMMcluster(data, orig_labels):
    num_classes = len( np.unique(orig_labels) )
    K = num_classes
    sizeTrain = size(Atrain,2)
    numFeatures = size(Atrain,1)
      
  
    # Begin GMM iterations
    notdone = 1
    totalcount = 0  
    while True:   
        [trainlabels, trainIndex] = sort(trainlabels)
        Atrain = Atrain(:,trainIndex)
        
         # Get indeces where class breaks occur in y (known classes) 
        count = 1
        currentY = trainlabels(1)
        breakIndeces = np.zeros(1,numClasses-1)
        classCounts = np.zeros(1,numClasses)
        for i in range(sizeTrain):
            classCounts(trainlabels(i))=classCounts(trainlabels(i))+1
            if trainlabels(i) != currentY # same class, keep incrementing
                breakIndeces(count) = i-1
                count = count+1        
            
            currentY = trainlabels(i)
        

        # get classwise data and stats
        for j in range(num_classes):
            # get appropriate section of A matrix
            if j == 1:
                 Aj(j).data = Atrain(:,1:breakIndeces(j)) 
            elif j == num_classes:
                 Aj(j).data = Atrain(:,(breakIndeces(j-1)+1):sizeTrain)            
            else:
                 Aj(j).data = Atrain(:,breakIndeces(j-1)+1:breakIndeces(j))
                 
             Aj(j).Mu = np.mean(Aj(j).data,2)
             Aj(j).Sigma = np.cov(Aj(j).data')
           
    
    
        switchcount = 0
        mahala = zeros(numClasses,size(Atrain,2));
        for i = 1:size(Atrain,2)
            x = Atrain(:,i)
            for j = 1:numClasses                   
                    mahala(j,i) = (x-Aj(j).Mu)'*pinv(Aj(j).Sigma)*(x-Aj(j).Mu)                
           
                    
            [~,ind] = sort(mahala(:,i));
            if ind(1) ~= trainlabels(i) && nnz(mahala(:,i))>0
                trainlabels(i) = ind(1)
                switchcount = switchcount + 1;
           
        
        if switchcount == 0:
            break
        else:
           totalcount = totalcount + switchcount;
       
    
    [trainlabels, trainIndex] = sort(trainlabels);
    Atrain = Atrain(:,trainIndex);
    return new_labels