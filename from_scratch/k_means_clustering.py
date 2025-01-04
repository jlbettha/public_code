""" Betthauser - 2016 --  Determine class means
% INPUTS: Atrain - training data (numF x N)
%         trainlabels - training data labels (1 x N)  
%         normalize - scale to 0-1 range
%         
% OUTPUT: [trainMeans, trainMeansPDF] - class means, class means as scaled PDF
"""
def getClassMeans(Atrain, trainlabels, normalize):
    numClasses = length(unique(trainlabels));
    sizeTrain = size(Atrain,2); 
    numFeatures = size(Atrain,1); 
    
    # Get indeces where class breaks occur in y (known classes) 
    [breakIndeces] = getBreakIndeces(Atrain, trainlabels)
    
    if normalize:  
        trainMax = max(Atrain,[],2); 
        Atrain = bsxfun(@rdivide, Atrain, trainMax)
    
    
    trainMeans = zeros(numFeatures,numClasses)
    trainMeansPDF = zeros(numFeatures,numClasses)
    for i = 1:sizeTrain        
        for j = 1:numClasses
            # get appropriate section of A matrix
            if j == 1:
                data = Atrain(:,1:breakIndeces(j)); 
            elif j == numClasses:
                data = Atrain(:,(breakIndeces(j-1)+1):sizeTrain)            
            else:
                data = Atrain(:,breakIndeces(j-1)+1:breakIndeces(j))
            
            trainMeans(:,j) = mean(data,2)  
            trainMeansPDF(:,j) = trainMeans(:,j) / sum(trainMeans(:,j))
       