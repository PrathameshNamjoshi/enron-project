
# coding: utf-8

# In[ ]:


def Params_dict(classifiers):
    import numpy as np
    ''''
    In this function a dictionary has been created about the parameters which 
    would be passed for each classsifier during the 'grid search'.
    '''
   
   # This list will be common to all classifiers and includes parameters passed for PCA 
   # and for slecting K best features from feature_list  

    featureParamList = dict( reduce__n_components=np.arange(1,4),
                               reduce__whiten =[True, 
                                                     False],
                               reduce__svd_solver =['auto', 
                                                         'full', 
                                                         'arpack', 
                                                         'randomized'],
                               selectK__k =[5,10,15])
    #parameters for SVC
    svcParam = dict(clf__C = [0.0001, 
                                0.001, 
                                0.01, 
                                0.1, 
                                1,10],
                          clf__gamma = [0.0005, 
                                        0.001, 
                                        0.005, 
                                        0.01, 
                                        0.1],
                          clf__kernel= ['rbf','linear'], 
                          clf__class_weight = ['balanced', 
                                               None])
    
    #parameters for DecisionTree
    decisionTreeParam = dict(clf__criterion = ['gini', 
                                                  'entropy'],
                                clf__max_features = ['sqrt', 
                                                     'log2', 
                                                     None],
                            clf__random_state=[56])
    #Parameters for RandomForest
    randomForestParam = dict(clf__n_estimators = np.arange(10, 30 ,10),
                                 clf__criterion = ['gini', 
                                                   'entropy'],
                                 clf__max_features = ['sqrt', 
                                                      'log2', 
                                                      None],
                                 clf__class_weight = ['balanced', 
                                                      None],
                                clf__random_state = [56])
    
    algoParametersList=[svcParam,decisionTreeParam,randomForestParam]
    
    
    ''' (In Main function)
    
     classifiers = [SVC(),
                   DecisionTreeClassifier(),
                   RandomForestClassifier(),
                   GaussianNB()]
    '''
    parameterDict={}
    #Avoiding GaussianNB in classifiers list in main function as it does not have its own parameters 
    #to tune hence -1
    for i,_ in  enumerate(classifiers[:-1]) :    
        
        algoParametersList[i].update(featureParamList)
        parameterDict.update({type(classifiers[i]):algoParametersList[i]})

    #for Gaussian Naive Bayes
    parameterDict.update({type(classifiers[3]):featureParamList})
    
    return parameterDict

