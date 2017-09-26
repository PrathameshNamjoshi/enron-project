
# coding: utf-8

# In[ ]:

def EstimateBest(classifier,params_list, cross_val, my_dataset, features_list,
                   features_train, labels_train, features_test, labels_test):
    
    
    
    
    from sklearn import preprocessing
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.grid_search import GridSearchCV
    from sklearn.feature_selection import SelectKBest
    from sklearn import metrics
    
    parameterlist=params_list[type(classifier)]

    print('\n'+str(type(classifier)))
    
    print('\n wait....')

    # Create pipeline and pass the series of operations we expect to take place with a classifier 
    # at the end.
    
    estimators = [('scaler', preprocessing.MinMaxScaler()),
                  ('selectK', SelectKBest()),
                  ('reduce', PCA()), 
                  ('clf', classifier)]
    pipe = Pipeline(estimators) 
    
    #GridSearch algorithm to tune the dataset
    
    grid = GridSearchCV(pipe, 
                        param_grid = parameterlist, 
                        scoring = 'f1',
                        cv = cross_val)

    
    grid.fit(features_train, labels_train)
    
    #select the predictions of best_estimator classifier selected on basis of f1 scoring
    
    pred = grid.best_estimator_.predict(features_test)
    
    clf=grid.best_estimator_

    clf.fit(features_train, labels_train)

    pred = clf.predict(features_test)

    
    acc = clf.score(features_test, labels_test)
    prec = metrics.precision_score(labels_test, pred)
    rcl = metrics.recall_score(labels_test, pred)
    f1score = metrics.f1_score(labels_test, pred)

    print ('\n {} \n Accuracy = {} \n Precision = {} \n Recall = {} \n F1 Score = {} ').format(type(clf),
                                                                                         str(acc),
                                                                                         str(prec),
                                                                                         str(rcl),
                                                                                         str(f1score))


    '''
    Reference
    http://scikit-learn.org/stable/modules/pipeline.html
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    
    '''
    #Show features selected for best_estimator and their weightage respectively
    
    feature_yesno = clf.named_steps['selectK'].get_support()
    
    #feature_yesno is a list of booleans correspoinding to the fearures list, where 
    # 1 represents feature selected and 0 represents feature rejected in best_estimator
    
    feature_weightage = clf.named_steps['selectK'].scores_
    print ('\n Feature:Weightage')
    
    for i in range(len(features_list)-1): 
        # since 'poi' is label and not feature we use 'length - 1'
        if feature_yesno[i]:
            print ('\n {}:{}').format(features_list[i+1],feature_weightage[i]) 
            #since 'poi' is label and not feature we use i+1

    
    return clf

