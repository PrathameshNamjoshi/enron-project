
# coding: utf-8

# In[ ]:

def removeOutliers(data_dict):
    '''
    Function to remove spreadsheet outlier
    
    '''
    
    import matplotlib.pyplot as pl  #Import matplotlib to plot scatterplot
    #plot salary vs bonus
    for name in data_dict:

        a=(data_dict[name]['salary'])

        b=(data_dict[name]['bonus'])

        pl.scatter(a,b)

    print 'Plot with outlier'

    pl.show()    

    print 'The \'Total\' outlier :'
    
    # The outlier above is a spreadsheet error!
    # Another outlier with  all'NaN' values removed.
    print data_dict["TOTAL"] 
    print('')
    data_dict.pop('TOTAL')
    print 'This Record of EUGENE E LOCKHART has no useful data'
    print data_dict["LOCKHART EUGENE E"]
    print('')
    data_dict.pop('LOCKHART EUGENE E')
    print '\n Outliers Removed'
    print '\n New Plot:'
    
    # plot again after removal of outlier
    
    for name in data_dict:
        a=(data_dict[name]['salary'])
        b=(data_dict[name]['bonus'])
        pl.scatter(a,b)
    pl.show()
    
    return data_dict
   

