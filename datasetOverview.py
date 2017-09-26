
# coding: utf-8

# In[ ]:

def datasetOverview(data_dict):
    
    '''
    This function outputs the number of data points and the
    information about pois in the enron dataset
    '''
    persons = 0 
    poi = 0
    notpoi = 0
    Nodataavailable=0
    for key,value in data_dict.iteritems():
        persons += 1
        for key1,value1 in value.iteritems(): 
        
            if key1 =='poi':
                if value1 == True:
                    poi += 1
                elif value1 == False:
                    notpoi += 1
                
                    
    print('Total Number of people under consideration :'+str(persons))
    print('Total Number of persons of interest under consideration :'+str(poi))
    print('Total Number of people who are not poi :'+str(notpoi))
            

