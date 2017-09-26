
# coding: utf-8

# In[ ]:

def featuresAdd(data_dict):
    '''
    This function adds a feature to the dataset called 'proportion_of_poi_mssgs'
    Instead of knowing how many messages were sent or received from the poi, it would be more
    important to know what proportion of total messages were sent or received from poi
    
    '''
    for name in data_dict:
        if data_dict[name]['from_poi_to_this_person'] != 'NaN' and         data_dict[name]['from_this_person_to_poi'] != 'NaN' and  data_dict[name]['to_messages'] != 'NaN' and         data_dict[name]['from_messages'] != 'NaN':

            total_msgs_with_poi=int(data_dict[name]['from_poi_to_this_person'])            +int(data_dict[name]['from_this_person_to_poi'])
            
            total_msgs=int(data_dict[name]['to_messages'])+            int(data_dict[name]['from_messages'])
            
            try:

                data_dict[name]['proportion_of_poi_mssgs']=round(float(total_msgs_with_poi)/float(total_msgs),3)

            except:
                data_dict[name]['proportion_of_poi_mssgs'] ='NaN'
        else:
             data_dict[name]['proportion_of_poi_mssgs'] ='NaN'

        
    my_dataset = data_dict
     
    
    return my_dataset

  

