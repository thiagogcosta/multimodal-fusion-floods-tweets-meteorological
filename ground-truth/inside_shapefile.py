from shapely.geometry import Point

class InsideShape:
    
    def In_shapefile(self, df, shapes):
        
        in_shapefile_sao_paulo = 0
        
        for aux_shape in shapes.geometry:
            if(df.within(aux_shape)):
                in_shapefile_sao_paulo = 1
        return in_shapefile_sao_paulo
    
    def __init__(self,geographic_information,shape):
        
        geographic_information['inside'] = 0
        
        count_flood = 0
        
        list_of_points = geographic_information.index.values.tolist()
        
        # 
        while(count_flood < len(list_of_points)):
        
            df = Point(geographic_information.loc[list_of_points[count_flood]]['longitude'],geographic_information.loc[list_of_points[count_flood]]['latitude'])
            
            if(self.In_shapefile(df, shape)):
                geographic_information.loc[list_of_points[count_flood],'inside'] = 1
            else:
                geographic_information.loc[list_of_points[count_flood],'inside'] = 0
                
            count_flood +=1
        
        # Return only geographic information inside in the city of São Paulo
        geographic_information = geographic_information[geographic_information['inside'] != 0]
        
        geographic_information = geographic_information.drop('inside', 1)
        
        geographic_information = geographic_information.reset_index(drop=True)
        
        print(geographic_information)
        
        self.geographic_information = geographic_information

