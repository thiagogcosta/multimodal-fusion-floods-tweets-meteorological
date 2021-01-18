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
        
        points_vector = geographic_information.index.values.tolist()

        inside_ok = 0
        
        inside_not_ok = 0
        
        for item in points_vector:
        
            df = Point(geographic_information.loc[item]['longitude'],geographic_information.loc[item]['latitude'])
            
            if(self.In_shapefile(df, shape)):
                geographic_information.loc[item,'inside'] = 1
                
                inside_ok+=1
            else:
                geographic_information.loc[item,'inside'] = 0
                
                inside_not_ok+=1
        
        # Return only geographic information inside in the city of São Paulo
        geographic_information = geographic_information[geographic_information['inside'] != 0]
        
        geographic_information = geographic_information.drop('inside', 1)
        
        geographic_information = geographic_information.reset_index(drop=True)
        
        print('Inside shapeFile: ', inside_ok)

        print('Not inside shapeFile: ', inside_not_ok)
        
        print(geographic_information)
        
        self.geographic_information = geographic_information