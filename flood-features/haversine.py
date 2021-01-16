import math

class Haversine:

    def __init__(self,coordinates_1,coordinates_2):
        
        longitude_1,latitude_1=coordinates_1          # longitude and latitude of the first geographical point
        longitude_2,latitude_2=coordinates_2          # longitude and latitude of the second geographical point
        
        radius = 6371000                               #  Radius of the earth
        first_phi = math.radians(latitude_1)
        second_phi = math.radians(latitude_2)

        delta_1 = math.radians(latitude_2-latitude_1)   # first delta
        delta_2 = math.radians(longitude_2-longitude_1) # second delta

        aux = math.sin(delta_1/2.0)**2 + math.cos(first_phi)*math.cos(second_phi) * math.sin(delta_2/2.0)**2
        result = 2*math.atan2(math.sqrt(aux),math.sqrt(1-aux))
        
        self.meters =radius* result              # Output in meters
        self.kilometers =self.meters/1000.0      # Output in kilometers