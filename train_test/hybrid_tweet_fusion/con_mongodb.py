# -*- coding: utf-8 -*-

import pymongo
import pandas as pd
from bson import ObjectId


class Connection_Mongo:  
    
    def __init__(self, server, mongodb_port):
        self.server = server
        self.mongodb_port = mongodb_port
        self.connection = pymongo.MongoClient("mongodb://"+server+":"+mongodb_port+"/")
        
    
    #********CREATE DATABASE********
    def create_db(self,name):
        
        my_database = self.connection[name]
        
        print('Database loaded!')
        
        return my_database
    
    #********DROP DATABASE********
    def drop_db(self,name):
    
        try:
            self.connection.drop_database(name)
            
            print('Deleted Database!')
            
        except ValueError as e:
            print('Error:', e)
            
        return(1)
        
        
    #********CREATE Collection********
    def create_collection(self, my_database, name_of_collection):
        
        try:
            my_collection = my_database[name_of_collection]
            
            print('Collection created!')
            
        except ValueError as e:
            print('Error:', e)
            
        return(my_collection)
        
     #********GET Collection********
    def get_collection(self, my_database, name_of_collection, query, remove_id):
        
        try:
            my_collection = my_database[name_of_collection]
            
            cursor = my_collection.find(query)
            
            dataframe =  pd.DataFrame(list(cursor))
            
            if remove_id:
                del dataframe['_id']

            
            print('Collection created!')
            
        except ValueError as e:
            print('Error:', e)
            
        return(dataframe)
    
    #********INSERT ONE Collection********
    def insert_one_collection(self, my_database, name_of_collection, vector_name, vector_data):
        
        try:
            
            if len(vector_name) == len(vector_data):
            
                my_collection = my_database[name_of_collection]
            
                dataframe = pd.DataFrame(columns = vector_name)
                
                index_dataframe = len(dataframe) + 1
                
                dataframe.loc[index_dataframe] = vector_data
                
                my_collection.insert_many(dataframe.to_dict('records'))
                
                print('Information successfully inserted in the collection!')
                
            else:
                
                print('Impossible to insert this information in the collection!')
        
        except ValueError as e:
            print('Error:', e)
        
        return(1)
    
     #********DROP ONE Collection********
    def drop_one_collection(self, my_database, name_of_collection, identifier):
        
        try:
                
            my_collection = my_database[name_of_collection]
            
            found = my_collection.find_one({"_id": ObjectId(identifier)})
            
            my_collection.delete_one({'_id': found['_id']})
            
            print('Information successfully removed!')
    
        except ValueError as e:
            print('Error:', e)
        
        return(1)    
    
    #********INSERT Collection PANDAS********
    def insert_collection_pandas(self, my_database, name_of_collection, vec_prec):
        
        try:
            
            for i in vec_prec:
                
                my_database[name_of_collection].insert_many(i.to_dict('records'))
            
                indexes = i.columns
                    
                my_database[name_of_collection].delete_many({indexes[0]: indexes[0]})
                
            print('Insertion of dataframe in the completed collection!')
            
        except ValueError as e:
            print('Error:', e)
        
        return(1)
        
    #********DELETE Collection********
    def drop_collection(self, my_database, name_of_collection):
        
        try:
            my_database.drop_collection(name_of_collection)
            
            print('Collection removal completed!')
            
        except ValueError as e:
            print('Error:', e)
            
        return(1)
        
    #********CLEAR Collection********
    def clear_collection(self, my_database, name_of_collection):
        
        try:
            my_database[name_of_collection].delete_many({})
            
            print('Removal of completed collection!')
            
        except ValueError as e:
            print('Error:', e)
            
        return(1)
        

