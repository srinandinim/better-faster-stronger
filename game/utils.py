import json
import os
from .graph import Graph

def save_graph(graph=Graph(nodes=50), filename="GAME_GRAPH.json"):
    ''' 
    Function to store a json from a given graph
    @param:graph - graph wished to be stored.  
    @param:filename - name of the json file to be stored to.  
    '''
    dirname = "graphs/"
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))

    filepath = dirname + filename
    with open(filepath, "w") as fp:
        json.dump(graph.nbrs, fp)

def retrieve_graph(filename="GAME_GRAPH.json"):
    ''' 
    Function to retrieve a json from a given file, intended to use on
    stored graphs. 
    @param:filename - name of the json file being loaded.  
    @return the dictionary loaded, intended to be a graph adjacency list
    '''
    def keysStrToInt(d):
        if isinstance(d, dict):
            return {int(k): v for k, v in d.items()}
        return d
    dirname = "graphs/"
    filepath = dirname + filename
    if os.path.exists(filepath):
        with open(filepath, "r") as fp:
            nbrs = json.load(fp, object_hook=keysStrToInt)
    return nbrs
