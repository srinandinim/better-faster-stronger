"""
THIS FILE COMPUTES U*(s), THE OPTIMAL VALUE FUNCTION, GIVEN AS INPUT A GRAPH. 
IT COMPUTES THE OPTIMAL VALUE FUNCTION ITERATIVELY, USING VALUE ITERATION AND DP.
"""
from .. import utils as utils 
from ..graph import Graph

# loads into memory the graph we are optimizing for
nbrs = utils.retrieve_graph()
graph = Graph(nbrs=nbrs)

# 