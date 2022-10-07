# recipe-generation


## AMR Graphs

In order to read an AMR graph in penman notation from a file or write it to a file, the [penman library](https://github.com/goodmami/penman/) is used. The main part of the project, however, works with [networkX](https://networkx.org/documentation/stable/index.html) graph objects. The conversions between penman and networkX graphs of the AMRs are done using the functions from the [conversion script](https://github.com/interactive-cookbook/recipe-generation/blob/main/amr_processing/penman_networkx_conversions.py). 
The networkX AMR graphs have the following attributes:


**Graph attributes**
* 'graph': a dictionary with the complete metadata of the AMR
* example: if AMR file includes `::id waflles_0_instr0 ` then `AMR_Graph.graph['graph']['id'] = 'waffles_0_instr0'`
* additionally added information: 
  * `AMR_GRAPH.graph['graph']['label']` is the same as `AMR_GRAPH.graph['graph']['snt']`
  * `AMR_GRAPH.graph['graph']['root']`: the root node of the AMR 
* name of the graph is set to `AMR_Graph.graph['graph']['id']`


**Node attributes**
* 'label': label of the AMR node
* 'type': 'instance', indicating the original type of the triple in the AMR 
* 'epi': the epidata of the instance triple
* 'alignment': the aligned token ID (as string)
* 'attr': optional, is used if the node has attached attributes; dictionary itself
  * 'source': the variable of the node it is attached to
  * 'label': the relation / role of the edge (e.g. 'mode')
  * 'target': the label / value of the attribute (e.g. 'imperative')
  * 'epi': the epi data attribute triple
  * 'alignment': the aligned token ID (as string)


**Edge attributes**
* 'label': the edge label
* 'type': 'edge', indicating the original type of the triple in the AMR
* 'epi': the epidata of the edge triple
