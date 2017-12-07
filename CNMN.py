# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 18:19:40 2017

@author: Jingming Zhang
"""
from random import sample
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.Graph(nx.read_pajek('datasets/karate.net'))
G = nx.Graph(nx.read_edgelist('datasets/email.edgelist'))
#G = nx.Graph(nx.read_pajek('datasets/simulation.net'))
#G = nx.Graph(nx.read_pajek('datasets/football_12.net'))

nodes = G.nodes()

node_num = len(nodes)
Neighbor = {node:G.neighbors(node) for node in nodes}
CommonNeighbor = np.zeros((node_num,node_num))
A = np.zeros((node_num,node_num))
Sim = np.zeros((node_num, node_num))

#计算图的邻接矩阵和任意两个顶点之间的公共邻居
for nodei in nodes:
    for nodej in nodes:
        nodei_index = nodes.index(nodei)
        nodej_index = nodes.index(nodej)
        CommonNeighbor[nodei_index][nodej_index] = len(set(Neighbor[nodei]) & set(Neighbor[nodej]))
        if (nodei in Neighbor[nodej]) or (nodej in Neighbor[nodei]):
            A[nodei_index][nodej_index] = 1

#根据苏醒那篇论文提到的相似性计算公式计算相似性
for nodei in nodes:
    for nodej in nodes:
        nodei_index = nodes.index(nodei)
        nodej_index = nodes.index(nodej)
        Sim[nodei_index][nodej_index] = A[nodei_index][nodej_index] + CommonNeighbor[nodei_index][nodej_index]

#建立顶点的度的字典
degreeNum = {}
for k,v in G.degree_iter():
    degreeNum[k] = v

#degreesmall = []
real_nodes = ['18','22']
#real_nodes = ['60','43']
#for i,j in degreeNum.items():
#    if j <= 3:
#        degreesmall.append(i)
#real_nodes = sample(degreesmall, 5)

remain_nodes = list(set(nodes).difference(set(real_nodes)))

pos = nx.spring_layout(G)
nx.draw(G,pos,with_labels = True)
plt.savefig("karate1.png")    
nx.draw_networkx_nodes(G, pos, nodelist=real_nodes, node_shape='s',node_color='b')
nx.draw_networkx_nodes(G, pos, nodelist=remain_nodes, node_shape='o',nodes_color='r')
nx.draw_networkx_edges(G, pos, with_labels = False)
plt.savefig("karate.png")
plt.show()

node_id = max(int(node) for node in nodes) + 1

#删除给出的顶点，并添加占位符和对应的边
nodedict = {}
addednode = []
for del_node in real_nodes:
    placeholder = []
    neig = G.neighbors(del_node)
    G.remove_node(del_node)
    for ng in neig:
        i = str(node_id)
        G.add_node(i)
        addednode.append(i)
        G.add_edge(ng,i)
        placeholder.append(i)
        node_id += 1
    nodedict[del_node] = placeholder
print nodedict
ChangedNodes = G.nodes() 
ChangedNeighbor = {node:G.neighbors(node) for node in ChangedNodes}
addednodeNeig = {}
for node in addednode:
    addednodeNeig[node] = ChangedNeighbor[node]

reverseaddedNodeNeig = {str(v):k for k,v in addednodeNeig.items()}

#计算相似性闭包
C = []
for i in addednodeNeig.values():
    Ct = set()
    for j in addednodeNeig.values():
        if i[0] != j[0]:
            i_index = nodes.index(i[0])
            j_index = nodes.index(j[0])
            if Sim[i_index][j_index] > 4:
                Ct.add(reverseaddedNodeNeig[str(i)])
                Ct.add(reverseaddedNodeNeig[str(j)])
    C.append(Ct)
print C

    
        




























































