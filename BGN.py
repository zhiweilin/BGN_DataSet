# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:29:54 2018

@author: 林志伟 
"""

from __future__ import division
import networkx as nx
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
import time
import heapq
import math
import csv
'''
输入 不同的网络
输出 BGN vs state-of-art不同算法的排序结果 R值 R曲线 运行时间
输出 BGN 和 init-BGN的对比  R值 排序时间
'''

def MDD():   
    '''
    sample network
    '''
    G = nx.Graph() 
    G.add_edge(0,1) 
    G.add_edge(0,2)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(3,4)
    G.add_edge(4,5)
    G.add_edge(4,6)
    #pos = nx.shell_layout(G)
    #nx.draw(G,pos,with_labels = True ,node_size = 50)
    #plt.show()
    return G

def loadGraph (filename,numfile):
    '''
    输入csv文件
    输出网络G
    '''
    df = pd.read_csv(filename)
    arr = np.array(df)
    row =  arr.shape[0]
    G = nx.Graph() 
    for i in range(row):
        G.add_edge(arr[i,0],arr[i,1])
    #WS8  CB 9  BA 7 Random 5
    first_edge = np.array([[0,1],[0,30],[0,23346],[0,3],[0,1],[0,1],[0,1],[0,10528],[0,12696],[0,273649]])
    G.add_edge(first_edge[numfile,0],first_edge[numfile,1])
    print (first_edge[numfile,0],first_edge[numfile,1])
    return G



def computeR1(li,G):
    '''
    计算网络的鲁棒性指标
    '''
    len_li = len(li)
    Cluster = [set()]*len_li#用以记录每一个集团的成员节点
    NodeCluster = [0]*len(G)#记录每一个节点所属集团的ID
    newClusterId = 1 #新集团ID初始化为1
    maxcluster = 1 #最大集团的节点数
    sumcluster = 0 #每一次删点的最大连通集团数之和
    sandiany = []
    for i in range(0,len_li)[::-1]:#访问节点重要性低的       
        vi = li[i]                  #节点vi
        ci  = set()                 #记录vi可能连接的集团的ID集合
        for vj in nx.all_neighbors(G,vi):
            if NodeCluster[vj] != 0: #vj已属于某个集团
                ci.add(NodeCluster[vj])
        if len(ci) == 0:
            NodeCluster[vi] = newClusterId #分配给新的集团
            Cluster[newClusterId] = set()   #新集团
            Cluster[newClusterId].add(vi)   #将vi加入
            newClusterId = newClusterId + 1
        else:
            minci = min(ci)# 记录Ci中编号最小的集团ID，记为minci 
            NodeCluster[vi] = minci #节点vi所属集团的ID minci
            Cluster[minci].add(vi) #将节点vi加入到编号为minci的集团中
            #将Ci中所有集团的成员节点都合并，均放入minci中
            for name in  ci:
                if name != minci:
                    for nod in Cluster[name]:
                        NodeCluster[nod] = minci 
                        Cluster[minci].add(nod) 
            if len(Cluster[minci]) > maxcluster:
                maxcluster = len(Cluster[minci])
        #print maxcluster
        sandiany.append(maxcluster)
        sumcluster = sumcluster + maxcluster
    rnum = (sumcluster - len_li)/len_li/len_li
    sandiany.append(len_li)
    sandiany=[x/len_li for x in sandiany]
    sandianx  = [0]*(len_li+1)
    for i in range(len_li+1):
        sandianx[i] = (len_li-i)/len_li     
#==============================================================================
#     fig = plt.figure()  
#     ax1 = fig.add_subplot(111)  
#     #设置标题  
#     ax1.set_title('Robustness')  
#     #设置X轴标签  
#     plt.xlabel('p',fontsize=12)  
#     #设置Y轴标签  
#     plt.ylabel('$\sigma$',fontsize=12)  
#     #画散点图  
#     #ax1.scatter(sandianx,sandiany,c = 'b',marker = '_')  
#     ax1.plot(sandianx,sandiany,'b')
#     #设置图标  
#     foo_fig = plt.gcf() # 'get current figure'
#     foo_fig.savefig('R.eps', format='eps', dpi=1000)
#     plt.show()
#==============================================================================
    return  rnum,sandianx,sandiany
    
    
def sortNode(G):    
    '''
    按度从大到小排序
    '''
    matG = []
    nodeG = nx.nodes(G) 
    matG.append(nodeG)
    degreeG = nx.degree(G).values()  
    matG.append(degreeG)
    #print matG
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  
    li =  result[0]                    
    return li     
    
    
def sortNode1(G):
    '''
    K-shell 排序
    '''
    matG = []
    nodeG = nx.nodes(G)  #节点编号
    matG.append(nodeG)
    kshell =  nx.core_number(G).values()    #节点的
    #print (kshell)
    matG.append(kshell)
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  #按节点的从大到小排
    li =  result[0]                       #输出节点大的编号依次减小
    li = li.astype(np.int32)
    return  li
    
    
def sortNode2(G):  
    '''
    PageRank 排序
    '''
    matG = []
    nodeG = nx.nodes(G)  #节点编号
    matG.append(nodeG)
    pagerankG  = nx.pagerank(G,alpha=1).values()    #节点的pagerank
    #print pagerankG
    matG.append(pagerankG)
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  #按节点的PR从大到小排
    li =  result[0]                       #输出节点PR大的编号依次减小
    li = li.astype(np.int32)
    return  li


def sortNode8(G): #介数
    '''
    介数排序
    '''
    matG = []
    nodeG = nx.nodes(G)  #节点编号
    matG.append(nodeG)
    betweenness = nx.betweenness_centrality(G,normalized = False)  #betweeness  
    matG.append(betweenness.values()) 
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  
    li =  result[0]                       
    li = li.astype(np.int32)  
    return  li

def sortNode9(G): #接近中心性
    '''
    接近中心性
    '''
    matG = []
    nodeG = nx.nodes(G)  #节点编号
    matG.append(nodeG)
    closeness = nx.closeness_centrality(G,normalized = False)      #closeness
    matG.append(closeness.values())  
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  
    li =  result[0]                       
    li = li.astype(np.int32)  
    return  li

def sortNode6(G):   #Clustering Rank
    '''
    Cluster Rank
    '''
    matG = []
    nodeG = nx.nodes(G)  
    matG.append(nodeG)
    cl = nx.clustering(G).values()
    for i in range(len(cl)):
        cl[i] = pow(10,-cl[i])
    cl = list(cl)
    si = []
    for nodei in nx.nodes_iter(G):
        num = 0
        for vj in nx.all_neighbors(G,nodei):
            num = num + nx.degree(G,vj) + 1
        si.append(num)
    prod = list(map(lambda a,b:a*b, cl, si))
    matG.append(prod)
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  
    li =  result[0]                       
    li = li.astype(np.int32)  
    return  li


def IR(G):
    '''
    IRIE  算法
    '''
    r = [1]*len(G)  #初始化
    iteration = 20
    alpha =  0.7
    degreeG = nx.degree(G).values()  #节点的度
    while(iteration):
        for u in nx.nodes_iter(G):
            sumpr = 0
            for uneighbor in nx.all_neighbors(G,u):
                p = 1/degreeG[uneighbor] #传播概率
                sumpr = sumpr + p*r[uneighbor]
            r[u] = 1 + alpha * sumpr
        iteration = iteration - 1
    matG = []
    nodeG = nx.nodes(G)
    matG.append(nodeG)
    matG.append(r)
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  #按节点的度从大到小排
    vrank =  result[0]
    vrank = vrank.astype(np.int32)
    return vrank 

def AR_Degree(G):
    '''
    AR_dgree排序
    '''
    alpha = 1.3
    nodeid = range(1,len(G)+1)
    a = []
    for i in nodeid:    
        a.append(math.pow(i, -alpha))
    a_sum = sum(a)
    degreeG = nx.degree(G).values()
    AR = []
    for i  in range(len(G)):
        ARi = degreeG[i]*degreeG[i]*(-math.log(a[i]/a_sum))
        AR.append(ARi)
    matG = []
    nodeG = nx.nodes(G) 
    matG.append(nodeG)
    matG.append(AR)
    matG = np.array(matG)
    result =  matG.T[np.lexsort(-matG)].T  
    li =  result[0]        
    li = li.astype(np.int32)              
    return li  
    
    
def greedHeap(G,vrank):
    heap = []
    len_G = len(G)
    S = []  #集合S存放每次选择的节点
    cost = [1]*len_G
    nodeId = nx.nodes(G)  #节点编号
    is_update = np.zeros(len_G)  #已更新为1
    Cluster = [set()]*len_G #用以记录每一个集团的成员节点
    NodeCluster = [0]*len_G #记录每一个节点所属集团的ID
    newClusterId = 1
    maxcluster = 1
    sumcluster = 0
    for i in range(len_G):
        heapq.heappush(heap,(cost[i],i,vrank[i]))#建堆  
    #print '建堆完成'
    while heap:
        v =  heapq.heappop(heap)#从堆中删除元素，返回值是堆中最小的元素 
        nodev = v[2]                          #弹出被选择的节点
        if is_update[nodev]==1  :
            NodeCluster,Cluster,newClusterId,maxcluster = chooseV (nodev,G,NodeCluster,Cluster,newClusterId,maxcluster)
            sumcluster = sumcluster + maxcluster
            S.append(nodev)
            is_update = np.zeros(len_G)    #选完一个节点后更新状态为未更新
        else:
            cost[nodev] = updateCost(nodev,G,NodeCluster,cost,Cluster,maxcluster)   #更新节点的cost
            is_update[nodev] = 1              #标记为已更新
            heapq.heappush(heap,(cost[nodev],v[1],nodev)) #放入堆
    rnum = (sumcluster - len_G)/len_G/len_G
    #print (rnum)
    return S

def updateCost(nodev,G,NodeCluster,cost,Cluster,maxcluster):        #更新节点的cost
    ci  = set()                 #记录nodev可能连接的集团的ID集合
    for vj in nx.all_neighbors(G,nodev):
        if NodeCluster[vj] != 0: #vj已属于某个集团
            ci.add(NodeCluster[vj])
    sumc = 1
    for name in ci:
        sumc  = sumc + len(Cluster[name])
    if sumc < maxcluster:
        sumc  =  maxcluster 
    cost[nodev] = sumc
    return cost[nodev]

def chooseV(nodev,G,NodeCluster,Cluster,newClusterId,maxcluster):  #选择该节点:创建新集团 or 合并集团
    ci  = set()                 #记录vi可能连接的集团的ID集合
    for vj in nx.all_neighbors(G,nodev):
        if NodeCluster[vj] != 0: #vj已属于某个集团
            ci.add(NodeCluster[vj])
    if len(ci) == 0:
        NodeCluster[nodev] = newClusterId  #分配给新的集团
        Cluster[newClusterId] = set()   #新集团
        Cluster[newClusterId].add(nodev)   #将vi加入
        newClusterId = newClusterId + 1
    else:
        minci = min(ci)# 记录Ci中编号最小的集团ID，记为minci 
        NodeCluster[nodev] = minci #节点vi所属集团的ID minci
        Cluster[minci].add(nodev) #将节点vi加入到编号为minci的集团中
        #将Ci中所有集团的成员节点都合并，均放入minci中
        for name in  ci:
            if name != minci:
                for nod in Cluster[name]:
                    NodeCluster[nod] = minci 
                    Cluster[minci].add(nod)
        if len(Cluster[minci]) > maxcluster:
            maxcluster = len(Cluster[minci])
    return NodeCluster,Cluster,newClusterId,maxcluster

def generate_graph(graph_type):
    '''
    生成不同类型的网络,并将网络所有边写入csv
    '''
    if graph_type == 'ER':
        #ER 随机图
        start =time.clock()
        #G = nx.random_graphs.erdos_renyi_graph(20000, 0.002)
        G = nx.random_graphs.fast_gnp_random_graph(50000,0.0001)
        end = time.clock()
        print ('ER图生成时间: %s s'%(end-start))       
    if graph_type == 'WS':
        #WS 小世界网络
        start =time.clock()
        G = nx.random_graphs.newman_watts_strogatz_graph(300000, 3, 0.3,seed=None)
        end = time.clock()
        print ('WS图生成时间: %s s'%(end-start)) 
    if graph_type == 'BA':
        #BA 无标度网络
        start =time.clock()
        G = nx.random_graphs.barabasi_albert_graph(5000, 3)
        end = time.clock()
        print ('BA图生成时间: %s s'%(end-start))  
    if graph_type == 'rand':
        start =time.clock()
        G = nx.random_graphs.gnm_random_graph(200000, 400000, seed=None, directed=False)
        end = time.clock()
        print ('随机图生成时间: %s s'%(end-start))     
    if graph_type == 'CB':
        start =time.clock()
        G = nx.random_partition_graph([100000,100000,100000,100000],0.00002,0.000001, seed=None, directed=False)
        #G= nx.planted_partition_graph (10,10000,0.0003,0.0001)       
        end = time.clock()
        print ('社区图生成时间: %s s'%(end-start))   
    if graph_type == 'powlerlow':
        start =time.clock()
        G = nx.random_graphs.powerlaw_cluster_graph(2000, 4,0.5, seed=None)
        end = time.clock()
        print ('幂率图生成时间: %s s'%(end-start))  
        
    #对于不连通的网络，通过添加边是之连通
    is_c = nx.is_connected(G)
    if is_c == False:
        list_cc = sorted(nx.connected_components(G), key=len, reverse=True)
        for i in range(len(list_cc)):
            list_cc[i] = list(list_cc[i])

        for i in range(1,len(list_cc)):
            G.add_edge(list_cc[i-1][0],list_cc[i][0])
    print (nx.is_connected(G))
    
    csvFile = open(graph_type + '.csv','wb') 
    writer = csv.writer(csvFile)
    edges = nx.edges(G)
    for e in edges:
        writer.writerow([e[0],e[1]])
    csvFile.close()
    return  G
    
def fenbu(G):
    '''
    网络度分布
    '''
    degree =  nx.degree_histogram(G)          #返回图中所有节点的度分布序列
    #print (degree)
    x = range(len(degree))                             #生成x轴序列，从1到最大度
    y = [z for z in degree]  
    #plt.plot(x,y,'b.')
    plt.semilogy(x,y,'ks')
    #plt.loglog(x,y,'b.')           #在双对数坐标轴上绘制度分布曲线  
    plt.show()    

def stat_G(G):
    '''
    网络的统计信息
    节点数 n  边数 m 密度dens 最大度d_max  平均度 d_ave
    '''
    n = nx.number_of_nodes(G)
    #fout.write ('节点数%d\n' % n)
    print ('节点数%d\n' % n)
    m = nx.number_of_edges(G)
    #fout.write  ('边数%d\n' % m)    
    print ('边数%d\n' % m) 
    density =  nx.density(G)
    #fout.write ('密度%f\n' % density)
    degreeG = nx.degree(G).values()  
    d_max = max(degreeG)
    #fout.write ('最大度%d\n' % d_max)
    d_ave = sum(degreeG)/len(degreeG)
    #fout.write  ('平均度%f\n' % d_ave)
    print ('平均度%f\n' % d_ave)


        
    
def mainfun(Dataset_ind):
    '''
    输入不同的网络 Dataset
    输出网络的统计信息和度分布
    输出不同算法的排序结果  保存在 Dataset_算法.csv文件中，不同算法的运行时间
    输出不同算法的R曲线保存在 Dataset.esp; R 值 Dataset_R.esp; 运行时间 Dataset_time.esp
    '''
    #G = MDD()
    if Dataset in  ['ia-email', 'bright','citeseer','douban']:
        G = loadGraph(Dataset+'.csv',0)
    if Dataset in ['CB','WS']:
        G = generate_graph(Dataset)
    if Dataset == 'BA':
        G = loadGraph('real4.csv',7)   
    if Dataset == 'Random':
        G = loadGraph('real2.csv',5)
    
    is_c = nx.is_connected(G)
    fout.write("================网络是否是连通图--%s=====================\n" % (is_c))
    print (is_c)
   
    fout.write("================网络的统计信息=====================\n")
    stat_G(G)
    print ("================网络的度分布=======================")
    fenbu(G)
     
    
    fout.write("================运行所有的排序算法===============\n")
    #methods = ['D1-BGN','Ks','IRIE','PR','CR','AR','CC','BC']
    methods = ['D1-BGN','Ks','IRIE','PR','CR','AR']
    methods_color = ['r','g','b','c','y','#778899','m','k']
    methods_name = ['D1-BGN','K-shell','IRIE','PageRank','ClusterRank','AR-Degree','Closeness','Betweenness']
    all_t = []
    all_R = []  
    for i in range(len(methods)):
        method = methods[i]
        start =time.clock()
        if method == 'D1-BGN':
            initrank =  list(sortNode(G))
            initrank.reverse()#度从小到大的编号
            vrank = greedHeap(G,initrank)
            vrank.reverse()
        if method == 'Ks':
            vrank = list(sortNode1(G))
        if method == 'IRIE':
            vrank = list(IR(G))
        if method == 'PR':
            vrank = list(sortNode2(G))
        if method == 'CR':
            vrank = list(sortNode6(G))
        if method == 'AR':
            vrank = list(AR_Degree(G))
        if method == 'CC':
            vrank = list(sortNode9(G))
        if method == 'BC':
            vrank = list(sortNode8(G))            
        end = time.clock()
        t = end - start
        all_t.append(t)
        np.savetxt(Dataset+'_' + method + '.csv', vrank, delimiter = ',')  
        fout.write ('%s排序运行时间: %s s\n' % (method,(end-start))) 
        print  ('%s排序运行时间: %s s' % (method,(end-start)))          
        R,x,y = computeR1(vrank,G)
        all_R.append(R)
        fout.write  ('%s R值为%f\n'% (method,R))
        print ('%s R值为%f\n'% (method,R))
        plt.plot(x,y,color = methods_color[i],label= methods_name[i])


    plt.xlabel('p',fontsize=20)  
    plt.ylabel('$\sigma$',fontsize=20)  
    plt.title('R-curve',fontsize=20)
    plt.legend(loc='upper right',fontsize=12)
    plt.axis([0,1,0,1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig(Dataset+'.eps',bbox_inches='tight', format='eps', dpi=1000)
    
    
    
    
    plt.figure()
    np.savetxt(Dataset+'_R.csv', all_R, delimiter = ',')  
    x_pos = np.arange(len(methods))  
    plt.bar(x_pos,all_R,align = 'center',color="darkblue", width=0.5)
    #autolabel(rects)
    plt.xticks(x_pos,methods)  
    plt.ylabel('R',fontsize=20)  
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=18)
    maxr = max(all_R)
    maxy = maxr*1.1
    plt.ylim((0, maxy))
    plt.title('R-value',fontsize=20)
    foo_fig = plt.gcf() 
    foo_fig.savefig(Dataset+'_R.eps', bbox_inches='tight',format='eps',dpi=1000)
    plt.show() 
    
    
    plt.figure()
    np.savetxt(Dataset+'_Time.csv', all_t, delimiter = ',')  
    x_pos = np.arange(len(methods))  
    all_t[0] = all_t[0]*10
    plt.bar(x_pos,all_t,align = 'center',color="darkblue" ,width=0.5,  log = True)
    plt.xticks(x_pos,methods)  
    plt.ylabel('Time(Sec)',fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=18)
    plt.title('RunningTime',fontsize=20)
    foo_fig = plt.gcf()
    foo_fig.savefig(Dataset+'_Time.eps',bbox_inches='tight', format='eps', dpi=1000)
    plt.show() 

        

def BGN_vs_initBGN():
    '''
    输出不同的初始化方法的R值和运行时间
    '''
    #all_network = ['WS','CB',BA','Random']
    all_network = ['Random','BA']
    methods = ['Ks1-BGN','Ks2-BGN','AR1-BGN','AR2-BGN']
    #methods = ['Ks1-BGN','Ks2-BGN','AR1-BGN','AR2-BGN']
    for Dataset in all_network:
        if Dataset in  ['ia-email', 'bright','douban','citeseer']:
            G = loadGraph(Dataset+'.csv',0)
        if Dataset == 'WS':
            G = loadGraph(Dataset+'.csv', 8)
        if Dataset == 'CB':
            G = loadGraph(Dataset+'.csv', 9)
        if Dataset == 'BA':
            G = loadGraph('real4.csv',7)   
        if Dataset == 'Random':
            G = loadGraph('real2.csv',5)
        is_c = nx.is_connected(G)
        print (is_c)
        stat_G(G)
        for m in methods :
            start =time.clock()
            if m == 'BGN':
                initrank = nx.nodes(G)
                print (max(initrank))
                print (len(G))
            if m == 'D1-BGN':
                initrank =  list(sortNode(G))
                initrank.reverse()
            if m == 'D2-BGN':
                initrank =  list(sortNode(G))
            if m == 'P1-BGN':
                initrank =   list(sortNode2(G))
                initrank.reverse()
            if m == 'P2-BGN':
                initrank =   list(sortNode2(G))               
            if m == 'Ks1-BGN':
                initrank =  list(sortNode1(G))
                initrank.reverse()
            if m == 'Ks2-BGN':
                initrank =  list(sortNode1(G))
            if m == 'AR1-BGN':
                initrank =  list(AR_Degree(G))
                initrank.reverse()
            if m == 'AR2-BGN':
                initrank =  list(AR_Degree(G))
            vrank = greedHeap(G,initrank)
            vrank.reverse()  
            end = time.clock()
            print  ('%s-%s排序运行时间: %s s' % (m, Dataset, (end-start)))
            R,x,y = computeR1(vrank,G)
            print ('%s-%s R值为%f\n'% (m, Dataset,R))
            
            
if __name__ == "__main__":
    all_network = ['ia-email','bright','citeseer','douban','CB','WS','BA','Random']
    Dataset = all_network[7]  
    outfile = Dataset+'_result'
    with open(outfile,"w") as fout:
        fout.write("****************数据集--%s****************\n" % Dataset)
        mainfun(Dataset)
    
    #BGN_vs_initBGN()
    
    
    
















