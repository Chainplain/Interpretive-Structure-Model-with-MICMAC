import numpy as np
import networkx as nx
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import pylab
from sklearn.cluster import Birch
from matplotlib.pyplot import MultipleLocator
import sys

class Interpretive_structure_model():
    def __init__(self, model_name : str, adjoint_mat : np.mat):
        self.m_name = model_name

        # a bool mat
        self.ad_mat = adjoint_mat.T >0
        self.priority_sets = []
        self.reach_sets = []
        self.level_sets = []
        
        self.is_Computed = False
        [self.rows, self.cols] = self.ad_mat.shape


        self.data_write = open(self.m_name + "_wirte.txt",'w',encoding="utf-8")

        eye_logic = np.eye(self.rows)>0
        self.re_mat =  np.logical_or(self.ad_mat ,eye_logic)
        print("self.re_mat initialized as ", self.re_mat )
        self.re_mat_last = self.re_mat


        if (self.rows != self.cols):
            print("Error: please check rows?=cols.")
            sleep(5)
            exit(1)
        
    def Compute_re_mat(self):
        ind = 0
        while (ind < self.cols):
            self.re_mat = np.matmul(self.re_mat_last, self.re_mat_last)
            if ((self.re_mat == self.re_mat_last).all()):
                break
            ind = ind + 1
            self.re_mat_last = self.re_mat
        print("Re_mat is computed for ", ind + 1, " times.")
        self.is_Computed = True
    
    def Get_re_mat(self):
        if self.is_Computed:
            return self.re_mat
        else:
            self.Compute_re_mat()
            return self.re_mat
        
    def Compute_reach_sets(self):
        print('Reach sets:',file=self.data_write)
        for i in range(self.rows):
            reach_set_temp = set()
            print('S'+str(i+1)+'\'s reach sets: ',end="",file=self.data_write) 
            for j in range(self.cols):
                if self.re_mat[i,j]:
                    reach_set_temp.add(j)
                    print('S'+str(j+1)+', ',end="",file=self.data_write)    
            self.reach_sets.append(reach_set_temp)
            print('',file=self.data_write)
        print('',file=self.data_write)
        print('reach_sets',self.reach_sets)
    
    def Compute_priori_sets(self):
        print('Priority sets:',file=self.data_write)
        for j in range(self.cols):
            priori_set_temp = set()
            print('S'+str(j+1)+'\'s priority sets: ',end="",file=self.data_write) 
            for i in range(self.rows):
                if self.re_mat[i,j]:
                    priori_set_temp.add(i)
                    print('S'+str(i+1)+', ',end="",file=self.data_write)
            self.priority_sets.append(priori_set_temp)
            print('',file=self.data_write)
        print('',file=self.data_write)
        print('priority_sets',self.priority_sets)

    def Compute_level_sets(self):
        is_leveled = self.cols * [False]
        temp_priority_sets = self. priority_sets
        temp_reach_sets    = self. reach_sets
        levelindex = 1
        print('Levels:',file=self.data_write)
        while(not all(is_leveled)):
            print('Level '+str(levelindex)+': ',end="",file=self.data_write)
            current_level_set = set()
            for i in range(self.rows):
                if (not is_leveled[i]):
                    if (temp_priority_sets[i] == temp_priority_sets[i] &  temp_reach_sets[i]):
                        print('S'+str(i+1)+', ',end="",file=self.data_write)
                        current_level_set. add(i)
                        is_leveled[i] = True
            for i in range(self.rows):
                for va in current_level_set:
                    temp_priority_sets[i].discard(va)
                    temp_reach_sets[i].discard(va)
            self.level_sets.append(current_level_set)
            levelindex += 1
            print('',file=self.data_write)
        print('',file=self.data_write)
        print('level_sets',self.level_sets)

    def Compute_MICMAC(self):
        col_sums = (np.sum(self.re_mat,axis = 1).flatten().tolist())
        raw_sums = (np.sum(self.re_mat,axis = 0).flatten().tolist())
        print('col_sums',col_sums)
        print('raw_sums',raw_sums)


        raw_MICMAC_points = [col_sums, raw_sums]
        print('raw_MICMAC_points',raw_MICMAC_points)
        self. MICMAC_points =  [list(x) for x in zip(*raw_MICMAC_points)]
        print('MICMAC_points', self. MICMAC_points)

        print('MICMAC points:',file=self.data_write)
        index_i = 1
        for micmac in self. MICMAC_points:
            print('S'+str(index_i)+': '+ str(micmac),file=self.data_write)
            index_i += 1
        print('',file=self.data_write)

        number_of_kernel = input('Please input the cluster kernel numbers:')
        brc = Birch(n_clusters=int(number_of_kernel))
        brc.fit(self. MICMAC_points)
        cat = brc.predict(self. MICMAC_points)
        print('Birch_Predict', cat)

        print('Cluster:',file=self.data_write)

        for i in range(int(number_of_kernel)):
            print('Category '+str(i+1)+': ',end="",file=self.data_write)
            index_i = 1
            for cate in cat:
                if cate == i:
                    print('S'+str(index_i)+', ',end="",file=self.data_write)
                index_i += 1
            print('',file=self.data_write)


        plt.figure()
        plt.scatter(np.array(col_sums), np.array(raw_sums), marker='o', c=cat, cmap='Accent')
        
        x_major_locator=MultipleLocator(2)
        
        plt.xlim(0,self.rows+1)
        plt.ylim(0,self.cols+1)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(x_major_locator)

        plt.grid(ls='--')
        plt.savefig(self.m_name + '_MICMAC.png')   
        plt.show()        

    def Img_show(self):
        plt.clf()
        G = nx.DiGraph()
        for i in range(0, self.rows):
                G.add_node(i, desc='S'+str(i+1))
        
        no_self_re_mat = np.logical_and(self.re_mat, np.logical_not( np.eye(self.rows)))
        for p in range(0,self.rows):
            for q in range(0,self.rows):
                if(no_self_re_mat[p,q]):
                    G.add_edges_from([(p, q)],weight='1')
        edge_labels=dict([((u,v,),d['weight'])
                        for u,v,d in G.edges(data=True)])
        Honeydew2 = [224/255, 238/255, 224/255]
        Honeydew4 = [131/255, 139/255, 131/255]#131 139 131
        DarkTurquoise = [0/255, 206/255, 209/255]
        # pos=nx.shell_layout(G)

    
        node_labels = nx.get_node_attributes(G, 'desc')

        print('level_sets_size:',len(self.level_sets))
        print('level_sets:',self.level_sets)

        pos_new = dict()
        ax=plt.gca()

        vertical_pos = np.linspace(-1,1,len(self.level_sets))

        iter_i = 0
        for this_line_set in self.level_sets:
            iter_j = 0
            horizontal_pos = np.linspace(-1,1,len(this_line_set))
            if  (len(this_line_set)==1):
                horizontal_pos[0] = 0      
            rect = patches.Rectangle((-1.3,1 - vertical_pos[iter_i] - 1/len(vertical_pos)), 2.6, 2/len(vertical_pos),\
                                         linestyle = 'dotted',edgecolor = DarkTurquoise,facecolor = 'none', linewidth=2)
            ax.add_patch(rect)
            for this_node in this_line_set:
                pos_new[this_node] = np.array([ horizontal_pos[iter_j],  1 - vertical_pos[iter_i]])
                iter_j = iter_j + 1
            iter_i = iter_i + 1
        print('pos_new:',pos_new)

        nx.draw_networkx_labels(G, pos_new, labels=node_labels)
        # nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

        #  {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', 
        #   '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right', 
        #   '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', 
        #   '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 
        #   'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', 
        #   '|': 'vline', '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 
        #   1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright', 
        #   6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 9: 'caretrightbase', 
        #   10: 'caretupbase', 11: 'caretdownbase', 'None': 'nothing',
        #     None: 'nothing', ' ': 'nothing', '': 'nothing'}
        

        nx.draw(G,pos_new, node_size=1000,node_color= Honeydew2,node_shape='s', edge_color=Honeydew4,edge_cmap=plt.cm.Reds)
        plt.title('Directed Graph', fontsize=12)

        

        plt.savefig(self.m_name + '_Directed_Graph.png') 
        plt.show() 


