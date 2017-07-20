#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
import time
import numpy as np
import pylab as plt

__author__ = 'Guo Li'
__email__ = 'leeguoo@hotmail.com'
__date__ = 'Feb 22, 2017'

#Boltzmann constant (eV/K)
kB = 8.6173303E-5

class Square(object):
    def __init__(self, len=1,x_begin=0,y_begin=0):
        """
           格子,length正方形的边长，
          x_begin,y_begin 表示 这个正方式的起始坐标，有了超始坐标
         与边长， 就可确定这个正方形覆盖的位置
        """
        self.len=len
        self.x_begin=x_begin
        self.y_begin=y_begin
    def is_covered(x,y):         # 判断x y 这个点 是否在Square 范围内
        for pos in self.get_pos_list():
            if pos[0]==x and pos[1]==y:
                return True
        return False
    
    def get_left_bottom(self):
        return self.x_begin,self.y_begin
    def get_right_bottom(self):
        return self.x_begin+self.len-1,self.y_begin
    def get_top_left(self):
        return self.x_begin,self.y_begin+self.len-1
    def get_top_right(self):
        return self.x_begin+self.len+1,self.y_begin+self.len-1
    def get_bottom_pos_list(self):
        posList=[]
        for x in range(self.len):
            posList.append([self.x_begin+x,self.y_begin])
        return posList
    def get_left_pos_list(self):
        posList=[]
        for dy in range(self.len):
            posList.append([self.x_begin,self.y_begin+dy])
        return posList
    def get_right_pos_list(self):
        posList=[]
        for dy in range(self.len):
            posList.append([self.x_begin+self.len-1,self.y_begin+dy])
        return posList
    def get_top_pos_list(self):
        posList=[]
        for dx in range(self.len):
            posList.append([self.x_begin+dx,self.y_begin+self.len-1])
        return posList
    # def set_self.x_begin(self,x):
    #     self.x_begin=x
    # def set_y_begin(self,y):
    #     self.y_begin=y
    def get_pos_list(self):
        "返回4个位置"
        posList=[]
        for x in range(self.len):
            for y in range(self.len):
                posList.append([self.x_begin+x,self.y_begin+y])
        return posList



class BinaryMC(object):
    """
    A class to simulation the formation of binary thin film on a substrate
    from a drop of particles.
    """
    def __init__(self, param, ngrid=100, nblocks=150, BAratio=0.3):
        """
        Initialize the simulation.
        """
        self.param = param
        self.ngrid = ngrid      # 整个区域边长+1 (因0based)
        self.nblocks = nblocks  # 需要红蓝格子的总数
        self.ratio = BAratio
        self.get_start()


    def get_start(self):
        """
        Constuct a square drop of particles.
        """
        nblocks = self.nblocks
        ngrid = self.ngrid
        ratio = self.ratio
        self.ground=np.zeros(shape=(ngrid/2,ngrid))
        # nsqrt = int(np.ceil(np.sqrt(nblocks)))
        # nmod = nblocks%nsqrt
        # print("nblocks,nsqrt,nmod,nblocks/nsqrt",nblocks,nsqrt,nmod,nblocks/nsqrt)
        self.xmiddle=ngrid/2
        self.ymiddle=0  #初始时， 尽量让各个格子围着（xmiddle,ymiddle） 这一点，寻找离这个点最近的点来放置下一个格子，


        self.blocs = []
        for i in range(nblocks):
            if random.uniform(0,1) > ratio:
                len=1
                x_begin,y_begin=self.get_min_distance(len)
                square=Square(len,x_begin,y_begin)
                self.set_ground_by_pos(square,1)
                self.blocs.append(square)
            else:
                len=2
                x_begin,y_begin=self.get_min_distance(len)
                square=Square(len,x_begin,y_begin)
                self.set_ground_by_pos(square,-1)
                self.blocs.append(square)

    def get_min_distance(self,len): # len 为正方形的边长，
        # (self.xmiddle,self.ymiddle)是一个点
        # 这个函数为于寻找 ground中距离(self.xmiddle,self.ymiddle)最近，且value值为0的位置，即此位置没有放置任何格子
        # 且此位置足以容纳边长为len的正方形
        # 以便于初始化的时候围绕着这个点排兵布阵
        is_started=False
        x_nearest=0
        y_nearest=0
        for y, arr1 in enumerate(self.ground):
            for x, value in enumerate(arr1):
                if value!=0:
                    continue

                is_enouth,x_begin,y_begin=self.is_enough_space_nearby(x,y,len)
                if not is_enouth:
                    continue


                if is_started==False:
                    is_started=True
                    x_nearest=x_begin
                    y_nearest=y_begin
                else:
                    if self.get_distance(x_begin,y_begin,self.xmiddle,self.ymiddle)<self.get_distance(x_nearest,y_nearest,self.xmiddle,self.ymiddle):
                        x_nearest=x_begin
                        y_nearest=y_begin
                    elif self.get_distance(x_begin,y_begin,self.xmiddle,self.ymiddle)==self.get_distance(x_nearest,y_nearest,self.xmiddle,self.ymiddle):
                        # 如果两个点距离相等，则选y值小的那一个，即，尽量让水滴 底粗头细,优先在底部放置格子
                        if y<y_nearest:
                            x_nearest=x_begin
                            y_nearest=y_begin
        return x_nearest,y_nearest
  
    def get_ground_y_len(self):
        return len(self.ground)
    def get_ground_x_len(self):
        return len(self.ground[0])


    def is_enough_space_nearby(self,x,y,len): # len =1 或2
        # 已知 (x,y) 为空格子，判断(x,y)附近是否为空，以足够容纳边长为len的正方形
        # 并返回这个足够容纳这个正方形的起点坐标
        if len==1:              # 边长为1时，直接返回这个点即可，因为忆知(x,y )为空格子
            return True,x,y

        # 以下为len=2的情形 ,如下图 判断 a bc 的位置是否同时为空格子
        if x<=self.xmiddle: # 如果此点位于基点的左侧 (主要为了让选中的点离基点越近越好)
            is_enough,x_begin,y_begin=self.is_enough_space_nearby_left_bottom(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin
            is_enough,x_begin,y_begin=self.is_enough_space_nearby_top_left(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin
            is_enough,x_begin,y_begin=self.is_enough_space_nearby_right_bottom(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin
            is_enough,x_begin,y_begin=self.is_enough_space_nearby_right_top(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin
        else:# 如果此点位于基点的左侧
            is_enough,x_begin,y_begin=self.is_enough_space_nearby_right_bottom(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin
            is_enough,x_begin,y_begin=self.is_enough_space_nearby_right_top(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin

            is_enough,x_begin,y_begin=self.is_enough_space_nearby_left_bottom(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin
            is_enough,x_begin,y_begin=self.is_enough_space_nearby_top_left(x,y)
            if  is_enough: return  is_enough,x_begin,y_begin
        return False,0,0

    def get_distance(self,x1,y1,x2,y2):#计算两点之间的距离的平方 (x1,y1),(x2,y2 ) 则 （x2-x1）^2+(y2-y1)^2
        return (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    def is_enough_space_nearby_left_bottom(self,x,y): #
        # 以下为len=2的情形 ,如下图 判断 a bc 的位置是否同时为空格子
        # 并返回起点的坐标(4个点中坐标最小的那个)
        # |b    |c |
        # |(x,y)| a|
        b_is_empty=(y!=self.get_ground_y_len()-1 and self.ground[y+1][x]==0)
        c_is_empty=(x!=self.get_ground_x_len()-1 and y!=self.get_ground_y_len()-1 and self.ground[y+1][x+1]==0)
        a_is_empty=(x!=self.get_ground_x_len()-1 and self.ground[y][x+1]==0)
        if a_is_empty and b_is_empty and c_is_empty:
            return True,x,y       # (x,y)点的位置
        return False,0,0
    def is_enough_space_nearby_top_left(self,x,y): #
        # 以下为len=2的情形 ,如下图 判断 a bc 的位置是否同时为空格子
        # 并返回起点的坐标(4个点中坐标最小的那个)
        # |(x,y)    |c |
        # |b        | a|
        b_is_empty=(y!=0 and self.ground[y-1][x]==0)
        c_is_empty=(x!=self.get_ground_x_len()-1 and self.ground[y][x+1]==0)
        a_is_empty=(y!=0  and x!=self.get_ground_x_len()-1  and self.ground[y-1][x+1]==0)
        if a_is_empty and b_is_empty and c_is_empty:
            return True,x,y-1       # b点的位置
        return False,0,0

    def is_enough_space_nearby_right_top(self,x,y): #
        # 以下为len=2的情形 ,如下图 判断 a bc 的位置是否同时为空格子
        # 并返回起点的坐标(4个点中坐标最小的那个)
        # |c        |(x,y) |
        # |b        | a    |
        a_is_empty=(y!=0 and self.ground[y-1][x]==0)
        c_is_empty=(x!=0 and self.ground[y][x-1]==0)
        b_is_empty=(y!=0 and x!=0 and self.ground[y-1][x-1]==0)
        if a_is_empty and b_is_empty and c_is_empty:
            return True,x-1,y-1       # b点的位置
        return False,0,0
    def is_enough_space_nearby_right_bottom(self,x,y): #
        # 以下为len=2的情形 ,如下图 判断 a bc 的位置是否同时为空格子
        # 并返回起点的坐标(4个点中坐标最小的那个)
        # |c        |a         |
        # |b        | (x,y)    |
        a_is_empty=(y!=self.get_ground_y_len()-1 and self.ground[y+1][x]==0)
        b_is_empty=(x!=0 and self.ground[y][x-1]==0)
        c_is_empty=(x!=0 and y!=self.get_ground_y_len()-1 and self.ground[y+1][x-1]==0)
        if a_is_empty and b_is_empty and c_is_empty:
            return True,x-1,y       # b点的位置
        return False,0,0


    def get_square_by_pos(self,x,y):
         # 判断xy 对应的坐标是否有对应的square, 否则话return None
        #  x,y 可以是大格子中4个点的任意一个点
            for i, square in enumerate(self.blocs):
                if square.is_covered(x,y):
                    return square
            return None

    def set_ground_by_pos(self,square,value):
        for dy in range(square.len):
            for dx in range(square.len):
                self.ground[square.y_begin+dy][square.x_begin+dx]=value


    def is_empty(self,square): # 检测 ground 是否允许放置square
        for pos in square.get_pos_list():
            x=pos[0]
            y=pos[1]
            if self.ground[y][x]!=0:
                return False
        return True

    def move_down_step(self,nb):        # 向下移动1部，
        square = self.blocs[nb]                      #get the location of the block
        old_value=self.ground[square.y_begin][square.x_begin]
        if square.y_begin==0:                        # 到底部了
            return False

        self.set_ground_by_pos(square,0) # 将当前位置占据的ground 置空，以方便检测移动之后的新位置是否有足够的空位（新旧位置有可能有交叉，所以提前把旧位置致空）
        x_new=square.x_begin
        y_new=square.y_begin-1
        square_new=Square(square.len,x_new,y_new)
        if self.is_empty(square_new):
            self.set_ground_by_pos(square_new,old_value)
            self.blocs[nb] =square_new
            return True     # move succ
        else:
            self.set_ground_by_pos(square,old_value) # reset to init value
            return False
    def move_down(self,nb):        # 向下移动， 如果下方有空间，一直向下移动，直到到底
        while self.move_down_step(nb):
            pass



    def move_left(self,nb):        # 如果没有障碍物 向左移动1步， 移到头后从最右侧出现
        if not self.move_left_step(nb): # 移动失败直接返回
            return
        #如果移动了一步，则检测下方有没有空间， 有就掉下去吧
        self.move_down(nb)
    def move_left_step(self,nb):        # 如果没有障碍物 向左移动1步， 移到头后从最右侧出现
        square = self.blocs[nb]                      #get the location of the block
        old_value=self.ground[square.y_begin][square.x_begin]

        y_new=square.y_begin
        x_new=square.x_begin-1
        if square.x_begin==0:                        # 当在最左侧时
            x_new=self.get_ground_x_len()-square.len
        self.set_ground_by_pos(square,0) # 将当前位置占据的ground 置空，以方便检测移动之后的新位置是否有足够的空位（新旧位置有可能有交叉，所以提前把旧位置致空）
        square_new=Square(square.len,x_new,y_new)
        if self.is_empty(square_new):
            self.set_ground_by_pos(square_new,old_value)
            self.blocs[nb] =square_new
            return True     # move succ
        else:
            self.set_ground_by_pos(square,old_value) # reset to init value
            return False

    def move_right(self,nb):        # 如果没有障碍物 向右移动1步， 移到头后从最左侧出现
        if not self.move_right_step(nb): # 移动失败直接返回
            return
        #如果移动了一步，则检测下方有没有空间， 有就掉下去吧
        self.move_down(nb)
        
    def move_right_step(self,nb):        # 如果没有障碍物 向右移动1步， 移到头后从最左侧出现
        square = self.blocs[nb]                      #get the location of the block
        old_value=self.ground[square.y_begin][square.x_begin]

        y_new=square.y_begin
        x_new=square.x_begin+1
        x_right_bottom,y_right_bottom=square.get_right_bottom()
        if x_right_bottom==self.get_ground_x_len()-1:                        # 当在最右侧时
            x_new=0
        self.set_ground_by_pos(square,0) # 将当前位置占据的ground 置空，以方便检测移动之后的新位置是否有足够的空位（新旧位置有可能有交叉，所以提前把旧位置致空）
        square_new=Square(square.len,x_new,y_new)
        if self.is_empty(square_new):
            self.set_ground_by_pos(square_new,old_value)
            self.blocs[nb] =square_new
            return True     # move succ
        else:
            self.set_ground_by_pos(square,old_value) # reset to init value
            return False





    def move(self):
        nb = random.randint(0,self.nblocks-1)      #select a block
        square = self.blocs[nb]                      #get the location of the block
        x=square.x_begin
        y=square.y_begin
        # y, x = self.blocs[nb]                      #get the location of the block
        dx = random.choice([-1,0,0,1])             #select a direction along x
        if dx == 0:                                #select a direction along y
            dy = random.choice([-1,1])
        else:
            dy = 0

        x1 = self.periodic(x,dx) # 这个x1,y1 跟真正移动到的位置还是有差别的，这个计算只是为了计算Probability
        y1 = y+dy
        if random.uniform(0,1)<=self.Probability(x,y,x1,y1,dx):
            if dx==-1:
                self.move_left(nb)
            elif dx==1:
                self.move_right(nb)
            elif dy==-1:
                self.move_down(nb)
            else:                   # move up ,似乎无意义
                pass



        # #diffusion
        # if self.ground[y1][x1]==0 and dx!=0 and self.ground[y+1][x]==0:
        #     #find the highest y at that x
        #     while True:
        #         if y1==0 or self.ground[y1-1][x1]!=0:
        #             break
        #         else:
        #             y1 -= 1
        #     #print self.Probability(x,y,x1,y1,dx), "diffusion"
        #     if random.uniform(0,1)<=self.Probability(x,y,x1,y1,dx):
        #         self.ground[y1][x1]=self.ground[y][x] #move the block
        #         self.ground[y][x]=0
        #         self.blocs[nb] = [y1,x1]        #update the block list
        # #exchange
        # elif self.ground[y1][x1]!=0 and y1>=0 and self.ground[y1][x1]!=self.ground[y][x]:
        #     for i, val in enumerate(self.blocs):
        #         if val == [y1,x1]:
        #             nb1 = i
        #             break
        #     #print self.Probability(x,y,x1,y1,dx), "exchange"
        #     if random.uniform(0,1)<=self.Probability(x,y,x1,y1,dx):
        #         #if y!=y1:
        #         #    print self.ground[y][x],self.ground[y1][x1]
        #         temp = self.ground[y1][x1]
        #         self.ground[y1][x1]=self.ground[y][x] #move the block
        #         self.ground[y][x]=temp
        #         self.blocs[nb] = [y1,x1]        #update the block list
        #         self.blocs[nb1] = [y,x]


    def periodic(self,x,dx):
        if x+dx < 0:
            return x+dx+self.ngrid
        elif x+dx > self.ngrid-1:
            return x+dx-self.ngrid
        else:
            return x+dx



    def bindingE(self,x,y):
        V = 0
        if self.ground[y][x]==1: #Type-A block
            #up
            if self.ground[y+1][x]==1:
                V += self.param['EAA']
            elif self.ground[y+1][x]==-1:
                V += self.param['EAB']
            #down
            if y==0:
                V += self.param['EAS']
            elif self.ground[y-1][x]==1:
                V += self.param['EAA']
            elif self.ground[y-1][x]==-1:
                V += self.param['EAB']
            #left
            if self.ground[y][self.periodic(x,-1)]==1:
                V += self.param['EAA']
            elif self.ground[y][self.periodic(x,-1)]==-1:
                V += self.param['EAB']
            #right
            if self.ground[y][self.periodic(x,1)]==1:
                V += self.param['EAA']
            elif self.ground[y][self.periodic(x,1)]==-1:
                V += self.param['EAB']
        elif self.ground[y][x]==-1: #Type-B block
            #up
            if self.ground[y+1][x]==1:
                V += self.param['EAB']
            elif self.ground[y+1][x]==-1:
                V += self.param['EBB']
            #down
            if y==0:
                V += self.param['EBS']
            elif self.ground[y-1][x]==1:
                V += self.param['EAB']
            elif self.ground[y-1][x]==-1:
                V += self.param['EBB']
            #left
            if self.ground[y][self.periodic(x,-1)]==1:
                V += self.param['EAB']
            elif self.ground[y][self.periodic(x,-1)]==-1:
                V += self.param['EBB']
            #right
            if self.ground[y][self.periodic(x,1)]==1:
                V += self.param['EAB']
            elif self.ground[y][self.periodic(x,1)]==-1:
                V += self.param['EBB']
        else:
            print "Error"
        return V


    def Probability(self,x,y,x1,y1,dx):
        """
        Calculate the probability of the movement at room temperature.

        Args:
            x, y (int): the current location of the block
            x1, y1 (int): the new location

        Returns:
            Relative probability (float): at room temperature (300K), and
            the probability of the fatest movement is set as 1.
        """
        V = 0
        #diffusion
        if self.ground[y1][x1]==0:
            if self.ground[y][x]==1: # A-type block
                #diffusion barrier
                if y==0:
                    V += self.param['EAS']*self.param['AlphaA']
                elif self.ground[y-1][x]==1:
                    V += self.param['EAA']*self.param['AlphaA']
                elif self.ground[y-1][x]==-1:
                    V += self.param['EAB']*self.param['AlphaA']
                #detaching barrier
                if self.ground[y][self.periodic(x,-dx)]==1:
                    V += self.param['EAA']
                elif self.ground[y][self.periodic(x,-dx)]==-1:
                    V += self.param['EAB']
            elif self.ground[y][x]==-1: #B-type block
                #diffusion barrier
                if y==0:
                    V += self.param['EBS']*self.param['AlphaB']
                elif self.ground[y-1][x]==1:
                    V += self.param['EAB']*self.param['AlphaB']
                elif self.ground[y-1][x]==-1:
                    V += self.param['EBB']*self.param['AlphaB']
                #detaching barrier
                if self.ground[y][self.periodic(x,-dx)]==1:
                    V += self.param['EAB']
                elif self.ground[y][self.periodic(x,-dx)]==-1:
                    V += self.param['EBB']
        else: #exchange
            V = self.bindingE(x,y)
            V = V+self.bindingE(x1,y1)
            V = V*self.param['AlphaC']
        #find the lowest energy barrier
        Vref = min(param['AlphaA'],param['AlphaB'])*min(param['EAA'],param['EBB'],param['EAB'],param['EAS'],param['EBS'])
        #print V, np.exp((Vref-V)/300/kB)
        return np.exp((Vref-V)/300/kB)

    def plot(self,name='tmp.png'):
        plt.imshow(MC.ground,cmap='bwr',origin='uper')
        plt.xticks([ 1])
        plt.yticks([])
        plt.savefig(name)

    def simulate(self,nsteps=10000000,nfigs=20):
        self.plot('0.png')
        nspan = nsteps/nfigs
        for i in range(nsteps):
            self.move()
            if (i+1)%nspan==0:
                self.plot(str((i+1)/nspan)+'.png')

param = {"EAA":0.12,   #A-A interaction
         "EBB":0.04,    #B-B interaction
         "EAB":0.05,    #A-B interaction
         "EAS":0.2,    #A-surface interaction
         "EBS":0.2,    #B-surface interaction
         "AlphaA":1.0,   #diffusion coefficient
         "AlphaB":1.0,
         "AlphaC":0.2    #exchange coefficient
        }
start = time.time()
MC = BinaryMC(param=param,ngrid=50,nblocks=70,BAratio=0.3)
MC.simulate(nsteps=8000000,nfigs=10)
print time.time()-start
