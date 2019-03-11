# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 14:33:51 2018

@author: Li XG
"""
import numpy as np
import cv2
import sys
import math
# from pyclustering.nnet.som import som, type_conn, type_init, som_parameters

def L2ofPoint(pt1, pt2):
    return np.sqrt(np.power((pt1[0]-pt2[0]),2)+np.power((pt1[1]-pt2[1]),2))

"""
计算连通域
opencv::connectedComponentsWtihStat
count：标记总数（0为背景，即为0.0的区域）
labels: 标记矩阵
# stats N*5的矩阵，行对应每个label，五列分别为[(leftmost ), (topmost ), width, height, area],左上角坐标
# centroids 是每个域的质心坐标
# connectivity 4或8 临近像素： 周围4像素或8像素
"""
def findConnectedComponentsWithStat(data, min_area=sys.maxsize, min_energy=sys.maxsize):
    ret = []
    
    d = data.copy()
    d[d>0.0]=1.0
    d = d.reshape(d.shape[0], d.shape[1], 1)
    d = d.astype(np.int8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(d,  connectivity=8)                
    for label in range(1, count):
        mask = np.where(labels==label, 1, 0)
        data1 = data*mask
        energy = np.mean(data1)
        stat = stats[label]
        left = stat[0]
        top = stat[1]
        width = stat[2]
        height = stat[3]
        area = stat[4]
        connected_data = data1[top:top+height, left:left+width]
        if(min_area == sys.maxsize and min_energy == sys.maxsize):
            ret.append({'top':top, 
                        'left':left, 
                        'width':width, 
                        'height':height, 
                        'area':area, 
                        'energy':energy,
                        'data':connected_data})
        elif(area>min_area and energy>=min_energy):
            ret.append({'top':top, 
                        'left':left, 
                        'width':width, 
                        'height':height, 
                        'area':area, 
                        'energy':energy,
                        'data':connected_data})
    return ret
"""
计算质心
"""
def getMassCenter(data):
    """
    data: n*m array
    """
    mc0 = np.sum(data, axis=0)
    mm0 = 0.0
    for i in range(mc0.shape[0]):
        mm0 = mm0+mc0[i]*i
        
    mc1 = np.sum(data, axis=1)
    mm1 = 0.0
    for i in range(mc1.shape[0]):
        mm1 = mm1+mc1[i]*i

    mc = np.sum(data)
    return int(mm1/mc), int(mm0/mc) #r, c
"""
获得连通区域的连续质心
"""
def getMassCenterSeq(connectedComponent, delta):
    data = connectedComponent['data']
    width = connectedComponent['width']
    height = connectedComponent['height']
    ratio = width/height
    if(ratio > 1.0):
        data = data.transpose()
    #计算质心
    rows = data.shape[0]
    cols = data.shape[1]
    mass_center = []
    for r in range(0, rows, delta):
        if r+delta>rows : 
            r_ = rows 
        else: 
            r_ = r+delta
        area = data[r:r_, 0:cols]
        c_r, c_c = getMassCenter(area)
        mass_center.append([r+c_r, c_c])
    
    mass_center = np.asarray(mass_center)
    if(ratio > 1.0):
        mass_center[:, [0,1]] = mass_center[:, [1,0]]#因为转置，所以互换c_r和c_c 
    return mass_center

def computeSlope(mass_center, mc_delta=1):
    slopes = []
    for i in range(mass_center.shape[0]-mc_delta):
        pt1 = mass_center[i]
        pt2 = mass_center[i+mc_delta]
        if(pt1[0]-pt2[0])==0 :
            slope = math.atan(np.inf)
        else:
            slope = math.atan((pt1[1]-pt2[1])/(pt1[0]-pt2[0]))
            
        slopes.append(slope)
    return np.asarray(slopes)

"""
计算斜率:区域间隔为delta，质心间隔mc_delta
connectedComponent：[top, left, width, height, area, energy, data]
"""
def computeSlopeofArea(connectedComponent, delta=1, mc_delta=1):
    mass_center = getMassCenterSeq(connectedComponent, delta)
    slopes = computeSlope(mass_center, mc_delta)
    return [slopes, mass_center]

def computeCurvature(mass_center, mc_delta=1):
    curvatures = []
    for i in range(mass_center.shape[0]-2*mc_delta):
        pt1 = mass_center[i]
        pt2 = mass_center[i+mc_delta]
        pt3 = mass_center[i+mc_delta*2]
        
        d12 = L2ofPoint(pt1, pt2)
        d23 = L2ofPoint(pt2, pt3)
        d13 = L2ofPoint(pt1, pt3)
        
        cur = 4*np.sqrt(np.power((d12+d23), 2)- d13*d13) / np.power((d12+d23), 2)
        curvatures.append(cur)
    return np.asarray(curvatures)

"""
计算曲率:区域间隔为delta，质心间隔mc_delta
connectedComponent：[top, left, width, height, area, energy, data]
"""
def computeCurvatureofArea(connectedComponent, delta=1, mc_delta=1):
    mass_center = getMassCenterSeq(connectedComponent, delta)
    curvatures = computeCurvature(mass_center, mc_delta)
    return [curvatures, mass_center]

def computeStatofArea(connectedComponent, delta=1, mc_delta=1):
    mass_center = getMassCenterSeq(connectedComponent, delta)
    slopes = computeSlope(mass_center, mc_delta)
    curvatures = computeCurvature(mass_center, mc_delta)
    if(slopes.size>=1 and curvatures.size>=1):
        s_mean, s_var = slopes.mean(), slopes.var()
        c_mean, c_var = curvatures.mean(), curvatures.var()
        return [s_mean, s_var, c_mean, c_var]
    else:
        return None
"""
自组织映射
data：list of data element
"""
def SOM(data, rows=2, cols=2, epochs=10):
    parameters = som_parameters()
    structure = type_conn.grid_eight  # each neuron has max. four neighbors.
    network = som(rows, cols, structure, parameters)
    # train network
    network.train(data, epochs)
    return network

"""
拼凑矩形。将多个矩形拼成一个矩形，用于结果显示
"""
def mergeRectangle(rects, cols=16):
    max_width = 0
    max_height = 0
    for i in range(len(rects)):
        w = rects[i]['width']
        h = rects[i]['height']
        if(w>max_width):
            max_width = w
        if(h>max_height):
            max_height = h
    
    resizeRect = []
    for i in range(len(rects)):
        w = rects[i]['width']
        h = rects[i]['height']
        data = rects[i]['data']
        rect = np.zeros(shape=(max_height, max_width))
        rect[0:h, 0:w] = data
        resizeRect.append(rect)
    
    rows = int(np.ceil(len(resizeRect)/cols))
    #print('height:%d width:%d'%(max_height, max_width))
    mergeRect = np.zeros(shape=(rows*max_height, cols*max_width))
    for i in range(len(resizeRect)):
        r = i//cols
        c = i%cols
        mergeRect[r*max_height: (r+1)*max_height, c*max_width:(c+1)*max_width] = resizeRect[i]
    return mergeRect
    
"""
矩形相交比例
"""        
def rectOverlap(rc1, rc2):
    #p1为相交位置的左上角坐标，p2为相交位置的右下角坐标
    p1_x = max([rc1['left'], rc2['left']])
    p1_y = max([rc1['top'], rc2['top']])
    p2_x = min([[rc1['left'] + rc1['width']], [rc2['left'] + rc2['width']]])
    p2_y = min([[rc1['top'] + rc1['height']], [rc2['top'] + rc2['height']]])
 
    AJoin = 0
    if(p2_x > p1_x and p2_y > p1_y): #判断是否相交
        AJoin = (p2_x - p1_x)*(p2_y - p1_y)#求出相交面积

    A1 = rc1['width'] * rc1['height']
    A2 = rc2['width'] * rc2['height']
    AUnion = (A1 + A2 - AJoin)#两者组合的面积
    if (AUnion > 0):
        return (AJoin / AUnion)#相交面积与组合面积的比例
    else:
        return 0
"""
计算矩形区域中像素重叠率。重叠像素/最小面积
"""
def PixelOverlap(rc1, rc2):
    #p1为合并后的左上角坐标，p2为合并后的右下角坐标
    p1_x = min([rc1['left'], rc2['left']])
    p1_y = min([rc1['top'], rc2['top']])
    p2_x = max([[rc1['left'] + rc1['width']], [rc2['left'] + rc2['width']]])
    p2_y = max([[rc1['top'] + rc1['height']], [rc2['top'] + rc2['height']]])
    
    d = np.zeros((int(p2_y-p1_y), int(p2_x-p1_x)))
    
    d1 = rc1['data'].copy()
    d1[d1>0]=1.0
    d2 = rc2['data'].copy()
    d2[d2>0]=1.0
    
    x = rc1['left']-p1_x
    y = rc1['top']-p1_y
    w = rc1['width']
    h = rc1['height']
    d[y:y+h, x:x+w] += d1
    
    x = rc2['left']-p1_x
    y = rc2['top']-p1_y
    w = rc2['width']
    h = rc2['height']
    d[y:y+h, x:x+w] += d2

    overlap = np.sum(np.where(d==2.0, 1, 0))
    min_area = min([rc1['area'], rc2['area']])
    return overlap/min_area

def filterLowEnergy(parts):
    rets=[]
    energy = 0.0
    var = 0.0
    for i in range(len(parts)):
        energy += parts[i]['energy']
        var += parts[i]['energy']*parts[i]['energy']
    mean_energy = energy / len(parts)
    var = var/len(parts) - mean_energy*mean_energy
    for i in range(len(parts)):
        energy = parts[i]['energy']
        if energy >= mean_energy-var :
            rets.append(parts[i])
    
    return rets

def filterLowArea(parts):
    rets=[]
    area = 0.0
    var = 0.0
    for i in range(len(parts)):
        area += parts[i]['area']
        var += parts[i]['area']*parts[i]['area']
    mean = area / len(parts)
    var = var/len(parts) - mean*mean
    for i in range(len(parts)):
        area = parts[i]['area']
        if area >= mean+var :
            rets.append(parts[i])
    
    return rets
    
###构建Gabor滤波器 
"""
https://blog.csdn.net/vitaminc4/article/details/78840904
https://www.jianshu.com/p/f1d9f2482191
https://docs.opencv.org/3.4.4/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599
cv.getGaborKernel(	ksize, sigma, theta, lambd, gamma[, psi[, ktype]]	)
"""
def buildGaborFilters(ksize, lamda, sigma, direction, gamma, psi):
    filters = []
    thetas = np.arange(0, np.pi, np.pi / direction) #gabor方向
    for K in range(len(ksize)):
        kernels = []
        for theta in thetas:
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            kernels.append(kern)
        filters.append(kernels)
    return np.asarray(filters) #(ksize, theta, kernel, kernel)
  
#Gabor滤波过程
def GaborProcess(img, filters):
    accum = np.zeros_like(img)
    ret = []
    for i in range(filters.shape[0]):
        kern = filters[i]
        fimg = cv2.filter2D(img, -1, kern)
        fimg = np.maximum(accum, fimg)
        ret.append(fimg)
    return np.asarray(ret)

def doGaborFilter(img,filters):
    res = [] #滤波结果
    for i in range(filters.shape[0]):        
        res1 = GaborProcess(img, filters[i])
        res.append(res1)
    return np.asarray(res)        

"""
计算直方图
"""
def getHist(data, bins):
    data = data.flatten()
    h, _ = np.histogram(data, range=(data.min(), data.max()), bins = bins, density = True)
    return h

######################################
#部件相关方法
######################################
"""
在img上绘制部件
"""
def drawParts(img, parts):
    imgs=[]
    for i in range(len(parts)):
        img1 = img.copy()
        left = parts[i]['left']
        top = parts[i]['top']
        width = parts[i]['width']
        height = parts[i]['height']
        data = parts[i]['data'].copy()
        data[data>0.0] = i+1
        img1[top:top+height,left:left+width]=data
        imgs.append(img1)
    return imgs

"""
按照标准坐标（水平方向），计算基本部件距离。按照part0--->part1序计算。
如果在垂直方向重叠，part间无交叉，则为最近距离，返回值为正。
如果在垂直方向重叠，part间有交叉，则为正负距离的绝对值最大最小值，返回值为负。
如果在垂直方向无重叠，则返回maxsize
"""
def BPartDistanceY(part0, part1):
    dist = 0.0
    
    left0 = part0['left']
    top0 = part0['top']
    w0 = part0['width']
    h0 = part0['height']
    data0 = part0['data']

    left1 = part1['left']
    top1 = part1['top']
    w1 = part1['width']
    h1 = part1['height']
    data1 = part1['data']
    
    #取垂直方向重叠范围
    r_min = max(top0, top1)
    r_max = min(top0+h0-1, top1+h1-1)

    if r_max < r_min: #垂直方向无重叠
        dist = sys.maxsize
        return dist
    
    pos_min = sys.maxsize
    pos_max = 0
    neg_min = sys.maxsize
    neg_max = 0

    for i in range(r_min, r_max+1, 1):
        i0 = i-top0
        c0 = 0 #part0的i行最右位置
        for c0 in range(w0-1, -1, -1): #由右向左
            if(data0[i0][c0] > 0.0): break
        i1 = i-top1
        c1 = 0 #part1的i行最左位置
        for c1 in range(0, w1, 1): #由左向右
            if(data1[i1][c1] > 0.0): break

        d = left1+c1-(left0+c0)-1

        if(d >= 0.0):
            pos_min = min(pos_min, d)
            pos_max = max(pos_max, d)
        else:
            neg_min = min(neg_min, abs(d))
            neg_max = max(neg_max, abs(d))

    if ((neg_min != sys.maxsize and neg_max != 0) 
        and (pos_min !=sys.maxsize and pos_max != 0)): #正负距离均有。垂直方向重叠，part间有交叉
        dist = min(pos_max, neg_max)
    # 只有正或负距离。垂直方向重叠，part间无交叉。两种情况需分开考虑，因四个值均初始化，无法直接比较
    elif pos_min == sys.maxsize and pos_max == 0:
        dist = -1*neg_min
    elif neg_min == sys.maxsize and neg_max == 0:
        dist = pos_min

    return dist

"""
按照标准坐标（垂直方向），计算基本部件距离。按照part0--->part1序计算。
如果在水平方向重叠，part间无交叉，则为最近距离，返回值为正。
如果在水平方向重叠，part间有交叉，则为正负距离的绝对值最大最小值，返回值为负。
如果在水平方向无重叠，则返回maxsize
"""
def BPartDistanceX(part0, part1):

    dist = 0.0
    
    left0 = part0['left']
    top0 = part0['top']
    w0 = part0['width']
    h0 = part0['height']
    data0 = part0['data']

    left1 = part1['left']
    top1 = part1['top']
    w1 = part1['width']
    h1 = part1['height']
    data1 = part1['data']
    
    #取水平方向重叠范围
    c_min = max(left0, left1)
    c_max = min(left0+w0-1, left1+w1-1)

    if c_max < c_min: #水平方向无重叠
        dist = sys.maxsize
        return dist
    
    pos_min = sys.maxsize
    pos_max = 0
    neg_min = sys.maxsize
    neg_max = 0
    
    for i in range(c_min, c_max+1, 1):
        i0 = i-left0
        r0 = 0 #part0的i列最下位置
        for r0 in range(h0-1, -1, -1): #由下向上
            if(data0[r0][i0] > 0.0): break
        i1 = i-left1
        r1 = 0 #part1的i列最上位置
        for r1 in range(0, h1, 1): #由上向下
            if(data1[r1][i1] > 0.0): break
        
        d = top1+r1-top0-r0-1

        if(d >= 0.0):
            pos_min = min(pos_min, d)
            pos_max = max(pos_max, d)
        else:
            neg_min = min(neg_min, abs(d))
            neg_max = max(neg_max, abs(d))
    
    if ((neg_min != sys.maxsize and neg_max != 0) 
        and (pos_min !=sys.maxsize and pos_max != 0)): #正负距离均有。垂直方向重叠，part间有交叉
        dist = min(pos_max, neg_max)
    elif pos_min == sys.maxsize and pos_max == 0:
        dist = -1*neg_min
    elif neg_min == sys.maxsize and neg_max == 0:
        dist = pos_min
    # else:
    #     dist = max(pos_min, abs(neg_min)) #只有正或负距离。垂直方向重叠，part间无交叉

    return dist

"""
计算基本部件距离（水平和垂直最小距离）.按照part0--->part1序计算
"""
def BPartDistanceXY(part0, part1):
    dist_x = BPartDistanceX(part0, part1)
    dist_y = BPartDistanceY(part0, part1)
    return min(dist_x, dist_y)
    
"""
计算基本部件质心方向距离.按照part0--->part1序计算
"""
def BPartMCDistance(part0, part1):
    dist = 0.0
    data0 = part0['data']
    mc_r0, mc_c0 = getMassCenter(data0) #relative coordination
    data1 = part1['data']
    mc_r1, mc_c1 = getMassCenter(data1) #relative coordination
    
    center = (mc_r0, mc_c0)
    theta = math.atan((mc_c1-mc_c0)/(mc_r1-mc_r0))

    #旋转矩阵
    part0 = rotatePart(part0, center, theta)
    part1 = rotatePart(part1, center, theta)

    dist = BPartDistanceXY(part0, part1)
    return dist
    
"""
旋转部件(顺时针)
"""    
def rotatePart(part, center, theta):

    # 位置、数据
    left = part.get('left')
    top = part.get('top')
    width = part.get('width')
    height = part.get('height')
    data = part.get('data')

    # 四个顶点坐标
    R1 = [left, top]
    R2 = [left + width, top]
    R3 = [left, top + height]
    R4 = [left + width, top + height]
    R = [R1, R2, R3, R4]

    # 旋转后顶点坐标
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    R_rotate = []
    for x, y in R:
        r_x = M[0][0] * x + M[0][1] * y + M[0][2]
        r_y = M[1][0] * x + M[1][1] * y + M[1][2]
        r_x = int(round(r_x))
        r_y = int(round(r_y))
        R_rotate.append([r_x, r_y])
    R_rotate = np.transpose(np.array(R_rotate))
    X_max = max(R_rotate[0])
    X_min = min(R_rotate[0])
    Y_max = max(R_rotate[1])
    Y_min = min(R_rotate[1])

    # data位置坐标转换
    dict = {}
    for x in range(left, left+width):
        for y in range(top, top+height):
            r_x = M[0][0]*x+M[0][1]*y+M[0][2]
            r_y = M[1][0]*x+M[1][1]*y+M[1][2]
            r_x = int(round(r_x))
            r_y = int(round(r_y))
            dict["%d_%d" % (r_y,r_x)] = data[y-top,x-left]

    # 旋转后新矩形区域赋值
    r_data = np.zeros([Y_max-Y_min, X_max-X_min])
    for x in range(X_min, X_max+1):
        for y in range(Y_min, Y_max+1):
            if "%d_%d" % (y,x) in dict:
                r_data[y - Y_min][x - X_min] = dict["%d_%d" % (y,x)]

    return {"left":X_min, "top":Y_min, "width":X_max-X_min, "height":Y_max-Y_min, "data":r_data}

"""
计算基本部件欧式距离。只计算包围盒端点（左上和右下）间距离。
"""
def BPartL2Distance(part0, part1):
    left0 = part0['left']
    top0 = part0['top']
    w0 = part0['width']
    h0 = part0['height']
    
    left1 = part1['left']
    top1 = part1['top']
    w1 = part1['width']
    h1 = part1['height']
    
    pt00 = [left0, top0]
    pt01 = [left0+w0, top0+h0]
    pt10 = [left1, top1]
    pt11 = [left1+w1, top1+h1]
    
    dist0 = L2ofPoint(pt00, pt10)
    dist1 = L2ofPoint(pt00, pt11)
    dist2 = L2ofPoint(pt01, pt10)
    dist3 = L2ofPoint(pt01, pt11)
    # print([dist0,dist1, dist2, dist3])
    return min([dist0,dist1, dist2, dist3])

"""
判断部件是否在原图上联通。thres为阈值，推荐为线宽平均值。
dist: 为距离函数类型 mc, xy, L2，any。mc为质心距离，xy为水平垂直距离，L2为欧式距离，any为任何距离中最小距离
"""
def CheckPartConnect(part0, part1, thres, dist='any'):
    if dist=='mc': 
        dist = BPartMCDistance(part0, part1)
    if dist=='xy': 
        dist = BPartDistanceXY(part0, part1)
    if dist=='L2':
        dist = BPartL2Distance(part0, part1)
    if dist=='any':
        dist0 = BPartMCDistance(part0, part1)
        dist1 = BPartDistanceXY(part0, part1)
        dist2 = BPartL2Distance(part0, part1)
        dist = min([dist0, dist1, dist2])
    if dist < thres:
        return True
    else:
        return False

"""
测试类
"""
class Test:

    def __init__(self):

        """
            整张图片大小为20×20

            外接矩形坐标:
            part0 （3 ,3 ）（14,3 ）
                  （3 ,12）（14,12）

            part1 （4 ,3 ）（12,3 ）
                  （4 ,14）（12,14）

            part0->part1自左至右距离: 6,4,1,-1,-3,-5,-6,-7,-10,-11
            part0->part1自上至下距离: 7,5,3,1,-1,-3,-5,-8,-10
        """
        img_1_0 = cv2.imread("./1_0.png")
        img_1_0 = cv2.cvtColor(img_1_0, cv2.COLOR_BGR2GRAY)
        img_1_1 = cv2.imread("./1_1.png")
        img_1_1 = cv2.cvtColor(img_1_1, cv2.COLOR_BGR2GRAY)

        img_1_0[img_1_0==255] = 0
        img_1_1[img_1_1==255] = 0

        self.part1_0 = {"left": 3, "top": 3, "width": 12, "height": 10, "data": img_1_0[3:12+1, 3:14+1]}  # [y方向，x方向]
        self.part1_1 = {"left": 4, "top": 3, "width": 9, "height": 12, "data": img_1_1[3:14+1, 4:12+1]}

        """
            整张图片大小为20×20 
            外接矩形坐标：
            part0 (3,3 ) (7,3 )
                  (3,10) (7,10)
            part1 (10,0) (15,0)
                  (10,10) (15,10)
            part0->part1 自左至右距离： 3,3,5,5,6,7,8,9
        """
        img_2_0 = cv2.imread("./2_0.png")
        img_2_0 = cv2.cvtColor(img_2_0, cv2.COLOR_BGR2GRAY)
        img_2_0 = img_2_0 - 255
        img_2_1 = cv2.imread("./2_1.png")
        img_2_1 = cv2.cvtColor(img_2_1, cv2.COLOR_BGR2GRAY)
        img_2_1 = img_2_1 - 255

        img_2_0[img_2_0==255] = 0
        img_2_1[img_2_1==255] = 0

        self.part2_0 = {"left": 3, "top": 3, "width": 5, "height": 8, "data": img_2_0[3:10+1, 3:7+1]}  # [y方向，x方向]
        self.part2_1 = {"left": 10, "top": 0, "width": 6, "height": 11, "data": img_2_1[0:10+1, 10:15+1]}

        # print(img_1_0)
        # print(img_1_1)
        # print(img_2_0)
        # print(img_2_1)


    def test_BPart_(self):

        test1_BPartDistanceY = BPartDistanceY(self.part1_0, self.part1_1)
        test1_BPartDistanceX = BPartDistanceX(self.part1_0, self.part1_1)

        test2_BPartDistanceY = BPartDistanceY(self.part2_0, self.part2_1)
        test2_BPartDistanceX = BPartDistanceX(self.part2_0, self.part2_1)

        # print(test1_BPartDistanceX)
        # print(test2_BPartDistanceY)

        # BPartDistanceY
        assert test1_BPartDistanceY == 6 and test2_BPartDistanceY == 3
        # BPartDistanceX
        assert test1_BPartDistanceX == 7 and test2_BPartDistanceX == sys.maxsize

        # BPartL2Distance
        test1_BPartL2Distance = BPartL2Distance(self.part1_0, self.part1_1)
        assert test1_BPartL2Distance == 1.0


    def test_getMassCenter(self):

        arr = np.array([[0,0,1],[0,0,1],[0,0,1]])
        mc_r1, mc_c1 = getMassCenter(arr)
        # getMassCenter 图像坐标系下 x,y
        assert mc_c1 == 2 and mc_r1 == 1

    def test_BPartMCDistance(self):

        test2_BPartMCDistance = BPartMCDistance(self.part2_0,self.part2_1)
        assert  test2_BPartMCDistance == 3





if __name__ == '__main__':
    Test().test_BPartMCDistance()
    # Test().test_getMassCenter()
    # Test().test_BPart_()