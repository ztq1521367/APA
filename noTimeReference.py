import numpy as np  
import matplotlib.pyplot as plt  
from scipy import linalg
import math
import time
from threading import Thread
import pdb
import random
import copy
import re


"""
输入：车位宽度，车辆起始点，最小停车位宽度，车距离车位的最短距离和最长距离（这个可以根据具体车型事先计算好）

输出：计算避障点坐标，终止点坐标，以及每个控制点的航向角，避障点可以有多个：驶离车位时右前角，驶出车位时右后角;
     根据控制点的约束条件，解线性方程组，得到多项式系数，拟合曲线，确认各个控制点对应的坐标是否满足约束要求，曲线曲率是否满足要求。
     如果不满足，采用多段路径进行规划，

     最后规划出符合要求的路径。

"""

#水平泊车
#参数系数 单位：米
safeDistance = 0.2          #安全距离
minParkLen = 6.6          #最小停车位宽度
minDiffWidth = 1            #车距离车位最小距离
carWidth = 1.85             #车宽
carLen = 4.85            #车高
rearLen = 1.0               #后悬
frontLen = 1.0             #前悬
axialLen = 1.5              #轴长
axialDistance = 2.8         #轴距
maxWheelAngle = 40          #最大车轮转角
zPointX = 2.9               #避障点x坐标
zPointY = -0.91             #避障点y坐标
zAngle = 30                 #避障点的航向角

carSize = [4.8, 1.8]

rrp = [-rearLen, -carSize[1]/2]
lrp = [-rearLen, carSize[1]/2]
lfp = [axialDistance+frontLen, carSize[1]/2]
rfp = [axialDistance+frontLen, -carSize[1]/2]

stopFlag = 1

hxAngle = 0
raxPoint = [0,0]

class car:
    """
    车辆信息
    """
    def __init__(self, carWidth, carLen, frontLen, rearLen, axialLen, axialDistance, maxWheelAngle):
        self.width = carWidth               #车宽
        self.height = carLen             #车宽
        self.frontLen = frontLen            #前悬
        self.rearLen = rearLen              #后悬
        self.axialLen = axialLen            #轴长
        self.axialDistance = axialDistance  #轴距
        self.maxWheelAngle = maxWheelAngle  #车轮最大转角

class contrlPoint:
    """
    控制点信息
    """
    def __init__(self, startX, startY, startAngle, barrierX, barrierY, barrierAngle, endX, endY, endAngle):
        self.startX = startX               
        self.startY = startY               
        self.startAngle = startAngle
        self.barrierX = barrierX           
        self.barrierY = barrierY           
        self.barrierAngle = barrierAngle   
        self.endX = endX                   
        self.endY = endY                   
        self.endAngle = endAngle           

def calAvoidBarrierPoint(parkL, parkW, carInfo):
    """
    计算避障点坐标以及此时的航向角
    parkW:车位长
    parkH:车位宽
    carInfo:车辆参数信息，如：长宽、轴距、轴长等
    """
    # TODO:
    barrierX = 2.9
    barrierY = -1.
    barrierAngle = 0.43633 #0.523333     #0.35
    return barrierX, barrierY, barrierAngle

def calContrlPoint(parkL, parkW, startX, startY, startAngle):
    """
    计算输出的起始点坐标，避障点坐标，终止点坐标
    parkL:车位长
    parkW:车位宽
    startX:起始点x坐标
    startY:起始点y坐标
    startAngle:起始点航向角
    """

    ctrlPoint = {}

    carInfo = car(carWidth, carLen, frontLen, rearLen, axialLen, axialDistance, maxWheelAngle)
    barrierX, barrierY, barrierAngle = calAvoidBarrierPoint(parkL, parkW, carInfo)

    endX = safeDistance + rearLen
    endY = -parkW / 2

    ctrlPoint['startX'] = startX
    ctrlPoint['startY'] = startY
    ctrlPoint['startAngle'] = startAngle
    ctrlPoint['barrierX'] = barrierX
    ctrlPoint['barrierY'] = barrierY
    ctrlPoint['barrierAngle'] = barrierAngle
    ctrlPoint['endX'] = endX
    ctrlPoint['endY'] = endY
    ctrlPoint['endAngle'] = 0
    
    return ctrlPoint

def solve(ctrlPoint, ifAvoid, ax):
    """
    建立多项式模型
    y = a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0

    y0 = a5*x0^5 + a4*x0^4 + a3*x0^3 + a2*x0^2 + a1*x0 + a0
    tan(startAngle) = 5*a5*x0^4 + 4*a4*x0^3 + 3*a3*x0^2 + 2*a2*x0 + a1
    y0'' = 20*a5*x0^3 + 12*a4*x0^2 + 6*a3*x0 + 2*a2//可选
    y1 = a5*x1^5 + a4*x1^4 + a3*x1^3 + a2*x1^2 + a1*x1 + a0
    tan(barrierAngle) = 5*a5*x1^4 + 4*a4*x1^3 + 3*a3*x1^2 + 2*a2*x1 + a1
    y2 = a5*x2^5 + a4*x2^4 + a3*x2^3 + a2*x2^2 + a1*x2 + a0
    tan(0) = 5*a5*x2^4 + 4*a4*x2^3 + 3*a3*x2^2 + 2*a2*x2 + a1
    y2'' = 20*a5*x2^3 + 12*a4*x2^2 + 6*a3*x2 + 2*a2//可选
    """
    x0 = ctrlPoint['startX']
    y0 = ctrlPoint['startY']
    startAngle = ctrlPoint['startAngle']

    x2 = ctrlPoint['endX']
    y2 = ctrlPoint['endY']
    endAngle = ctrlPoint['endAngle']
    
    if ifAvoid == 'yes':
        x1 = ctrlPoint['barrierX']
        y1 = ctrlPoint['barrierY']
        barrierAngle = ctrlPoint['barrierAngle']
        A=np.array([[pow(x0,5), pow(x0,4), pow(x0,3), pow(x0,2), x0, 1],\
                    [5*pow(x0,4), 4*pow(x0,3), 3*pow(x0,2), 2*x0, 1, 0],\
                    [pow(x1,5), pow(x1,4), pow(x1,3), pow(x1,2), x1, 1],\
                    [5*pow(x1,4), 4*pow(x1,3), 3*pow(x1,2), 2*x1, 1, 0],\
                    [pow(x2,5), pow(x2,4), pow(x2,3), pow(x2,2), x2, 1],\
                    [5*pow(x2,4), 4*pow(x2,3), 3*pow(x2,2), 2*x2, 1, 0]])
                    #[20*pow(x2,3), 12*pow(x2,2), 6*pow(x2,1), 2, 0, 0]])
        B=np.array([y0, math.tan(startAngle), y1,  math.tan(barrierAngle), y2, math.tan(endAngle)]) #, 0.0]) #math.tan(barrierAngle),
        xlist = [x0, x1, x2]
        ylist = [y0, y1, y2]
    else:        
        #两个控制点的多项式方程组
        A=np.array([[pow(x0,4), pow(x0,3), pow(x0,2), x0, 1],\
                    [4*pow(x0,3), 3*pow(x0,2), 2*x0, 1, 0],\
                    [pow(x2,4), pow(x2,3), pow(x2,2), x2, 1],\
                    [4*pow(x2,3), 3*pow(x2,2), 2*x2, 1, 0]])
        B=np.array([y0, math.tan(startAngle), y2, math.tan(endAngle)])
        xlist = [x0, x2]
        ylist = [y0, y2]

    X=linalg.lstsq(A,B) #解超定方程组
    #X=linalg.solve(A,B) #解定方程组
    '''    
    xx = np.arange(x2, x0, 0.01) 
    p1 = np.poly1d(X[0])
    pp1=p1(xx)

    #曲率计算
    xx = np.arange(x2, x0, 0.01)
    ydf1 = lambda x: 5*X[0][0]*pow(x,4) + 4*X[0][1]*pow(x,3) + 3*X[0][2]*pow(x,2) + 2*X[0][3]*x + X[0][4]
    ydf2 = lambda x: 20*X[0][0]*pow(x,3) + 12*X[0][1]*pow(x,2) + 6*X[0][2]*pow(x,1) + 2*X[0][3]

    k = []
    for xinx in xx:
        k.append(abs(ydf2(xinx) / pow( 1 + pow(ydf1(xinx),2), 1.5)))

    print('k: ', k)
    print('xx: ',xx)

    ax.plot(xlist,ylist, ',', linewidth=1, color='m')#(xlist,ylist,color='m','.', linewidth=1, label=u'lhdata')
    ax.plot(xx,pp1, '-', linewidth=1, color='b')#(xx,pp1,color='b','-', linewidth=1, label=u"lhquxian") 
    ax.plot(xx,k, '-', linewidth=1, color='r')#(xx,pp1,color='b','-', linewidth=1, label=u"lhquxian")
    '''
    return X[0]


def carMove(tyreAngle = 0, speed = 0, line = None, line1 = None, clearFlag = None):
    """
    车移动状态仿真
    """
    global rrp, lfp, lrp, rfp, carSize, axialDistance, hxAngle, raxPoint

    cta = -tyreAngle #*0.017444444
    s = -speed * 10
    
    if cta != 0:

        r1 = axialDistance/math.tan(cta)
        r = r1 + carSize[1]/2
        moveAngle = s / r

        #中轴点为原点的转弯圆心坐标
        rax0Circle = [0, -r]

        #转弯圆心旋转航向角后的坐标
        sinHxAngle = math.sin(hxAngle)
        cosHxAngle = math.cos(hxAngle)
        rax0HxCircle = [rax0Circle[0]*cosHxAngle-rax0Circle[1]*sinHxAngle, rax0Circle[0]*sinHxAngle+rax0Circle[1]*cosHxAngle]

        #以转弯圆心旋转航向角后的坐标为原点坐标,并旋转cta角度
        sinMoveAngle = math.sin(moveAngle)
        cosMoveAngle = math.cos(moveAngle)
        rax0HxCircleO = [-rax0HxCircle[0]*cosMoveAngle+rax0HxCircle[1]*sinMoveAngle, -rax0HxCircle[0]*sinMoveAngle-rax0HxCircle[1]*cosMoveAngle]

        raxPoint = [rax0HxCircleO[0]+rax0HxCircle[0]+raxPoint[0], rax0HxCircleO[1]+rax0HxCircle[1]+raxPoint[1]]
        
        hxAngle += moveAngle

        sinHxAngle = math.sin(hxAngle)
        cosHxAngle = math.cos(hxAngle)
        rrpm = [(rrp[0]*cosHxAngle-rrp[1]*sinHxAngle + raxPoint[0]), (rrp[0]*sinHxAngle+rrp[1]*cosHxAngle + raxPoint[1])]
        lrpm = [(lrp[0]*cosHxAngle-lrp[1]*sinHxAngle + raxPoint[0]), (lrp[0]*sinHxAngle+lrp[1]*cosHxAngle + raxPoint[1])]
        lfpm = [(lfp[0]*cosHxAngle-lfp[1]*sinHxAngle + raxPoint[0]), (lfp[0]*sinHxAngle+lfp[1]*cosHxAngle + raxPoint[1])]
        rfpm = [(rfp[0]*cosHxAngle-rfp[1]*sinHxAngle + raxPoint[0]), (rfp[0]*sinHxAngle+rfp[1]*cosHxAngle + raxPoint[1])]

        XX = [rrpm[0], lrpm[0], lfpm[0], rfpm[0], rrpm[0]]
        YY = [rrpm[1], lrpm[1], lfpm[1], rfpm[1], rrpm[1]]

    else:
        sinHxAngle = math.sin(hxAngle)
        cosHxAngle = math.cos(hxAngle)
        raxPoint = [raxPoint[0]-cosHxAngle*s, raxPoint[1]-sinHxAngle*s]

        rrpm = [rrp[0]*cosHxAngle-rrp[1]*sinHxAngle + raxPoint[0], rrp[0]*sinHxAngle+rrp[1]*cosHxAngle + raxPoint[1]]
        lrpm = [lrp[0]*cosHxAngle-lrp[1]*sinHxAngle + raxPoint[0], lrp[0]*sinHxAngle+lrp[1]*cosHxAngle + raxPoint[1]]
        lfpm = [lfp[0]*cosHxAngle-lfp[1]*sinHxAngle + raxPoint[0], lfp[0]*sinHxAngle+lfp[1]*cosHxAngle + raxPoint[1]]
        rfpm = [rfp[0]*cosHxAngle-rfp[1]*sinHxAngle + raxPoint[0], rfp[0]*sinHxAngle+rfp[1]*cosHxAngle + raxPoint[1]]

        XX = [rrpm[0], lrpm[0], lfpm[0], rfpm[0], rrpm[0]]
        YY = [rrpm[1], lrpm[1], lfpm[1], rfpm[1], rrpm[1]]

    if not hasattr(carMove, 'raxList'):
        carMove.raxList = [[],[]]

    if clearFlag is False:
       carMove.raxList[0].append(raxPoint[0])
       carMove.raxList[1].append(raxPoint[1])

       line.set_data(carMove.raxList[0],carMove.raxList[1])
       line1.set_data(XX,YY)
       plt.draw() #line.draw()
    else:
       carMove.raxList = [[],[]]
       carMove.raxList[0].append(raxPoint[0])
       carMove.raxList[1].append(raxPoint[1])
       #ax.plot(raxPoint[0],raxPoint[1],color='m',linestyle='',marker='o')

    return s   


def notTimeCtrl(thetar,theta, yr, y, curvature): #pDiff, caDiff
    """
    基于非时间参考的路径跟踪控制
    """
    k1 = 200 #180
    k2 = 100 #160
    expression1 = axialDistance*pow(math.cos(theta), 3)
    expression2 = k1*(yr-y)-k2*(math.tan(thetar)-math.tan(theta))#倒车用-k2  前进用+k2
    expression3 = (math.cos(thetar)**2)*math.cos(theta)
    outCtrl = expression1 * (expression2 + curvature/expression3)
    if outCtrl > 0.698131689:
        outCtrl = 0.698131689
    elif outCtrl < -0.698131689:
        outCtrl = -0.698131689
    '''
    if not hasattr(notTimeCtrl, 'maxK'):
        notTimeCtrl.maxK = 0

    if notTimeCtrl.maxK < abs(curvature):
        notTimeCtrl.maxK = abs(curvature)
    print('maxK: ', notTimeCtrl.maxK)
    '''
    print('outCtrl: ', outCtrl)
    return outCtrl

def mulPointCtrl(theta, thetaq, x, xq, y, yq, flag):
    """
    多点控制算法
    """
    k3 = 200
    k4 = 100
    xzAlg = 0.523598767
    #围绕坐标原点做旋转  rax0Circle[0]*cosHxAngle-rax0Circle[1]*sinHxAngle, rax0Circle[0]*sinHxAngle+rax0Circle[1]*cosHxAngle
    '''
    x = xx*math.cos(xzAlg) - yy*math.sin(xzAlg)
    y = xx*math.sin(xzAlg) + yy*math.cos(xzAlg)

    xq = xqq*math.cos(xzAlg) - yqq*math.sin(xzAlg)
    yq = xqq*math.sin(xzAlg) - yqq*math.cos(xzAlg)
    
    thetaq += xzAlg
    theta += xzAlg
    '''
    xe = (x - xq) * math.cos(thetaq) + (y - yq) * math.sin(thetaq)
    ye = (y - yq) * math.cos(thetaq) - (x - xq) * math.sin(thetaq)
    thetae = theta - thetaq

    expression1 = pow(math.cos(thetae),3)*axialDistance

    if flag is True:
        expression2 = -k3*ye - k4*math.tan(thetae)
    else:
        expression2 = -k3*ye + k4*math.tan(thetae)

    outCtrl = math.atan(expression1 * expression2)

    if outCtrl > 0.698131689:
        outCtrl = 0.698131689
    elif outCtrl < -0.698131689:
        outCtrl = -0.698131689

    return outCtrl


def apaTest(adaptationParam, startAngle, mohuMatrix, sleepFlag, ctrlFlag):
    """
    水平泊车位测试
    adaptationParam = startP, p1, xlp,
    """
    global hxAngle, raxPoint, safeDistance, rearLen

    #pdb.set_trace() #debug调试
    startP = adaptationParam[0]
    p1 = adaptationParam[1]
    xlp = adaptationParam[2]
    xlp2 = adaptationParam[3]
    lineTmp = adaptationParam[4]
    lineTmp1 = adaptationParam[5]

    outCtrl = 0

    hxAngle = startAngle
    raxPoint = copy.deepcopy(startP)
    #raxPoint = [1.2,-1.5]

    conDiff = 0
    angleDiff = 0
    mvs = 0

    if raxPoint[0] < xlp:
        dirFlag = True
    else:
        dirFlag = False

    #print('move start ')
    limit = xlp #safeDistance + rearLen
    clearFlag = True
    while 1:
        if raxPoint[0] <= limit: #mvs >= 8.0: # >= startP[0]: #
           break
        if ctrlFlag is True:

            curY = p1(raxPoint[0])

            #航向角误差
            curAngle = math.atan(xlp(raxPoint[0]))

            ydf1 = xlp(raxPoint[0])
            ydf2 = xlp2(raxPoint[0])

            #当前理想路径的曲率
            k = ydf2 / pow(1 + pow(ydf1,2), 1.5)

            outCtrl = notTimeCtrl(curAngle, hxAngle, curY, raxPoint[1], k)

            if not hasattr(apaTest, 'maxK'):
                apaTest.maxK = [0,0]

            if apaTest.maxK[0] < abs(k):
                apaTest.maxK = [abs(k), raxPoint[0]] 
            print('maxK: ', apaTest.maxK)
        else:
            outCtrl = mulPointCtrl(hxAngle, p1, raxPoint[0], xlp, raxPoint[1], xlp2, dirFlag)

        mvs += abs(carMove(tyreAngle = outCtrl, speed = -0.0005, line = lineTmp, line1 = lineTmp1, clearFlag = clearFlag))#-0.000833333
        clearFlag = False

        #pdb.set_trace() #debug调试
        if sleepFlag is True:
            time.sleep(0.01)

    return conDiff, angleDiff

def ycBegin(params, ax, ax1):
    '''
    遗传测试开始
    '''
    global safeDistance,rearLen

    #初始化图表
    line, = ax1.plot([], [], '-', linewidth=1, color='r')
    line1, = ax1.plot([], [], '-', linewidth=1, color='g')
    line2, = ax1.plot([], [], '-', linewidth=1, color='b')

    #startx = np.arange(params['startXs'], params['startXe'], 0) #0.1)
    #starty = np.arange(params['startYs'], params['startYe'], 0) #0.1)

    #路径拟合
    p1 = []
    xlp = []
    xlp2 = []
    startPoint = []
    count = 0
    iny = params['startYs']
    inx = params['startXs']
    #for iny in starty:
        #for inx in startx:

    ctrlPoint = calContrlPoint(params['parkL'], params['parkW'], inx, iny, params['startAngles'])
    ctrlPoint['endAngle'] = params['endAngles']
    ifAvoid = 'yes'
    plotCoeff = solve(ctrlPoint, ifAvoid, ax1)
    startPoint.append([inx, iny])
    #多项式计算
    p1.append(np.poly1d(plotCoeff))
 
    xlp.append(lambda x: 5*plotCoeff[0]*pow(x,4) + 4*plotCoeff[1]*pow(x,3) + 3*plotCoeff[2]*pow(x,2) + 2*plotCoeff[3]*x + plotCoeff[4])
    xlp2.append(lambda x: 20*plotCoeff[0]*pow(x,3) + 12*plotCoeff[1]*pow(x,2) + 6*plotCoeff[2]*x + 2*plotCoeff[3])

    count+=1

    ctrlType = False #多控制点

    for j in range(count): 
        if ctrlType is False:
            xx = np.arange(safeDistance + rearLen, startPoint[j][0], 0.01) 
            pp1=p1[j](xx)

            line2.set_data(xx,pp1)
            apaTest((startPoint[j], p1[j], xlp[j], xlp2[j], line, line1), params['startAngles'], None, True, True)
        else:
            apaTest((startPoint[j], ctrlPoint['endAngle'], ctrlPoint['endX'], ctrlPoint['endY'], line, line1), params['startAngles'], None, True, False)
        plt.draw()

    print('learn complete ')


def mulPointCtrlZ(params, ax, ax1):
    """
    使用多点控制的路径跟踪算法
    """
    line, = ax1.plot([], [], '-', linewidth=1, color='r')
    line1, = ax1.plot([], [], '-', linewidth=1, color='g')
    #line2, = ax1.plot([], [], '.', linewidth=1, color='b')
    #ax.plot(xlist,ylist,color='m',linestyle='',marker='.',label=u'lhdata')
    startPoint = [params['startXs'], params['startYs']]
    startAngle = params['startAngles']
    distPoint = [params['endX'], params['endY']]
    distAngle = params['endAngles']

    drawP = [[startPoint[0], distPoint[0]], [startPoint[1], distPoint[1]]]

    ax1.plot(drawP[0],drawP[1],color='b',linestyle='',marker='.',label=u'lhdata')
    apaTest((startPoint, distAngle, distPoint[0], distPoint[1], line, line1), startAngle, None, True, False)

def main(params):

    fig = plt.figure(figsize = [10,10])
    #ax = fig.add_subplot(2,1,1,xlim=(-5, 15), ylim=(-5, 15))
    ax1 = fig.add_subplot(1,1,1,xlim=(0, 10), ylim=(-5, 5))
    t = Thread(target = mulPointCtrlZ, args=(params, None, ax1))#ycBegin
    t.start()
    plt.show()          

if __name__ == '__main__':
    params = {
        'parkL': 7.5,
        'parkW': 3,
        'startXs': 9.5, #6.5, #7.5,
        'startYs': 1.4, #1.6, #1.0,
        'startAngles': 0.1,
        'startXe': 8.5, #9.5,
        'startYe': 1.4, #2.0,
        'startAnglee': 0,
        'endX':6.5,#1.5,
        'endY':1.0,#-4.0,
        'endAngles':0.2 #1.221730456
    }
    main(params)