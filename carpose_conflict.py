# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:59:23 2020

@author: yi683992
"""

from math import sqrt
import pandas as pd 
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from shapely import geometry
import re
from sqlalchemy import Table, Column, Float, Integer, String, MetaData, ForeignKey
from numpy.polynomial.polynomial import polyfit
import os
import math
from matplotlib import cm
import multiprocessing 
from multiprocessing import Pool
import time
start_time = time.time()
############################################################################################
root_path = os.getcwd()    
#raw2=pd.read_csv("Desktop/paper/With Jinghui/carpose/trajectory_v2.csv") 
raw3=pd.read_excel("Desktop/paper/With Jinghui/carpose/trajectory_v2.xlsx") 
lost=pd.read_excel("Desktop/paper/With Jinghui/carpose/trajectory_v3.xlsx") 



raw2=raw3.merge(lost, on=['objectID','frameNUM'],how='left')
raw2=raw2.loc[raw2['lostCounter']<5]
raw2['rfhy']=-raw2['rfhy']
raw2['lfhy']=-raw2['lfhy']
raw2['lbhy']=-raw2['lbhy']
raw2['rbhy']=-raw2['rbhy']
raw2['cy']=-raw2['cy']
raw2['p1y_y']=-raw2['p1y_y']
raw2['p2y_y']=-raw2['p2y_y']
raw2['by']=-raw2['by']
##repeat until test shape =0;
raw2=raw2.sort_values(by=['objectID','frameNUM'])
raw2['rfhx_p']= raw2.rfhx.shift(1)
raw2['rfhy_p']= raw2.rfhy.shift(1)
raw2['lfhx_p']= raw2.lfhx.shift(1)
raw2['lfhy_p']= raw2.lfhy.shift(1)
raw2['lbhx_p']= raw2.lbhx.shift(1)
raw2['lbhy_p']= raw2.lbhy.shift(1)
raw2['rbhx_p']= raw2.rbhx.shift(1)
raw2['rbhy_p']= raw2.rbhy.shift(1)
test=pd.DataFrame()


lenraw2=raw2.shape[0]
#for k in range (0,lenraw2):
#    if raw2['iou'][k]==0 and raw2['rfhy'][k]==-1:
#        raw2['rfhx'][k] =raw2['rfhx_p'][k]
#        raw2['rfhy'][k] =raw2['rfhy_p'][k]  
#        raw2['lfhx'][k] =raw2['lfhx_p'][k]  
#        raw2['lfhy'][k] =raw2['lfhy_p'][k]  
#        raw2['lbhx'][k] =raw2['lbhx_p'][k]  
#        raw2['lbhy'][k] =raw2['lbhy_p'][k]         
#        raw2['rbhx'][k] =raw2['rbhx_p'][k]  
#        raw2['rbhy'][k] =raw2['rbhy_p'][k]  
raw2=raw2.loc[raw2['rfhx']>0]
raw2=raw2.loc[ raw2['lfhx']>0]
raw2=raw2.loc[raw2['lbhx']>0]
raw2=raw2.loc[raw2['rbhx']>0]
raw2=raw2.loc[raw2['rfhy']<0]
raw2=raw2.loc[raw2['lfhy']<0]
raw2=raw2.loc[raw2['lbhy']<0]
raw2=raw2.loc[raw2['rbhy']<0]

#test.shape[0]
        
raw2['rfhx'] = raw2.groupby('objectID')['rfhx'].transform(lambda x: x.rolling(10, 1).mean())
raw2['rfhy'] = raw2.groupby('objectID')['rfhy'].transform(lambda x: x.rolling(10, 1).mean())
raw2['lfhx'] = raw2.groupby('objectID')['rfhx'].transform(lambda x: x.rolling(10, 1).mean())
raw2['lfhy'] = raw2.groupby('objectID')['lfhy'].transform(lambda x: x.rolling(10, 1).mean())
raw2['lbhx'] = raw2.groupby('objectID')['lbhx'].transform(lambda x: x.rolling(10, 1).mean())
raw2['lbhy'] = raw2.groupby('objectID')['lbhy'].transform(lambda x: x.rolling(10, 1).mean())
raw2['rbhx'] = raw2.groupby('objectID')['rbhx'].transform(lambda x: x.rolling(10, 1).mean())
raw2['rbhy'] = raw2.groupby('objectID')['rbhy'].transform(lambda x: x.rolling(10, 1).mean())
raw=raw2.iloc[::10,::]

test=raw.groupby(['objectID'], as_index=False).min()
###############################################################################################
####bounding
conflict_b=pd.DataFrame()
lenraw=raw.shape[0]
#i=0
#j=1
for i in range (0,lenraw):
    print("i=" + str(i) )
    frame1=raw.iloc[i,0]
    car_id1=raw.iloc[i,1]
    p1x1=raw.iloc[i,31]
    p1y1=raw.iloc[i,32]
    p2x1=raw.iloc[i,33]
    p2y1=raw.iloc[i,34]
    cx1=raw.iloc[i,35]
    cy1=raw.iloc[i,36]
    bx1=raw.iloc[i,37]
    by1=raw.iloc[i,38]
    p3x1=2*bx1-p2x1
    p3y1=2*by1-p2y1
    p4x1=2*cx1-p3x1
    p4y1=2*cy1-p3y1
    a1=geometry.Point(p1x1,p1y1)
    a2=geometry.Point(p2x1,p2y1)
    a3=geometry.Point(p3x1,p3y1)
    a4=geometry.Point(p4x1,p4y1)
    p1=[a1,a3,a2,a4]
    poly1= geometry.Polygon(p1)
      # plt.scatter(p1x,p1y)
    #plt.scatter(p2x,p2y)
    #plt.scatter(bx,by)
    x_b=np.array([ p1x1, p3x1, p2x1, p4x1])
    y_b=np.array([ p1y1, p3y1, p2y1, p4y1])
    #plt.fill(x_b,y_b,facecolor="none",edgecolor='red')
    #plt.show()
    
    raw_j=raw[(raw['frameNUM']>frame1)&(raw['frameNUM']<(frame1+25*3))]
    raw_j=raw_j.reset_index()
    raw_j=raw_j.drop(columns=['index'])

    for j in  range (0,len(raw_j)):
        frame2=raw_j['frameNUM'][j]
        car_id2=raw_j['objectID'][j]
        p1x2=raw_j['p1x_y'][j]
        p1y2=raw_j['p1y_y'][j]
        p2x2=raw_j['p2x_y'][j]
        p2y2=raw_j['p2y_y'][j]
        cx2=raw_j['bx'][j]
        cy2=raw_j['by'][j]
        bx2=raw_j['cx'][j]
        by2=raw_j['cy'][j]
        p3x2=2*bx2-p2x2
        p3y2=2*by2-p2y2
        p4x2=2*cx2-p3x2
        p4y2=2*cy2-p3y2
        b1=geometry.Point(p1x2,p1y2)
        b2=geometry.Point(p2x2,p2y2)
        b3=geometry.Point(p3x2,p3y2)
        b4=geometry.Point(p4x2,p4y2)
        p2=[b1,b3,b2,b4]
        poly2= geometry.Polygon(p2)
      # po1y2 = Polygon([(p1x2, p1y2), (p2x2, p2y2),(p3x2,p3y2),(p4x2,p4y2)])  
        if  car_id2 !=  car_id1:
           #columns=['car_id1','car_id2']

           #conflict_temp =pd.DataFrame(columns=columns)
           #conflict_temp=conflict_temp.fillna(0)
            intersection = poly1.intersects(poly2)
            if intersection == True:
                #conflict_temp['car_id1']=car_id1
                #conflict_temp['car_id2']=car_id2
                #conflict_temp['frame1']=frame1
                #conflict_temp['PET'][1]=abs(frame1-frame2)
                PET=frame2-frame1
                carid_min=min(car_id1,car_id2)
                carid_max=max(car_id1,car_id2)
                conflictb_temp =pd.DataFrame({'frame1':[frame1],'carid_min':[carid_min],'carid_max':[carid_max],'cx1':cx1,'cy1':cy1,'PET':[PET]})
                conflict_b=conflict_b.append(conflictb_temp)
               
                print("detect conflict" +str(i) + str(j) )
            else:
                print("no conflict")


conflict_b1=conflict_b.groupby(['carid_min','carid_max'], as_index=False).min()
conflict_b2=conflict_b1.loc[conflict_b1['cx1']>0]
conflict_b2=conflict_b2.loc[conflict_b2['cy1']<0]
conflict_b2.plot(x='cx1',y='cy1', style='o')




###############################################################################################
#car pose
conflict_c =pd.DataFrame()
#raw=raw[raw2['rfhx']>0]
lenraw=raw.shape[0]
#i=0
#j=41
for i in range (0,lenraw):
    try:
        print("i=" + str(i) )
        frame1=raw.iloc[i,0]
        car_id1=raw.iloc[i,1]
        rfhx1=raw.iloc[i,7]
        rfhy1=raw.iloc[i,8]
        lfhx1=raw.iloc[i,9]
        lfhy1=raw.iloc[i,10]
        lbhx1=raw.iloc[i,11]
        lbhy1=raw.iloc[i,12]
        rbhx1=raw.iloc[i,13]
        rbhy1=raw.iloc[i,14]
        
        a1=geometry.Point(rfhx1,rfhy1)
        a2=geometry.Point(lfhx1,lfhy1)
        a3=geometry.Point(lbhx1,lbhy1)
        a4=geometry.Point(rbhx1,rbhy1)
        p1=[a1,a3,a2,a4]
        poly1= geometry.Polygon(p1)
        center1=list(poly1.centroid.coords)[0]
        center1_x=center1[0]
        center1_y=center1[1]
        #raw['center1_x'][i]=center1[0]
        #raw['center1_y'][i]=center1[1]
        raw_j=raw[(raw['frameNUM']>frame1)&(raw['frameNUM']<(frame1+25*3))]
        raw_j=raw_j.reset_index()
        raw_j=raw_j.drop(columns=['index'])

        for j in  range (0,len(raw_j)):
            frame2=raw_j.iloc[j,0]
            car_id2=raw_j.iloc[j,1]
            rfhx2=raw_j.iloc[j,7]
            rfhy2=raw_j.iloc[j,8]
            lfhx2=raw_j.iloc[j,9]
            lfhy2=raw_j.iloc[j,10]
            lbhx2=raw_j.iloc[j,11]
            lbhy2=raw_j.iloc[j,12]
            rbhx2=raw_j.iloc[j,13]
            rbhy2=raw_j.iloc[j,14]
        
            b1=geometry.Point(rfhx2,rfhy2)
            b2=geometry.Point(lfhx2,lfhy2)
            b3=geometry.Point(lbhx2,lbhy2)
            b4=geometry.Point(rbhx2,rbhy2)
            p2=[b1,b3,b2,b4]
            poly2= geometry.Polygon(p2)
               
            if  car_id2 !=  car_id1:
           #columns=['car_id1','car_id2']

           #conflict_temp =pd.DataFrame(columns=columns)
           #conflict_temp=conflict_temp.fillna(0)
                intersection = poly1.intersects(poly2)
                if intersection == True:
                #conflict_temp['car_id1']=car_id1
                #conflict_temp['car_id2']=car_id2
                #conflict_temp['frame1']=frame1
                #conflict_temp['PET'][1]=abs(frame1-frame2)
                    PET=frame2-frame1
                    carid_min=min(car_id1,car_id2)
                    carid_max=max(car_id1,car_id2)
                    conflictc_temp =pd.DataFrame({'frame1':[frame1],'carid_min':[carid_min],'carid_max':[carid_max],'center1_x':[center1_x],'center1_y':[center1_y],'PET':[PET]})
                    conflict_c=conflict_c.append(conflictc_temp)
               
                    print("detect conflict" +str(i) + str(j) )
                #else:

                    #print("no conflict")
    except:
        print ("ERROR @ j " +str(j) +" i" +str(i))
        
conflict_c2=conflict_c.groupby(['carid_min','carid_max'], as_index=False).min()
conflict_c3=conflict_c2.loc[conflict_c2['center1_x']>0]
conflict_c3=conflict_c3.loc[conflict_c3['center1_y']<0]
conflict_c3=conflict_c3.loc[conflict_c3['center1_y']>-750]
conflict_c3.plot(x='center1_x',y='center1_y', style='o')
#####################################################################################3



conflict_b=pd.read_excel("Desktop/paper/With Jinghui/carpose/conflict_b.xlsx") 
conflict_final=conflict_c3.merge(conflict_b2, on=['carid_min','carid_max'],how='inner')
threshold=1.5*25
conflict_b4=conflict_b2[conflict_b2['PET']<threshold]
conflict_c4=conflict_c3[conflict_c3['PET']<threshold]
conflict_b4.plot(x='cx1',y='cy1', style='o')
conflict_c4.plot(x='center1_x',y='center1_y', style='o')
