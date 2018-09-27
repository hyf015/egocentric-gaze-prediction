import numpy as np
import os
import math
import operator
import cv2
from scipy import ndimage
from tqdm import tqdm
from scipy.interpolate import interp1d

txt = sorted(os.listdir('gazepositions'))
for i, t in enumerate(txt):
    txtname = t[:-4]
    print ('begin processing video %s'%txtname)
    
    gps = np.loadtxt('gazepositions/'+t)
    gpx = gps[:,0]
    gpy = gps[:,1]
    if gpx[0] == 0:
        gpx[0] = 320
    if gpy[0] == 0:
        gpy[0] = 240
    if gpx[-1] == 0:
        gpx[-1] = 320
    if gpy[-1] == 0:
        gpy[-1] = 240
    x = np.arange(len(gpx))
    idx = np.nonzero(gpx)
    interpx = interp1d(x[idx], gpx[idx])
    interpx = interpx(x)
    idx = np.nonzero(gpy)
    interpy = interp1d(x[idx], gpy[idx])
    interpy = interpy(x)
    interpx = interpx.clip(0,640)
    interpy = interpy.clip(0,480)

    fix_state = np.zeros((len(interpx),))
    fix_state[0] = 0
    center_pt = (interpx[0], interpy[0])
    fix_num = 0
    for pt_idx in range(1, interpx.shape[0]):
        current_pt = (interpx[pt_idx], interpy[pt_idx])
        if (center_pt[0]-current_pt[0])**2 + (center_pt[1]-current_pt[1])**2 < 2500.0:
            fix_state[pt_idx] = 1
            # update center_pt & fix_num
            fix_num += 1
            center_pt = tuple(map(operator.mul, center_pt, (fix_num, fix_num)))
            center_pt = tuple(map(operator.add, center_pt, current_pt))
            center_pt = tuple(map(operator.truediv, center_pt, (fix_num+1, fix_num+1)))
        else:
            fix_state[pt_idx] = 0
            if fix_num==1:
                fix_state[pt_idx-1] = 0
            fix_num = 0
            center_pt = current_pt

    file_name = t.strip().split('.')[0]+'_fixation.txt'
    fix_file = open(os.path.join('fixations', file_name), 'w')
    for pt_idx in range(0, interpx.shape[0]):
        fix_file.write(str(fix_state[pt_idx])+'\n')
    fix_file.close()


    '''
    for j in tqdm(range(len(interpx))):
        gazemap = np.zeros((480,640))
        gazemap[int(round(interpy[j])) - 1][int(round(interpx[j])) - 1] = 1
        gazemap = ndimage.filters.gaussian_filter(gazemap, 35)
        gazemap -= np.min(gazemap)
        gazemap /= np.max(gazemap)
        gazemap = np.uint8(255*gazemap)
        gazeim = cv2.resize(gazemap, (224,224), interpolation=cv2.INTER_AREA)
        cv2.imwrite('images/GTEA_Gaze/%s/gt_%05d.jpg'%(txtname,j), gazeim)
    print ('done for video %s' % txtname)
    '''
