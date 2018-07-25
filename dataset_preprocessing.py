import os
import cv2
import numpy as np
from scipy import ndimage
import scipy.misc
from skimage import io
from shutil import copyfile
img_size = (256, 192) #for opencv
gtea_size = (960, 1280) #for np

def parsetxt(filename):
	gazex = []
	gazey = []
	nframe = []
	fixsac = []
	file = open(filename, 'r')
	for line in file:
		if line.startswith('#') or line.startswith('T'):
			continue
		else:
			s=line.split()
			# s[3] is x, s[4] is y, s[5] is frame
			if len(nframe) == 0:
				if int(round(float(s[3]))) in range(1280) and int(round(float(s[4]))) in range(960):
					nframe.append(int(s[5]))
					gazex.append(float(s[3]))
					gazey.append(float(s[4]))
					if 'Fix' in s[6]:
						fixsac.append(1)
					else:
						fixsac.append(0)
				else:
					nframe.append(0)
					gazex.append(640.0)
					gazey.append(480.0)
					if 'Fix' in s[6]:
						fixsac.append(1)
					else:
						fixsac.append(0)
			elif int(s[5]) not in nframe:
				if nframe[-1] + 1 == int(s[5]):
					if int(round(float(s[3]))) in range(1280) and int(round(float(s[4]))) in range(960):
						nframe.append(int(s[5]))
						gazex.append(float(s[3]))
						gazey.append(float(s[4]))
						if 'Fix' in s[6]:
							fixsac.append(1)
						else:
							fixsac.append(0)
					else:
						nframe.append(int(s[5]))
						gazex.append(gazex[-1])
						gazey.append(gazey[-1])
						fixsac.append(0) #treat gaze estimation error as saccade
				else:
					while nframe[-1] + 1 != int(s[5]):
						nframe.append(nframe[-1]+1)
						gazex.append(gazex[-1])
						gazey.append(gazey[-1])
						fixsac.append(fixsac[-1])
					if int(round(float(s[3]))) in range(1280) and int(round(float(s[4]))) in range(960):
						nframe.append(int(s[5]))
						gazex.append(float(s[3]))
						gazey.append(float(s[4]))
						if 'Fix' in s[6]:
							fixsac.append(1)
						else:
							fixsac.append(0)
					else:
						nframe.append(int(s[5]))
						gazex.append(gazex[-1])
						gazey.append(gazey[-1])
						fixsac.append(0)
			else:
				if int(round(float(s[3]))) in range(1280) and int(round(float(s[4]))) in range(960):
					gazex[-1] = (gazex[-1] + float(s[3])) / 2
					gazey[-1] = (gazey[-1] + float(s[4])) / 2
	return gazex, gazey, nframe, fixsac

gazefiles = os.listdir('gtea_gaze/')
gazefiles.sort()
flowfolder = 'gtea_flows/'
imagefolder = 'gtea_images/'
gtfolder = 'gtea_gts/'
sourcefolder = 'gtea_imgflow/'


for num,f in enumerate(gazefiles):
	#if num <= 21:
		#continue
	folder_name = f[:-9] +'/'
	print(folder_name)
	images = os.listdir(sourcefolder + folder_name)
	flowx = [k for k in images if 'flow_x' in k]
	flowy = [k for k in images if 'flow_y' in k]
	ims = [k for k in images if 'img' in k]
	flowx.sort()
	flowy.sort()
	ims.sort()
	gazex, gazey, nframe, fixsac = parsetxt('gtea_gaze/' + f)
	fixsacsave = []
	#remove the first 1000 frames and the last 700 frames
	#if not os.path.exists(flowfolder + folder_name):
		#os.makedirs(flowfolder + folder_name)
	#if not os.path.exists(imagefolder + folder_name):
		#os.makedirs(imagefolder + folder_name)
	#if not os.path.exists(gtfolder + folder_name):
		#os.makedirs(gtfolder + folder_name)
	for i in range(1001, len(nframe)-700):
		#rgb image
		'''
		copyfile(sourcefolder + folder_name + ims[i], imagefolder + folder_name[:-1] + '_' + ims[i])
		#groundtruth
		gazemap = np.zeros((960,1280))
		gazemap[int(round(gazey[i])) - 1][int(round(gazex[i])) - 1] = 1
		gazemap = ndimage.filters.gaussian_filter(gazemap, 70)
		gazemap -= np.min(gazemap)
		gazemap /= np.max(gazemap)
		gazemap *= 255
		gazeim = cv2.resize(gazemap, (224,224), interpolation=cv2.INTER_AREA)
		cv2.imwrite(gtfolder + folder_name[:-1] + '_gt_' + ims[i], gazeim)
		'''
		fixsacsave.append(fixsac[i])
		#flow image is too large to store
		'''
		flowarr = np.zeros((224,224,20))
		for flowi in range(10):
			currflowx = io.imread(sourcefolder + folder_name + flowx[i - flowi])
			currflowy = io.imread(sourcefolder + folder_name + flowy[i - flowi])
			flowarr[:,:,2*flowi] = currflowx
			flowarr[:,:,2*flowi+1] = currflowy
		np.save(flowfolder + folder_name + 'flow_' + ims[i][:-4] + '.npy', flowarr)
		'''
	np.savetxt('fixsac/'+f[:-9]+'.txt', fixsacsave)