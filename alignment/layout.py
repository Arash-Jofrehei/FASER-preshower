import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rcParams
from math import atan, sin, cos
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tiltedChips", type=bool, required=True, help="account for chip tilt instead of averaging them out")
args = parser.parse_args()

tiltedChips = args.tiltedChips

rcParams['lines.linewidth'] = 1
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 20

boardName = ['Up','Down']
filenames = []

#canvas = TCanvas('c', 'c', 1000, 1000)

pointsX = np.zeros((4,12,12)) # plane, row (Y), col (X)
pointsY = np.zeros((4,12,12))

#angles = [[] for plane in range(4)]

pointsXtrimmed = np.zeros((4,12,6))
pointsYtrimmed = np.zeros((4,12,6))
angles = np.zeros((4,12,6))

pitchH = 97.73 #um
pitchV = 111.7
chipWidth = pitchH*17*13 /1000. #mm
chipHeight = pitchV*16*8 /1000. #mm
#offsetTopLeft = [0.2343-pitchV/2000.,0.227365-pitchV/2000.] #x,y [mm]
offsetTopLeft = [0.2343-pitchV/2000.,0.746735-pitchV/2000.] #x,y [mm]

plotDir = '/eos/user/a/ajofrehe/www/FASER/preshower/layout/'

for plane in range(4):
  for boardID in range(2):
    fName = 'metrology/Plane'+str(plane+1)+boardName[boardID]+'.txt'
    
    with open(fName) as data:
        lines = data.readlines()
        for modID in range(6):
          moduleLine = modID*14
          for p in range(6): #upper chips of modules
            posX = float(lines[moduleLine+p].split()[8])
            posY = float(lines[moduleLine+p].split()[17])
            pointsX[plane][4*int(modID/2)+(-2*(boardID-1))][p+6*(modID%2)] = posX
            pointsY[plane][4*int(modID/2)+(-2*(boardID-1))][p+6*(modID%2)] = posY
          for p in range(6): #lower chips of modules
            posX = float(lines[moduleLine+11-p].split()[8])
            posY = float(lines[moduleLine+11-p].split()[17])
            pointsX[plane][4*int(modID/2)+1+(-2*(boardID-1))][p+6*(modID%2)] = posX
            pointsY[plane][4*int(modID/2)+1+(-2*(boardID-1))][p+6*(modID%2)] = posY
  
  # Correcting for tilt
  for i in range(12):
    for j in range(6):
      dx = pointsX[plane][i][2*j+1] - pointsX[plane][i][2*j]
      dy = pointsY[plane][i][2*j+1] - pointsY[plane][i][2*j]
      if (tiltedChips):
        angles[plane][i][j] = atan(dy/dx) #rad
  np.save('angles.npy',angles)
  
  fig = plt.figure(figsize=(18,18))
  ax = fig.add_subplot(111)
  plt.hist(1000*angles[plane].ravel(), bins = 50) #mrad
  ax.set_title('plane '+str(plane), fontsize=20)
  plt.xlabel('tilt [mrad]')
  print('average tilt plane '+str(plane)+':',np.mean(angles[plane]))
  plt.savefig(plotDir+'angles_plane'+str(plane)+'.png', bbox_inches='tight')
  
  # average out chip-to-chip tilts
  if (tiltedChips == False):
    for i in range(12):
      m = pointsX[plane,:,i].mean()
      for j in range(12):
        pointsX[plane][j][i] = m
    for i in range(12):
      m = pointsY[plane,i,:].mean()
      for j in range(12):
        pointsY[plane][i][j] = m
  
  
  
  
  ax.clear()
  ax.set_title('plane '+str(plane), fontsize=20)
  plt.xlabel('X [mm]')
  plt.ylabel('Y [mm]')
  plt.plot(pointsX[plane].ravel(),pointsY[plane].ravel(),'+r',markersize=14)
  ax.xaxis.set_major_locator(MultipleLocator(10))
  #ax.xaxis.set_major_formatter('{x:.0f}')
  ax.xaxis.set_minor_locator(MultipleLocator(1))
  ax.yaxis.set_major_locator(MultipleLocator(10))
  ax.yaxis.set_minor_locator(MultipleLocator(1))
  for i in range(12):
    for j in range(6):
      if (i%2==0):
        pointsXtrimmed[plane][i][j] = pointsX[plane][i][2*j] + offsetTopLeft[0]*cos(angles[plane][i][j]) + (chipHeight+offsetTopLeft[1])*sin((angles[plane][i][j]))
        pointsYtrimmed[plane][i][j] = pointsY[plane][i][2*j] - (chipHeight+offsetTopLeft[1])*cos(angles[plane][i][j]) + offsetTopLeft[0]*sin(angles[plane][i][j])
        rect = patches.Rectangle((pointsXtrimmed[plane][i][j], pointsYtrimmed[plane][i][j]), chipWidth, chipHeight, angle=angles[plane][i][j], linewidth=2, edgecolor='b', facecolor='none')
      if (i%2==1):
        pointsXtrimmed[plane][i][j] = pointsX[plane][i][2*j+1] - (offsetTopLeft[0]+chipWidth)*cos(angles[plane][i][j]) - offsetTopLeft[1]*sin((angles[plane][i][j]))
        pointsYtrimmed[plane][i][j] = pointsY[plane][i][2*j+1] + offsetTopLeft[1]*cos(angles[plane][i][j]) - (offsetTopLeft[0]+chipWidth)*sin(angles[plane][i][j])
        rect = patches.Rectangle((pointsXtrimmed[plane][i][j], pointsYtrimmed[plane][i][j]), chipWidth, chipHeight, angle=angles[plane][i][j] ,linewidth=2, edgecolor='g', facecolor='none')
      ax.add_patch(rect)
  plt.savefig(plotDir+'plane'+str(plane)+'.png', bbox_inches='tight')
  np.save('chipPosX.npy',pointsXtrimmed)
  np.save('chipPosY.npy',pointsYtrimmed)
  
  