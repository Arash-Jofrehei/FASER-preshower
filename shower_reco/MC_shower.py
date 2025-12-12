#!/bin/env python3

from ROOT import TChain, TFile, TCanvas, TProfile2D, TH1D, TH2D, gStyle, gROOT
from tqdm import tqdm
import os
import numpy as np
gROOT.SetBatch(True)
gStyle.SetOptStat(0)
gStyle.SetPalette(57) #kBird
gStyle.SetPadTickX(1)
gStyle.SetPadTickY(1)
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser()
parser.add_argument("--plotDirTag", nargs='?', type=str, help="name of the subdirectory to store plots")
parser.add_argument("--runTag", type=str, required=True, help="input file directory")
args = parser.parse_args()

if (args.plotDirTag != None):
  plotDirTag = args.plotDirTag
else:
  plotDirTag = ''

if (args.runTag != None):
  runTag = args.runTag

layerShiftX = [500,254.5,58,-80] #um
layerShiftY = [-80,-15,138.4,179.7] #um


if (os.path.exists('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipShiftX.npy')):
  chipShiftX = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipShiftX.npy')
  chipShiftY = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipShiftY.npy')
else:
  chipShiftX = np.zeros((4,12,6))
  chipShiftY = np.zeros((4,12,6))

#ADC_thr = [2,9,1,4] # a seed should have ADC greater than or equal to this threshold
#ADC_thr = [1,3,1,1] # a seed should have ADC greater than or equal to this threshold
busyPlaneThr = [20,40,30,20]
#busyHaloThr  = [
#ADC_thr = [[1,1,1,1],[1,2,2,2],[1,3,2,2]] # a seed should have ADC greater than or equal to this threshold
#ADC_thr = [[1,2,1,2],[2,5,2,2]] # a seed should have ADC greater than or equal to this threshold
ADC_thr = [[1,2,1,2],[2,5,2,2]] # a seed should have ADC greater than or equal to this threshold
#ADC_thr = [[1,2,1,2],[1,4,2,2]] # a seed should have ADC greater than or equal to this threshold
maxADC_rad_x = 0
maxADC_rad_y = 0
occ_rad_x = 1 # how many pixels afar in X the hit from another plane can be, to be considered from the same shower (~1 for high energy photons)
occ_rad_y = 1 # how many pixels afar in Y the hit from another plane can be, to be considered from the same shower (~1 for high energy photons)
showerMerge_rad_x = 1 # how many pixels afar in X another shower can be, to merge the showers (~1 for high energy photons)
showerMerge_rad_y = 1 # how many pixels afar in Y another shower can be, to merge the showers (~1 for high energy photons)
maxDist2 = 2
match_rad_x = 1
match_rad_y = 1
minLayersInShower = 3
minNoRes = 3
#allowed_seed_distance = 500 #um
#allowed_cluster_distance = 500 #um
#busy_thr = 1000


pitchH = 97.73 #um
#jump_down_same_row = 55.85 #um
pitchV = 111.7

PS_height = 175000 #um
PS_width = 134000 #um

nrows = 1536
ncols = 1248 # 1326 with one gap every 16 cols

plotDir = '/eos/user/a/ajofrehe/www/FASER/preshower/clustering/'+plotDirTag
os.makedirs(plotDir, exist_ok=True)


#runTag = 'mu-highCalo'
#runTag = 'muCal400GeV'
#runTag = 'muee-conversions'
#runTag = 'muee_3trk'

nExpectedTracks = 1
if (runTag == 'muee_3trk'):
  nExpectedTracks = 3

nExpectedShower = 1
showerPos = [[114,117]]

input_dir = '/eos/user/a/ajofrehe/FASER/preshower/rentuplized/'+runTag
PS_chain = TChain("eventTree")
#PS_chain.Add(input_dir+'/hits_'+runTag+'.root')
if (runTag == 'singlePhoton500GeV'):
  PS_chain.Add('/eos/user/e/ebornand/Allpix/Photon/sim_allpixsquared_WP2233_single_photon_500Gev.root')
if (runTag == 'singlePhoton500GeV_load10'):
  PS_chain.Add('/eos/user/e/ebornand/Allpix/Photon/sim_allpixsquared_WP2233_WP3load10_single_photon_500Gev.root')
if (runTag == 'singlePhoton1000GeV'):
  PS_chain.Add('/eos/user/e/ebornand/Allpix/Photon/sim_allpixsquared_WP2233_single_photon_1000Gev.root')
#PS_chain.Add('/eos/user/e/ebornand/Allpix/Photon/old/sim_allpixsquared_WP2_3_single_photon_500Gev.root')
if (runTag == 'twoPhoton500GeV'):
  PS_chain.Add('/eos/user/e/ebornand/Allpix/Photon/sim_allpixsquared_WP2233_two_photon_500Gev_d0p3mm.root')
  nExpectedShower = 2
  showerPos = [[114,117],[114,115]]

output_dir = '/eos/user/a/ajofrehe/FASER/preshower/clustered/'+runTag
os.makedirs(output_dir, exist_ok=True)
#outFile = uproot.recreate(output_dir+'/showers_'+runTag+'.root')

plotDir = '/eos/user/a/ajofrehe/www/FASER/preshower/shower_reco/'+runTag
os.makedirs(plotDir, exist_ok=True)


pbins = [i for i in range(100,800,100)]+[i for i in range(800,1400,300)]
EoPbins = [0,0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.5,2.5,5]
pbins = np.asarray(pbins, 'd')
EoPbins = np.asarray(EoPbins, 'd')

h_adc_core_halo = [0 for i in range(4)]
for plane in range(4):
  h_adc_core_halo[plane] = TH2D('h_adc_core_halo_plane'+str(plane),'plane '+str(plane)+';adc core; sum adc neighbors',14, 1, 15, 50, 0, 50)
h_adc_core_nHalo = [0 for i in range(4)]
h_adc_core_nHalo_matched = [0 for i in range(4)]
h_adc_core_nHalo_unmatched = [0 for i in range(4)]
h_adc_core_nHalo_missed = [0 for i in range(4)]
for plane in range(4):
  h_adc_core_nHalo[plane] = TH2D('h_adc_core_nHalo_plane'+str(plane),'plane '+str(plane)+';adc core; number of neighbors',14, 1, 15, 8, 0, 8)
  h_adc_core_nHalo_matched[plane] = TH2D('h_adc_core_nHalo_matched_plane'+str(plane),'matched: plane '+str(plane)+';adc core; number of neighbors',14, 1, 15, 8, 0, 8)
  h_adc_core_nHalo_unmatched[plane] = TH2D('h_adc_core_nHalo_unmatched_plane'+str(plane),'unmatched: plane '+str(plane)+';adc core; number of neighbors',14, 1, 15, 8, 0, 8)
  h_adc_core_nHalo_missed[plane] = TH2D('h_adc_core_nHalo_missed_plane'+str(plane),'missed: plane '+str(plane)+';adc core; number of neighbors',14, 1, 15, 8, 0, 8)
h_adc_core_nHits = [0 for i in range(4)]
h_adc_core_nHits_missed = [0 for i in range(4)]
for plane in range(4):
  h_adc_core_nHits[plane] = TH2D('h_adc_core_nHits_plane'+str(plane),'plane '+str(plane)+';adc core; nHits in plane',14, 1, 15, 50, 0, 500)
  h_adc_core_nHits_missed[plane] = TH2D('h_adc_core_nHits_missed_plane'+str(plane),'missed: plane '+str(plane)+';adc core; nHits in plane',14, 1, 15, 50, 0, 500)

h_EoP_p = TH2D('h_EoP_p','single photon: number of events;track p[GeV];E/p',len(pbins)-1,pbins,len(EoPbins)-1,EoPbins)
h_EoP_p.Sumw2()
h_EoP_p.SetLineWidth(3)

h_showerEff_EoP_p = TH2D('h_showerEff_EoP_p','single photon: shower reco efficiency;photon E[GeV];E/p',len(pbins)-1,pbins,len(EoPbins)-1,EoPbins)
h_showerEff_EoP_p.Sumw2()
h_showerEff_EoP_p.SetLineWidth(3)

h_nReco = TH1D('h_nReco','number of reconstructed showers',5,0,5)
h_nReco.Sumw2()
h_nReco.SetLineWidth(3)

h_nMatched = TH1D('h_nMatched','number of matched showers',3,0,3)
h_nMatched.Sumw2()
h_nMatched.SetLineWidth(3)


nEntries = PS_chain.GetEntries()

runs      = [-1 for e in range(nEntries)]
eventIDs  = [-1 for e in range(nEntries)]


pbar = tqdm(total=nEntries, unit="")
for e in range(int(nEntries*1)):
  pbar.update()
  PS_chain.GetEntry(e)
  #runs[e]     = PS_chain.run
  eventIDs[e] = PS_chain.eventID
  if (nExpectedTracks == 3):
    if (PS_chain.Track_p1 < 100000 or PS_chain.Track2_p1 < 100000 or PS_chain.Track3_p1 < 100000):
      continue
  #print(PS_chain.Track_p1/1000, PS_chain.Track2_p1/1000, PS_chain.Track3_p1/1000)
  #if (nExpectedTracks == 1 and PS_chain.Track_p1 < 100000):
  #  continue
  #print(PS_chain.Track_p1/1000, PS_chain.CaloHi_total_E_EM/1000)
  #EoP = PS_chain.CaloHi_total_E_EM / PS_chain.Track_E
  EoP = 1
  h_EoP_p.Fill(PS_chain.Track_E[0], EoP)
  
  #adcMap  = np.zeros((4,nrows,ncols))
  adcMap  = np.zeros((4,1600,1400),dtype=int)
  adcMapRaw  = np.zeros((4,1600,1400),dtype=int)
  rolOc   = np.zeros((4,1600,1400),dtype=bool) # roughly one element per pixel, stores binary occupancy within a radius
  rolOc_sum   = np.zeros((1600,1400),dtype=int) # roughly one element per pixel, stores nPlanes with occupancy within a radius
  noRes_sum   = np.zeros((1600,1400),dtype=int) # roughly one element per pixel, stores nPlanes with occupancy within zero radius
  hitDist   = np.zeros((4,1600,1400),dtype=int) # distance squared wrt the closest hit
  sumDist   = np.zeros((1600,1400),dtype=int) # distance squared wrt the closest hit, summed for 4 planes
  sumDistGE3   = np.zeros((1600,1400),dtype=float) # distance squared wrt the closest hit, summed for 4 planes & divided by nPlanes, only for >=3 planes
  shower_index   = np.zeros((1600,1400),dtype=int) - 1 
  
  raw_shower_list = [] # list of [y_index,x_index] with at least 3 planes
  
  planeOcc = [0 for i in range(4)]
  planeOccHighAdc = [0 for i in range(4)]
  busy = 2
  if (len(PS_chain.plane)<200):
    busy = 1
  if (len(PS_chain.plane)<100):
    busy = 0
  
  for hit in range(len(PS_chain.plane)): # loop on hits to get plance occupancy
    plane = PS_chain.plane[hit]
    planeOcc[plane] += 1
  busyPlane = [0,0,0,0]
  for plane in range(4):
    if (planeOcc[plane]>busyPlaneThr[plane]):
      busyPlane[plane] = 1
  for hit in range(len(PS_chain.plane)): # loop on hits to fill adcMap
    plane = PS_chain.plane[hit]
    adc = PS_chain.adc[hit]
    x_index = int(PS_chain.gPosX[hit]/pitchH) + 700
    y_index = int(PS_chain.gPosY[hit]/pitchV) + 800
    if (adc>0 and adc<15):
      adcMapRaw[plane][y_index][x_index] = adc
    #if(adc<ADC_thr[busy][plane] or adc>15):
    if(adc<ADC_thr[busyPlane[plane]][plane] or adc>15):
      continue
    planeOccHighAdc[plane] += 1
    #print(plane, '->', adc)
    adcMap[plane][y_index][x_index] = adc
  
  nRawShowers = 0
  
  for hit in range(len(PS_chain.plane)): # loop on hits to find core and fill rolling occupancy matrix (rolOc)
    plane = PS_chain.plane[hit]
    #row = PS_chain.grow[hit]
    #col = PS_chain.gcol[hit]
    adc = PS_chain.adc[hit]
    x_index = int(PS_chain.gPosX[hit]/pitchH) + 700
    y_index = int(PS_chain.gPosY[hit]/pitchV) + 800
    isCore = True
    #if(adc<ADC_thr[busy][plane] or adc>15):
    if(adc<ADC_thr[busyPlane[plane]][plane] or adc>15):
      continue
    
    nNeighbor = 0
    for i in range(-1*maxADC_rad_y,maxADC_rad_y+1):
      if (y_index+i < 0 or y_index+i >= 1600):
        continue
      for j in range(-1*maxADC_rad_x,maxADC_rad_x+1):
        if (x_index+j < 0 or x_index+j >= 1400):
          continue
        if (adcMap[plane][y_index+i][x_index+j] > 0):
          nNeighbor += 1
        if (adcMap[plane][y_index+i][x_index+j] > adc):
          isCore = False
    #if(x_index == 114 and y_index==115):
    #  print('plane', plane, rolOc[plane][y_index][x_index], rolOc_sum[y_index][x_index])
    
    #if (nNeighbor>=3 and adc<ADC_thr[1][plane]):
    #  isCore = False
    
    if (isCore):
      #if (rolOc[plane][y_index][x_index] == False):
      noRes_sum[y_index][x_index] += 1
        #print('plane', plane, ' x =', x_index-114, ' y =', y_index-117)
      for i in range(-1*occ_rad_y,occ_rad_y+1):
        if (y_index+i < 0 or y_index+i >= 1600):
          continue
        for j in range(-1*occ_rad_x,occ_rad_x+1):
          if (x_index+j < 0 or x_index+j >= 1400):
            continue
          dist = i**2 + j**2
          if (rolOc[plane][y_index + i][x_index + j] == False):
            rolOc[plane][y_index + i][x_index + j] = True
            rolOc_sum[y_index + i][x_index + j] += 1
            hitDist[plane][y_index + i][x_index + j] = dist
            sumDist[y_index + i][x_index + j] += dist
            if (rolOc_sum[y_index + i][x_index + j] == minLayersInShower):
              raw_shower_list += [[y_index + i,x_index + j]]
              shower_index[y_index + i][x_index + j] = nRawShowers
              nRawShowers +=1
            if (rolOc_sum[y_index + i][x_index + j] >= minLayersInShower):
              sumDistGE3[y_index + i][x_index + j] = sumDist[y_index + i][x_index + j] / rolOc_sum[y_index + i][x_index + j]
              #print(x_index+j-114, y_index+i-117, rolOc_sum[y_index + i][x_index + j], sumDist[y_index + i][x_index + j], sumDistGE3[y_index + i][x_index + j], noRes_sum[y_index + i][x_index + j])
          else:
            if (hitDist[plane][y_index + i][x_index + j] > dist):
              sumDist[y_index + i][x_index + j] += (dist-hitDist[plane][y_index + i][x_index + j])
              hitDist[plane][y_index + i][x_index + j] = dist
              if (rolOc_sum[y_index + i][x_index + j] >= minLayersInShower):
                sumDistGE3[y_index + i][x_index + j] = sumDist[y_index + i][x_index + j] / rolOc_sum[y_index + i][x_index + j]
                #print(x_index+j-114, y_index+i-117, rolOc_sum[y_index + i][x_index + j], sumDist[y_index + i][x_index + j], sumDistGE3[y_index + i][x_index + j], noRes_sum[y_index + i][x_index + j])
    #if(x_index == 114 and y_index==115):
    #  print('plane', plane, rolOc[plane][y_index][x_index], rolOc_sum[y_index][x_index])
  
  
  # choosing the right showers, minimizing the variance
  
  accepted = np.ones((nRawShowers),dtype=int)
  matched = np.zeros((nRawShowers),dtype=int)
  
  for sh in range(nRawShowers):
    y = raw_shower_list[sh][0]
    x = raw_shower_list[sh][1]
    if (noRes_sum[y][x]<minNoRes):
      accepted[sh] = 0
    for i in range(-1*showerMerge_rad_y,showerMerge_rad_y+1):
      if (y+i < 0 or y+i >= 1600):
          continue
      for j in range(-1*showerMerge_rad_x,showerMerge_rad_x+1):
        if (x+j < 0 or x+j >= 1400):
          continue
        if (i==0 and j==0):
          continue
        if (accepted[sh] == 0):
          continue
        sh2 = shower_index[y+i][x+j]
        if (sh2 >= 0):
          if (accepted[sh2] == 0):
            continue
          if (sumDistGE3[y+i][x+j] >= sumDistGE3[y][x]):
            accepted[sh2] = 0
          else:
            accepted[sh] = 0
  
  #print(PS_chain.Track_p1/1000, PS_chain.Track2_p1/1000, PS_chain.Track3_p1/1000)
  #print(1000*PS_chain.Track_X_atPreshower1/pitchH + 700, 1000*PS_chain.Track_Y_atPreshower1/pitchV + 800)
  #print(1000*PS_chain.Track2_X_atPreshower1/pitchH + 700, 1000*PS_chain.Track2_Y_atPreshower1/pitchV + 800)
  #print(1000*PS_chain.Track3_X_atPreshower1/pitchH + 700, 1000*PS_chain.Track3_Y_atPreshower1/pitchV + 800)
  
  #for sh in range(nRawShowers):
  #  if (accepted[sh]):
  #    print(raw_shower_list[sh][1], raw_shower_list[sh][0])
  
  
  nRecoShower = np.sum(accepted)
  h_nReco.Fill(nRecoShower)
  
  for sh in range(nRawShowers):
    if (accepted[sh]):
      for plane in range(4):
        sumNeighbors = 0
        nNeighbors = -1
        coreADC = adcMapRaw[plane][raw_shower_list[sh][0]][raw_shower_list[sh][1]]
        for i in range(-1,2):
          for j in range(-1,2):
            sumNeighbors += adcMapRaw[plane][raw_shower_list[sh][0]+i][raw_shower_list[sh][1]+j]
            if (adcMapRaw[plane][raw_shower_list[sh][0]+i][raw_shower_list[sh][1]+j] > 0):
              nNeighbors += 1
        sumNeighbors -= coreADC
        if (coreADC > 0):
          h_adc_core_halo[plane].Fill(coreADC, sumNeighbors)
          h_adc_core_nHalo[plane].Fill(coreADC, nNeighbors)
          h_adc_core_nHits[plane].Fill(coreADC, planeOcc[plane])
          if ((raw_shower_list[sh][1] == showerPos[0][0] and raw_shower_list[sh][0] == showerPos[0][1]) or (nExpectedShower==2 and raw_shower_list[sh][1] == showerPos[1][0] and raw_shower_list[sh][0] == showerPos[1][1])):
            h_adc_core_nHalo_matched[plane].Fill(coreADC, nNeighbors)
          else:
            h_adc_core_nHalo_unmatched[plane].Fill(coreADC, nNeighbors)
            
      #photonX = float(PS_chain.Track_X_atPreshower1[0])/(1000*pitchH)
      #photonY = float(PS_chain.Track_Y_atPreshower1[0])/(1000*pitchV)
      #if (abs(raw_shower_list[sh][1]-1000*PS_chain.Track_X_atPreshower1[0]/pitchH - 700) < match_rad_x and abs(raw_shower_list[sh][0]-1000*PS_chain.Track_Y_atPreshower1[0]/pitchV - 800) < match_rad_y):
      if ((raw_shower_list[sh][1] == showerPos[0][0] and raw_shower_list[sh][0] == showerPos[0][1]) or (nExpectedShower==2 and raw_shower_list[sh][1] == showerPos[1][0] and raw_shower_list[sh][0] == showerPos[1][1])):
        matched[sh] = 1
      #if (abs(raw_shower_list[sh][1]-1000*PS_chain.Track2_X_atPreshower1/pitchH - 700) < match_rad_x and abs(raw_shower_list[sh][0]-1000*PS_chain.Track2_Y_atPreshower1/pitchV - 800) < match_rad_y):
      #  matched[sh] = 1
      #if (abs(raw_shower_list[sh][1]-1000*PS_chain.Track3_X_atPreshower1/pitchH - 700) < match_rad_x and abs(raw_shower_list[sh][0]-1000*PS_chain.Track3_Y_atPreshower1/pitchV - 800) < match_rad_y):
      #  matched[sh] = 1
      #if (abs(raw_shower_list[sh][1]-photonX) < match_rad_x and abs(raw_shower_list[sh][0]-photonY) < match_rad_y):
      #  matched[sh] = 1
      #print(abs(raw_shower_list[sh][1]-1000*PS_chain.Track_X_atPreshower1/pitchH - 700),abs(raw_shower_list[sh][0]-1000*PS_chain.Track_Y_atPreshower1/pitchV - 800))
      #print(abs(raw_shower_list[sh][1]-1000*PS_chain.Track2_X_atPreshower1/pitchH - 700),abs(raw_shower_list[sh][0]-1000*PS_chain.Track2_Y_atPreshower1/pitchV - 800))
      #print(abs(raw_shower_list[sh][1]-1000*PS_chain.Track3_X_atPreshower1/pitchH - 700),abs(raw_shower_list[sh][0]-1000*PS_chain.Track3_Y_atPreshower1/pitchV - 800))
  
  
  rejected = []
  
  for s in range(nExpectedShower):
    sh = shower_index[showerPos[s][1]][showerPos[s][0]] # check for reconstructed showers at the expected shower spot
    if (sh >= 0):
      if (accepted[sh]):
        continue
      rejected += [sh]
    for plane in range(4):
      sumNeighbors = 0
      nNeighbors = -1
      coreADC = adcMapRaw[plane][showerPos[s][1]][showerPos[s][0]]
      for i in range(-1,2):
        for j in range(-1,2):
          sumNeighbors += adcMapRaw[plane][showerPos[s][1]+i][showerPos[s][0]+j]
          if (adcMapRaw[plane][showerPos[s][1]+i][showerPos[s][0]+j] > 0):
            nNeighbors += 1
      sumNeighbors -= coreADC
      if (coreADC > 0):
        h_adc_core_nHalo_missed[plane].Fill(coreADC, nNeighbors)
        h_adc_core_nHits_missed[plane].Fill(coreADC, planeOcc[plane])
  
  nMatched = np.sum(matched)
  h_nMatched.Fill(nMatched)
  
  if (nRecoShower == 1):
    h_showerEff_EoP_p.Fill(PS_chain.Track_E[0], EoP)
  
  if (nRecoShower<=2):
    continue
  continue
  print(nRecoShower, 'out of', nRawShowers)
  #print('matched:', nMatched)
  print(planeOcc,planeOccHighAdc)
  for sh in range(nRawShowers):
    if (accepted[sh] or sh in rejected):
      print('accepted?', accepted[sh])
      y = raw_shower_list[sh][0]
      x = raw_shower_list[sh][1]
      print(x-114, y-117, noRes_sum[y][x], sumDistGE3[y][x])
      for plane in range(4):
        sumNeighbors = 0
        nNeighbors = 0
        coreADC = adcMapRaw[plane][y][x]
        for i in range(-1,2):
          for j in range(-1,2):
            sumNeighbors += adcMapRaw[plane][y+i][x+j]
            if (adcMapRaw[plane][y+i][x+j]>0 and (i!=0 or j!=0)):
              nNeighbors += 1
        sumNeighbors -= coreADC
        print('plane'+str(plane)+':', coreADC, nNeighbors, sumNeighbors, sumNeighbors/(coreADC+0.00001))
  
  
  #plt.imshow(sumDistGE3-np.multiply(np.array(sumDistGE3,dtype=bool),np.max(sumDistGE3)))
  fig = plt.figure(figsize=(40, 10))
  axarr = fig.subplots(1,4)
  for plane in range(4):
    im = axarr[plane].imshow(adcMapRaw[plane][106:126,104:124])
    divider = make_axes_locatable(axarr[plane])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
  #plt.imshow(rolOc_sum[100:130,100:130])
  #plt.imshow(adcMap[0]/ADC_thr[0]+adcMap[1]/ADC_thr[1]+adcMap[2]/ADC_thr[2]+adcMap[3]/ADC_thr[3])
  #plt.colorbar()  
  plt.show()
    
c_2D = TCanvas("c_2D","showers", 1200, 1200) 
c_2D.SetTopMargin(0.09)
c_2D.SetBottomMargin(0.1)
c_2D.SetLeftMargin(0.1)
c_2D.SetRightMargin(0.14)

h_EoP_p.Draw('colz')
c_2D.SaveAs(plotDir+'/EoP_p.png')
c_2D.SaveAs(plotDir+'/EoP_p.pdf')

h_showerEff_EoP_p.Divide(h_EoP_p)
h_showerEff_EoP_p.Draw('colz')
c_2D.SaveAs(plotDir+'/showerEff_EoP_p.png')
c_2D.SaveAs(plotDir+'/showerEff_EoP_p.pdf')

h_nReco.Scale(1./nEntries)
h_nReco.Draw("he1x0")
c_2D.SaveAs(plotDir+'/nReco.png')
c_2D.SaveAs(plotDir+'/nReco.pdf')

h_nMatched.Scale(1./nEntries)
h_nMatched.Draw("he1x0")
c_2D.SaveAs(plotDir+'/nMatched.png')
c_2D.SaveAs(plotDir+'/nMatched.pdf')

for plane in range(4):
  h_adc_core_halo[plane].Draw('colz')
  c_2D.SaveAs(plotDir+'/adc_core_halo_plane'+str(plane)+'.png')
  c_2D.SaveAs(plotDir+'/adc_core_halo_plane'+str(plane)+'.pdf')
  
  h_adc_core_nHalo[plane].Draw('colz')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_plane'+str(plane)+'.png')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_plane'+str(plane)+'.pdf')
  
  h_adc_core_nHalo_matched[plane].Draw('colz')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_matched_plane'+str(plane)+'.png')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_matched_plane'+str(plane)+'.pdf')
  
  h_adc_core_nHalo_unmatched[plane].Draw('colz')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_unmatched_plane'+str(plane)+'.png')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_unmatched_plane'+str(plane)+'.pdf')
  
  h_adc_core_nHalo_missed[plane].Draw('colz')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_missed_plane'+str(plane)+'.png')
  c_2D.SaveAs(plotDir+'/adc_core_nHalo_missed_plane'+str(plane)+'.pdf')
  
  h_adc_core_nHits[plane].Draw('colz')
  c_2D.SaveAs(plotDir+'/adc_core_nHits_plane'+str(plane)+'.png')
  c_2D.SaveAs(plotDir+'/adc_core_nHits_plane'+str(plane)+'.pdf')
  
  h_adc_core_nHits_missed[plane].Draw('colz')
  c_2D.SaveAs(plotDir+'/adc_core_nHits_missed_plane'+str(plane)+'.png')
  c_2D.SaveAs(plotDir+'/adc_core_nHits_missed_plane'+str(plane)+'.pdf')
  


    
