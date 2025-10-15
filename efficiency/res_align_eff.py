#!/bin/env python3


import ROOT
from ROOT import TH1D, TChain, TCanvas, TProfile2D, TH2D, TH2I
import numpy as np
from time import time
import os
import sys
import math
#from array import array
from tqdm import tqdm
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
#ROOT.gStyle.SetPalette(1,0,1)
ROOT.gStyle.SetPalette(57) #kBird
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run", "-r", action='append', type=str, required=True, help="run number starting with 0 (appendable)")
parser.add_argument("--updateChipShift", nargs='?', type=str, help="recalculate relative chip shifts in X and Y as opposed to using existing ones. If called, shifts will not be applied and only calculated and saved.")
parser.add_argument("--updateChipBlockShift", nargs='?', type=str, help="recalculate chip block (4 chips of 4 layers) shifts in X and Y")
parser.add_argument("--fraction", nargs='?', type=float, help="fraction of events to run")
parser.add_argument("--plotDirTag", nargs='?', type=str, help="name of the subdirectory to store plots")
args = parser.parse_args()

if (args.fraction != None):
  fraction = args.fraction
else:
  fraction = 1.0

if (args.plotDirTag != None):
  plotDirTag = '/'+args.plotDirTag
else:
  plotDirTag = ''


if (args.updateChipShift != None):
  updateChipShift = True
  print('updateChipShift: True')
else:
  updateChipShift = False

if (args.updateChipBlockShift != None):
  updateChipBlockShift = True
  print('updateChipBlockShift: True')
else:
  updateChipBlockShift = False

Run = args.run


layerShiftX = [500,254.5,58,-80] #um
layerShiftY = [-80,-15,138.4,179.7] #um

slabSizeV = 400 #um
slabSizeH = 2000 #um
ProbeConeSize = 400 #um
thetaThr = 5e-4


if (os.path.exists('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipShiftX.npy')):
  chipShiftX = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipShiftX.npy')
  chipShiftY = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipShiftY.npy')
else:
  chipShiftX = np.zeros((4,12,6))
  chipShiftY = np.zeros((4,12,6))

chipShiftX_temp = np.zeros((4,12,6))
chipShiftY_temp = np.zeros((4,12,6))
chipOcc = np.zeros((5,12,6))

if (os.path.exists('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipBlockShiftX.npy')):
  chipBlockShiftX = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipBlockShiftX.npy')
  chipBlockShiftY = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipBlockShiftY.npy')
else:
  chipBlockShiftX = np.zeros((12,6))
  chipBlockShiftY = np.zeros((12,6))

pitchH = 97.73 #um
#jump_down_same_row = 55.85 #um
pitchV = 111.7


PS_height = 175000 #um
PS_width = 134000 #um


mid_fid_grow = 768
height_fid_grow = 128
mid_fid_gcol = 624
width_fid_gcol = 208

#scalingV = PS_height / (2*mid_fid_grow*pitchV)
#scalingH = PS_width*16 / (2*mid_fid_gcol*pitchH*17)

#print('scaling vertical:', scalingV, 'horizontal:', scalingH)


if (len(Run)==1):
  plotDir = '/eos/user/a/ajofrehe/www/FASER/preshower/efficiency_residual/'+Run[0]+plotDirTag
else:
  plotDir = '/eos/user/a/ajofrehe/www/FASER/preshower/efficiency_residual/multipleRuns'+plotDirTag
os.makedirs(plotDir, exist_ok=True)

PS_chain = TChain("eventTree")
for r in Run:
  PS_chain.Add('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/efficiency/merged_PS_'+r+'.root')

nEntries = PS_chain.GetEntries()
nEntries = int(nEntries*fraction)
#print(nEntries)


muon_count = 0
eff_labels = ['1st layer', '2nd layer', '3rd layer', '4th layer', '1st & 2nd layers', '>= 2 layers', '>= 3 layers', 'probe 2nd layer', 'probe 3rd layer', 'probe 4th layer']
h_mu_eff = TH1D("h_mu_eff","#mu efficiency in PS",len(eff_labels),0,len(eff_labels))
for i in range(len(eff_labels)):
  h_mu_eff.GetXaxis().SetBinLabel(i+1,eff_labels[i])

pbins = [i for i in range(25,200,50)]+[i for i in range(200,400,50)]+[i for i in range(400,1000,100)]+[i for i in range(1000,1400,200)]+[i for i in range(1400,2700,400)]
pbins = np.asarray(pbins, 'd')
h_eff_TnP_layer1_p = TH1D("h_eff_TnP_layer1_p","#mu efficiency in 2nd PS layer;track p [GeV]",len(pbins)-1,pbins)
h_layer0_p = TH1D("h_layer0_p",";p [GeV];number of tag #mu in 1st layer",len(pbins)-1,pbins)
h_inclusive_p = TH1D("h_inclusive_p",";p [GeV];number of #mu in tracker",len(pbins)-1,pbins)
h_resX_track = [TH1D('h_resX_track_layer'+str(p),'layer '+str(p)+';hit X - track X [mm]',41,-2050,2050) for p in range(5)]
h_resY_track = [TH1D('h_resY_track_layer'+str(p),'layer '+str(p)+';hit Y - track Y [mm]',41,-410,410) for p in range(5)]
h_resX_track[4].SetTitle('all layers after inter-alignment')
h_resY_track[4].SetTitle('all layers after inter-alignment')
for p in range(5):
  h_resX_track[p].SetLineWidth(2)
  h_resY_track[p].SetLineWidth(2)
h_TnP_resX = [0 for i in range(4)]
h_TnP_resY = [0 for i in range(4)]
h_TnP_resX[0] = TH1D("h_TnP_resX_layer0",";1st layer hit X - 2nd layer hit X [#mum]",61,-305,305)
h_TnP_resY[0] = TH1D("h_TnP_resY_layer0",";1st layer hit Y - 2nd layer hit Y [#mum]",61,-305,305)
h_TnP_resX[2] = TH1D("h_TnP_resX_layer2",";3rd layer hit X - 2nd layer hit X [#mum]",41,-512.5,512.5)
h_TnP_resY[2] = TH1D("h_TnP_resY_layer2",";3rd layer hit Y - 2nd layer hit Y [#mum]",41,-512.5,512.5)
h_TnP_resX[3] = TH1D("h_TnP_resX_layer3",";4th layer hit X - 2nd layer hit X [#mum]",41,-512.5,512.5)
h_TnP_resY[3] = TH1D("h_TnP_resY_layer3",";4th layer hit Y - 2nd layer hit Y [#mum]",41,-512.5,512.5)
for p in range(5):
  h_resX_track[p].SetLineWidth(2)
  h_resY_track[p].SetLineWidth(2)
  if (p == 0 or p==2 or p==3):
    h_TnP_resX[p].SetLineWidth(2)
    h_TnP_resY[p].SetLineWidth(2)

h_resX_2Dmap = [TProfile2D('h_resX_2Dmap_plane'+str(p),'plane'+str(p)+': average hitX - trackX [#mum];chip index X;chip index Y',6,0,6,12,0,12) for p in range(5)]
h_resY_2Dmap = [TProfile2D('h_resY_2Dmap_plane'+str(p),'plane'+str(p)+': average hitY - trackY [#mum];chip index X;chip index Y',6,0,6,12,0,12) for p in range(5)]
h_resX_2Dmap[4].SetTitle('all layers: average hitX - trackX [#mum]')
h_resY_2Dmap[4].SetTitle('all layers: average hitY - trackY [#mum]')

h_shiftX_2Dmap = [TH2D('h_shiftX_2Dmap_plane'+str(p),'plane'+str(p)+': shift in X [#mum];chip index X;chip index Y',6,0,6,12,0,12) for p in range(5)]
h_shiftY_2Dmap = [TH2D('h_shiftY_2Dmap_plane'+str(p),'plane'+str(p)+': shift in Y [#mum];chip index X;chip index Y',6,0,6,12,0,12) for p in range(5)]
h_shiftX_2Dmap[4].SetTitle('all layers: shift in X [#mum]')
h_shiftY_2Dmap[4].SetTitle('all layers: shift in Y [#mum]')

h_chip_occ = [TH2I('h_chip_occ_plane'+str(p),'plane'+str(p)+': chip occupancy;chip index X;chip index Y',6,0,6,12,0,12) for p in range(5)]
h_chip_occ[4].SetTitle('all layers: chip occupancy')


h_occ_layer1_frac = TH1D("h_occ_layer1_frac","plane 1 occupancy;fraction of run",20,0,1)

h_test = TH2D('h_test','',64,-160,160,64,-160,160)
h_layer0_p.Sumw2()
h_layer0_p.SetLineWidth(3)
h_eff_TnP_layer1_p.Sumw2()
h_eff_TnP_layer1_p.SetLineWidth(3)
h_inclusive_p.Sumw2()
h_inclusive_p.SetLineWidth(3)


pbar = tqdm(total=nEntries, unit="")
for e in range(nEntries):
  pbar.update()
  PS_chain.GetEntry(e)
  
  #fiducial phase space selection
  #if (abs(PS_chain.Track_X_atPreshower1)>(PS_width/2000.) or abs(PS_chain.Track_Y_atPreshower1)>(PS_height/2000.) or PS_chain.Track_p1/1000. < 25 or abs(PS_chain.Track_ThetaX_atPreshower1)>thetaThr or abs(PS_chain.Track_ThetaY_atPreshower1)>thetaThr):
    #continue
  
  muon_count += 1
  h_inclusive_p.Fill(PS_chain.Track_p1/1000.)
  
  hitPos = [[]for i in range(4)]
  hitID = [[]for i in range(4)]
  layer_counter = [0 for i in range(4)]
  probe_counter = [0 for i in range(4)]
  for hit in range(len(PS_chain.grow)): # loop on hits
    
    
    #if (PS_chain.chip[hit] != 0 or PS_chain.board[hit]%2 != 1 or PS_chain.module[hit] != 0):
    #  continue
    
    
    #PS_hit_V = (PS_chain.grow[hit]-mid_fid_grow) * pitchV * scalingV #um
    #PS_hit_H = (PS_chain.gcol[hit]-mid_fid_gcol) * pitchH * scalingH
    PS_hit_V = PS_chain.gPosY[hit] - layerShiftY[PS_chain.plane[hit]] - chipShiftY[PS_chain.plane[hit]][PS_chain.grow[hit]//128][PS_chain.gcol[hit]//208] - chipBlockShiftY[PS_chain.grow[hit]//128][PS_chain.gcol[hit]//208] #um
    PS_hit_H = PS_chain.gPosX[hit] - layerShiftX[PS_chain.plane[hit]] - chipShiftX[PS_chain.plane[hit]][PS_chain.grow[hit]//128][PS_chain.gcol[hit]//208] - chipBlockShiftX[PS_chain.grow[hit]//128][PS_chain.gcol[hit]//208] #um
      #print(PS_hit_H, chipBlockShiftX[PS_chain.grow[hit]//128][PS_chain.gcol[hit]//208])
    if (abs( PS_hit_V -  1000*PS_chain.Track_Y_atPreshower1 )<slabSizeV and abs( PS_hit_H -  1000*PS_chain.Track_X_atPreshower1 )<slabSizeH):
      #h_mu_eff.Fill(PS_chain.plane[hit])
      layer_counter[PS_chain.plane[hit]] += 1
      hitPos[PS_chain.plane[hit]] = [PS_hit_V, PS_hit_H]
      hitID[PS_chain.plane[hit]] = [PS_chain.grow[hit]//128, PS_chain.gcol[hit]//208]
      if (PS_chain.plane[hit] == 1):
        tempPos = [PS_chain.grow[hit],PS_chain.gcol[hit]]
  
  if (layer_counter[0] == 1):
    for l in range(1,4):
      if (layer_counter[l] == 1):
        resX = hitPos[l][1] -  hitPos[0][1]
        resY = hitPos[l][0] -  hitPos[0][0]
        if (abs(resX)<ProbeConeSize and abs(resY)<ProbeConeSize):
          probe_counter[l] += 1
  
  for l in range(4):
    if (layer_counter[l] == 1):
      h_chip_occ[l].Fill(hitID[l][1],hitID[l][0])
      h_chip_occ[4].Fill(hitID[l][1],hitID[l][0])
  
  if (layer_counter[1] == 1):
    for l in range(4):
      if (l==1):
        continue
      if (layer_counter[l] == 1):
        resX = hitPos[l][1] -  hitPos[1][1]
        resY = hitPos[l][0] -  hitPos[1][0]
        if (abs(resX)<ProbeConeSize and abs(resY)<ProbeConeSize):
          h_TnP_resX[l].Fill(resX)
          h_TnP_resY[l].Fill(resY)
          chipOcc[l][hitID[l][0]][hitID[l][1]] += 1
          chipOcc[4][hitID[l][0]][hitID[l][1]] += 1
          if (updateChipShift):
            chipShiftX_temp[l][hitID[l][0]][hitID[l][1]] += resX
            chipShiftY_temp[l][hitID[l][0]][hitID[l][1]] += resY
        if (l == 0):
          h_test.Fill(resX,resY)
  
      
  
  if (layer_counter[0]>=1):
    h_mu_eff.Fill(0)
    h_layer0_p.Fill(PS_chain.Track_p1/1000.)
  if (layer_counter[1]==1):
    h_mu_eff.Fill(1)
    h_occ_layer1_frac.Fill(e/nEntries)
    #h_occ_layer1_frac.Fill(e/nEntries,PS_chain.nadc)
    #h_occ_layer1_frac.Fill(PS_chain.eventID/17000000)
  if (layer_counter[2]==1):
    h_mu_eff.Fill(2)
  if (layer_counter[3]==1):
    h_mu_eff.Fill(3)
  if (layer_counter[0]==1 and layer_counter[1]==1):
    h_mu_eff.Fill(4)
  if (sum(layer_counter)>=2):
    h_mu_eff.Fill(5)
  if (sum(layer_counter)>=3):
    h_mu_eff.Fill(6)
  if (layer_counter[0]==1 and probe_counter[1]==1):
    h_mu_eff.Fill(7)
    h_eff_TnP_layer1_p.Fill(PS_chain.Track_p1/1000.)
  if (layer_counter[0]==1 and probe_counter[2]==1):
    h_mu_eff.Fill(8)
  if (layer_counter[0]==1 and probe_counter[3]==1):
    h_mu_eff.Fill(9)
  
  
  for p in range(4):
    if (layer_counter[p]==1):
      h_resX_track[p].Fill(hitPos[p][1]-1000*PS_chain.Track_X_atPreshower1)
      h_resY_track[p].Fill(hitPos[p][0]-1000*PS_chain.Track_Y_atPreshower1)
      h_resX_track[4].Fill(hitPos[p][1]-1000*PS_chain.Track_X_atPreshower1)
      h_resY_track[4].Fill(hitPos[p][0]-1000*PS_chain.Track_Y_atPreshower1)
    
  for p in range(4):
    if (layer_counter[p]==1):
      #h_resX_2Dmap[p].Fill(hitPos[p][1]/1000,hitPos[p][0]/1000,hitPos[p][1]-1000*PS_chain.Track_X_atPreshower1)
      #h_resY_2Dmap[p].Fill(hitPos[p][1]/1000,hitPos[p][0]/1000,hitPos[p][0]-1000*PS_chain.Track_Y_atPreshower1)
      track_x = 1000*PS_chain.Track_X_atPreshower1 #um
      track_y = 1000*PS_chain.Track_Y_atPreshower1
      if (p>1):
        track_x = 1000*PS_chain.Track_X_atPreshower2
        track_y = 1000*PS_chain.Track_Y_atPreshower2
      h_resX_2Dmap[p].Fill(hitID[p][1]+0.5,hitID[p][0]+0.5,hitPos[p][1]-track_x)
      h_resY_2Dmap[p].Fill(hitID[p][1]+0.5,hitID[p][0]+0.5,hitPos[p][0]-track_y)
      h_resX_2Dmap[4].Fill(hitID[p][1]+0.5,hitID[p][0]+0.5,hitPos[p][1]-track_x)
      h_resY_2Dmap[4].Fill(hitID[p][1]+0.5,hitID[p][0]+0.5,hitPos[p][0]-track_y)

h_mu_eff.Scale(1./muon_count)
h_eff_TnP_layer1_p.Divide(h_layer0_p)
pbar.close()

if (updateChipShift):
  for p in range(4):
    for i in range(12):
      for j in range(6):
        #chipShiftX_temp[p][i][j] = h_resX_2Dmap[p].GetBinContent(j+1,i+1)# - h_resX_2Dmap[1].GetBinContent(j+1,i+1)
        #chipShiftY_temp[p][i][j] = h_resY_2Dmap[p].GetBinContent(j+1,i+1)# - h_resY_2Dmap[1].GetBinContent(j+1,i+1)
        occ = chipOcc[p][i][j]
        if (occ != 0):
          chipShiftX_temp[p][i][j] /= occ
          chipShiftY_temp[p][i][j] /= occ
  #print(chipShiftX_temp[3])
  #print(chipShiftY[0])
  chipShiftX += chipShiftX_temp
  chipShiftY += chipShiftY_temp
  np.save('../alignment/chipShiftX.npy',chipShiftX)
  np.save('../alignment/chipShiftY.npy',chipShiftY)


if(updateChipBlockShift):
  for i in range(12):
    for j in range(6):
      chipBlockShiftX[i][j] += h_resX_2Dmap[4].GetBinContent(j+1,i+1)
      chipBlockShiftY[i][j] += h_resY_2Dmap[4].GetBinContent(j+1,i+1)
  np.save('../alignment/chipBlockShiftX.npy',chipBlockShiftX)
  np.save('../alignment/chipBlockShiftY.npy',chipBlockShiftY)

for p in range(4):
  for i in range(12):
    for j in range(6):
      h_shiftX_2Dmap[p].SetBinContent(j+1,i+1,chipShiftX[p][i][j]+chipBlockShiftX[i][j])
      h_shiftY_2Dmap[p].SetBinContent(j+1,i+1,chipShiftY[p][i][j]+chipBlockShiftY[i][j])
for i in range(12):
  for j in range(6):
    h_shiftX_2Dmap[4].SetBinContent(j+1,i+1,chipBlockShiftX[i][j])
    h_shiftY_2Dmap[4].SetBinContent(j+1,i+1,chipBlockShiftY[i][j])

#for p in range(5):
#  for i in range(12):
#    for j in range(6):
#      h_chip_occ[p].SetBinContent(j+1,i+1,chipOcc[p][i][j])

print('nMuons in fiducial area:', muon_count)


for i in range(len(eff_labels)-3):
  print(eff_labels[i]+':',100*h_mu_eff.GetBinContent(i+1),'%')
for i in range(len(eff_labels)-3, len(eff_labels)):
  print(eff_labels[i]+':',100*h_mu_eff.GetBinContent(i+1)/h_mu_eff.GetBinContent(1),'%')

c_2D = TCanvas("c_2D","muon efficiency", 1200, 1200) 
c_2D.SetTopMargin(0.09)
c_2D.SetBottomMargin(0.1)
c_2D.SetLeftMargin(0.1)
c_2D.SetRightMargin(0.14)

h_mu_eff.SetMinimum(0)
h_mu_eff.Draw("he1x0")
c_2D.SaveAs(plotDir+'/mu_eff.png')

#h_eff_TnP_layer1_p.SetMinimum(0)
h_eff_TnP_layer1_p.Draw("he1x0")
c_2D.SaveAs(plotDir+'/eff_TnP_layer1_p.png')

#h_inclusive_p.SetMinimum(0)
h_inclusive_p.Draw("he1x0")
c_2D.SaveAs(plotDir+'/mu_inclusive_p.png')

h_layer0_p.Divide(h_inclusive_p)
h_layer0_p.Draw("he1x0")
c_2D.SaveAs(plotDir+'/mu_eff_layer0_p.png')

h_occ_layer1_frac.Draw("he1x0")
c_2D.SaveAs(plotDir+'/occ_layer1_frac.png')

for p in range(4):
  h_resX_track[p].Draw("he1x0")
  c_2D.SaveAs(plotDir+'/resX_track_layer'+str(p)+'.png')
  h_resY_track[p].Draw("he1x0")
  c_2D.SaveAs(plotDir+'/resY_track_layer'+str(p)+'.png')

h_resX_track[4].Draw("he1x0")
c_2D.SaveAs(plotDir+'/resX_track_all_layers.png')
h_resY_track[4].Draw("he1x0")
c_2D.SaveAs(plotDir+'/resY_track_all_layers.png')

for p in range(4):
  h_resX_2Dmap[p].Draw("colz")
  c_2D.SaveAs(plotDir+'/resX_2Dmap_layer'+str(p)+'.png')
  h_resY_2Dmap[p].Draw("colz")
  c_2D.SaveAs(plotDir+'/resY_2Dmap_layer'+str(p)+'.png')

h_resX_2Dmap[4].Draw("colz")
c_2D.SaveAs(plotDir+'/resX_2Dmap_all_layer.png')
h_resY_2Dmap[4].Draw("colz")
c_2D.SaveAs(plotDir+'/resY_2Dmap_all_layer.png')

for p in range(4):
  h_shiftX_2Dmap[p].Draw("colz")
  c_2D.SaveAs(plotDir+'/shiftX_2Dmap_layer'+str(p)+'.png')
  h_shiftY_2Dmap[p].Draw("colz")
  c_2D.SaveAs(plotDir+'/shiftY_2Dmap_layer'+str(p)+'.png')

h_shiftX_2Dmap[4].Draw("colz")
c_2D.SaveAs(plotDir+'/shiftX_2Dmap_all_layer.png')
h_shiftY_2Dmap[4].Draw("colz")
c_2D.SaveAs(plotDir+'/shiftY_2Dmap_all_layer.png')

for p in range(4):
  h_chip_occ[p].Draw("colz")
  c_2D.SaveAs(plotDir+'/chip_occ_layer'+str(p)+'.png')
h_chip_occ[4].Draw("colz")
c_2D.SaveAs(plotDir+'/chip_occ_all_layer.png')

h_test.Draw("colz")
c_2D.SaveAs('plots/test.pdf')

for l in range(4):
  if (l==1):
    continue
  h_TnP_resX[l].Draw("he1x0")
  c_2D.SaveAs(plotDir+'/TnP_resX_layer'+str(l)+'.png')
  h_TnP_resY[l].Draw("he1x0")
  c_2D.SaveAs(plotDir+'/TnP_resY_layer'+str(l)+'.png')

for l in range(4):
  if (l==1):
    continue
  print('overall shift of plane', l, 'wrt plane 1:')
  print('X:', layerShiftX[l]-layerShiftX[1]+h_TnP_resX[l].GetMean(),'+-',h_TnP_resX[l].GetMeanError())
  print('Y:', layerShiftY[l]-layerShiftY[1]+h_TnP_resY[l].GetMean(),'+-',h_TnP_resY[l].GetMeanError())
  print('plane', l, 'rmsX:', h_TnP_resX[l].GetRMS(), 'rmsY:', h_TnP_resY[l].GetRMS())


