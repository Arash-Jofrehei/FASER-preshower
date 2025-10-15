#!/bin/env python3

import uproot
from ROOT import TChain, TFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse
from tqdm import tqdm
from math import tan, sin, cos
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('typing_extensions')
import awkward as ak


# Constants
NUM_BOARDS = 8
NUM_MODULES = 6
NUM_CHIPS = 6
CHIP_WIDTH = 208   #16 * 13  (in reality 17 * 13)
CHIP_HEIGHT = 128  #16 *  8
MODULE_WIDTH = CHIP_WIDTH * 3
MODULE_HEIGHT = CHIP_HEIGHT * 2

pitchH = 97.73 #um
#jump_down_same_row = 55.85 #um
pitchV = 111.7


PS_height = 175000 #um
PS_width = 134000 #um
thetaThr = 5e-4

mid_fid_grow = 768
mid_fid_gcol = 624

scalingV = PS_height / (2*mid_fid_grow*pitchV)
scalingH = PS_width*16 / (2*mid_fid_gcol*pitchH*17)

chipOccThr = -1 # -1 for turning off

pointsX = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipPosX.npy')
pointsY = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/chipPosY.npy')
angles = np.load('/afs/cern.ch/work/a/ajofrehe/FASER/preshower/alignment/angles.npy')



#pixel_horizontal_pitch = 97.73 #um
#jump_down_same_row = 55.85 #um
#vertical_pitch = 111.7

# Create a mask
def mask(xs,ys):
  ms = np.zeros((CHIP_HEIGHT,CHIP_WIDTH))
  skip = False
  #return ms, skip # Fix me
  
  for i in range(np.size(xs)):
    for j in range(np.size(xs[i])):
      if (ms[ys[i][j],xs[i][j]] != -1e-6):
        ms[ys[i][j],xs[i][j]] += 1
      if (ms[ys[i][j],xs[i][j]] > occThr and occThr != -1):
        ms[ys[i][j],xs[i][j]] = -1e-6
  chipOcc = np.sum(ms) / (CHIP_HEIGHT*CHIP_WIDTH)
  if (chipOccThr != -1 and chipOcc > chipOccThr):
    skip = True
  return ms, skip


# Geometry helpers
def get_global_coordinates(module_id, chip_id, x_local, y_local, side, plane):
    offsetIDx = 0
    if side == "left":
        x_offset = 0
        y_offset = module_id * MODULE_HEIGHT
        offsetIDy = (5 - module_id) * 2
    else:
        offsetIDx += 3
        x_offset = MODULE_WIDTH
        y_offset = (5 - module_id) * MODULE_HEIGHT
        offsetIDy = module_id * 2
        
    col = {0: 2, 1: 2, 2: 1, 3: 1, 4: 0, 5: 0}[chip_id]
    row = 0 if chip_id % 2 == 0 else 1
    
    offsetIDx += col
    offsetIDy += (1-row)

    #if row == 0: ###block works for geoS
    #    y_local = CHIP_HEIGHT - 1 - y_local
    #if row == 1:
    #    x_local = CHIP_WIDTH - 1 - x_local

    ########## If geoT converted data:##########
    if row == 1:                              #
        x_local = CHIP_WIDTH - 1 - x_local    #
        y_local = CHIP_HEIGHT - 1 - y_local   #
    ############################################    

    x_global = x_offset + col * CHIP_WIDTH + x_local
    y_global = y_offset + row * CHIP_HEIGHT + y_local
    
    angle = angles[plane][offsetIDy][offsetIDx]
    gaps = x_local//16 + (x_local%16)//8
    gPosX = 1000*pointsX[plane][offsetIDy][offsetIDx] + cos(angle)*(x_local + gaps)*pitchH - sin(angle)*y_local*pitchV
    gPosY = 1000*pointsY[plane][offsetIDy][offsetIDx] + cos(angle)*y_local*pitchV + sin(angle)* (x_local + gaps)*pitchH
    return x_global, y_global, gPosX, gPosY
    


    
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inDir", type=str, required=True, help="input file directory")
    parser.add_argument("--occThr", type=int, required=True, help="maximum occupancy of a pixel to not be hot")
    #parser.add_argument("--runTag", type=str, required=True, help="input file directory")
    args = parser.parse_args()
    
    DATA_DIR = args.inDir
    occThr = args.occThr # -1 to turn off
    #if (args.runTag != None):
      #runTag = args.runTag
    runTag = os.listdir(DATA_DIR)[0][-25:-19]
    fileID = '_'+os.listdir(DATA_DIR)[0][-18:-13]
    print('Run Tag:', runTag, 'file ID:', fileID)
    
    
    chain = TChain("nt")
    chain.Add("/eos/experiment/faser/data0/phys/2025/p001[3-4]/"+runTag+"/*")
    FEntries = chain.GetEntries()
    
    output_dir = '/eos/user/a/ajofrehe/FASER/preshower/rentuplized/'+runTag
    os.makedirs(output_dir, exist_ok=True)
    outFile = uproot.recreate(output_dir+'/hits_'+runTag+fileID+'.root')
    
    xg_ = [] # hit-based gloabal x, one element per hit
    yg_ = []
    gPosX_ = []
    gPosY_ = []
    sc_ = []
    sp_ = []
    row_ = []
    col_ = []
    adc_ = []
    eventID_ = []
    run_ = []
    plane_ = []
    chip_ = []
    module_ = []
    board_ = []
    
    nEvents = 0
    firstEvent = 0
    
    baseFileName = os.listdir(DATA_DIR)[0][:-6]
    
    pbar = tqdm(total=NUM_BOARDS*NUM_MODULES*NUM_CHIPS, unit="chip")
    for board_id in range(NUM_BOARDS):
      
      fname = baseFileName+f'{board_id}'+'.root'
      if not os.path.exists(os.path.join(DATA_DIR, fname)):
        print(f"  No file found for board {board_id}. Skipping.")
        continue
      
      side = "left" if board_id % 2 == 1 else "right"
      plane_id = board_id // 2
        
      
      with uproot.open(os.path.join(DATA_DIR, fname)) as f:
        if (board_id==0):
          eventIDs = []
          runs = []
          Track_X_atPreshower1 = [] 
          Track_X_atPreshower2 = [] 
          Track_Y_atPreshower1 = [] 
          Track_Y_atPreshower2 = []
          Track_ThetaX_atPreshower1 = []
          Track_ThetaY_atPreshower1 = []
          Track_p1 = []
          event_info_tree = f["event_info_tree"]
          all_runs = event_info_tree["run"].array(library="np")
          all_eventIDs = event_info_tree["eventID"].array(library="np")
          firstEvent = all_eventIDs[0]
          PSEntries = all_eventIDs.shape[0]
          Fe = 0  # event index of the FASER ntuple
          PSe = 0 # event index of the PS ntuple
          nEvents = 0 # number of final skimmed events
          chain.GetEntry(Fe)
          while(firstEvent > chain.eventID and Fe < FEntries):
            Fe += FEntries//10
            chain.GetEntry(Fe)
          if (firstEvent < chain.eventID or Fe >= FEntries):
            Fe -= FEntries//10
            chain.GetEntry(Fe)
          while(firstEvent > chain.eventID and Fe < FEntries):
            Fe += FEntries//100
            chain.GetEntry(Fe)
          if (firstEvent < chain.eventID or Fe >= FEntries):
            Fe -= FEntries//100
            chain.GetEntry(Fe)
          while(firstEvent > chain.eventID and Fe < FEntries):
            Fe += 1
            chain.GetEntry(Fe)
          for PSe in range(PSEntries):
            if (all_eventIDs[PSe] == chain.eventID):
              if (chain.Track_X_atPreshower1.size() == 1 and abs(chain.Track_X_atPreshower1[0])<(PS_width/2000.) and abs(chain.Track_Y_atPreshower1[0])<(PS_height/2000.) and chain.Track_p1[0]/1000. > 25 and abs(chain.Track_ThetaX_atPreshower1[0])<thetaThr and abs(chain.Track_ThetaY_atPreshower1[0])<thetaThr):
                eventIDs += [chain.eventID]
                runs += [all_runs[PSe]]
                Track_X_atPreshower1 += [chain.Track_X_atPreshower1[0]] 
                Track_X_atPreshower2 += [chain.Track_X_atPreshower2[0]] 
                Track_Y_atPreshower1 += [chain.Track_Y_atPreshower1[0]] 
                Track_Y_atPreshower2 += [chain.Track_Y_atPreshower2[0]]
                Track_ThetaX_atPreshower1 += [chain.Track_ThetaX_atPreshower1[0]]
                Track_ThetaY_atPreshower1 += [chain.Track_ThetaY_atPreshower1[0]]
                Track_p1 += [chain.Track_p1[0]]
                nEvents += 1
              Fe += 1
              chain.GetEntry(Fe)
          exg = [[] for i in range(nEvents)] # event based global x, one array per event
          eyg = [[] for i in range(nEvents)]
          egPosX = [[] for i in range(nEvents)]
          egPosY = [[] for i in range(nEvents)]
          esc = [[] for i in range(nEvents)]
          esp = [[] for i in range(nEvents)]
          erow = [[] for i in range(nEvents)]
          ecol = [[] for i in range(nEvents)]
          eadc = [[] for i in range(nEvents)]
          eplane = [[] for i in range(nEvents)]
          echip = [[] for i in range(nEvents)]
          emodule = [[] for i in range(nEvents)]
          eboard = [[] for i in range(nEvents)]
        for module_id in range(NUM_MODULES):
          for chip_id in range(NUM_CHIPS):
            pbar.update()
            tree = f[f"module_{module_id}/chip_{chip_id}"]
            eventID = tree["eventID"].array(library="np")
            run = tree["run"].array(library="np")
            sc = tree["SC_id"].array(library="np")
            sp = tree["SP_id"].array(library="np")
            col = tree["PIX_col"].array(library="np")
            row = tree["PIX_row"].array(library="np")
            adc = tree["PIX_adc"].array(library="np")
            xs = sc * 16 + col
            ys = sp * 16 + row
            ms, skip = mask(xs,ys) #look for hot pixels and return the mask. Hot pixels are assigned -1, and the rest is occupancy
            if (skip==True):
              continue
            
            xg, yg, gPosX, gPosY = get_global_coordinates(module_id, chip_id, xs, ys, side, plane_id)
            
            e = 0
            ePast = 0
            for i in range(np.size(eventID)):
              #print ('1',ePast, e, nEvents, len(eventIDs) i, np.size(eventID), eventID[i], eventIDs[e])
              while (eventIDs[e] < eventID[i] and e < nEvents-1):
                e += 1
              if (eventIDs[e] != eventID[i]):
                e = ePast
                continue
              ePast = e
              for j in range(np.size(xs[i])):
                if (ms[ys[i][j],xs[i][j]] != -1e-6):
                  xg_ += [xg[i][j]]
                  yg_ += [yg[i][j]]
                  sc_ += [sc[i][j]]
                  sp_ += [sp[i][j]]
                  row_ += [row[i][j]]
                  col_ += [col[i][j]]
                  adc_ += [adc[i][j]]
                  eventID_ += [eventID[i]]
                  run_ += [run[i]]
                  plane_ += [plane_id]
                  chip_ += [chip_id]
                  module_ += [module_id]
                  board_ += [board_id]
                  PS_hit_H = gPosX[i][j] #um
                  PS_hit_V = gPosY[i][j]
                  #PS_hit_V = (yg[i][j]-mid_fid_grow) * pitchV * scalingV #um
                  #PS_hit_H = (xg[i][j]-mid_fid_gcol) * pitchH * scalingH
                  if (sc[i][j]%2 == 0):
                    if (col[i][j] <= 8 and col[i][j]%2 == 1 or col[i][j] > 8 and col[i][j]%2 == 0):
                      PS_hit_V -= pitchV * scalingV / 2.0
                  if (sc[i][j]%2 == 1):
                    if (col[i][j] <= 8 and col[i][j]%2 == 0 or col[i][j] > 8 and col[i][j]%2 == 1):
                      PS_hit_V -= pitchV * scalingV / 2.0
                  gPosX_ += [PS_hit_H]
                  gPosY_ += [PS_hit_V]
                  
                  exg[e] += [xg[i][j]]
                  eyg[e] += [yg[i][j]]
                  egPosX[e] += [PS_hit_H]
                  egPosY[e] += [PS_hit_V]
                  esc[e] += [sc[i][j]]
                  esp[e] += [sp[i][j]]
                  erow[e] += [row[i][j]]
                  ecol[e] += [col[i][j]]
                  eadc[e] += [adc[i][j]]
                  eplane[e] += [plane_id]
                  echip[e] += [chip_id]
                  emodule[e] += [module_id]
                  eboard[e] += [board_id]
                    
                  
            
    pbar.close()
    xg_ = np.array(xg_)
    yg_ = np.array(yg_)
    gPosX_ = np.array(gPosX_)
    gPosY_ = np.array(gPosY_)
    sc_ = np.array(sc_)
    sp_ = np.array(sp_)
    row_ = np.array(row_)
    col_ = np.array(col_)
    adc_ = np.array(adc_)
    eventID_ = np.array(eventID_)
    run_ = np.array(run_)
    plane_ = np.array(plane_)
    chip_ = np.array(chip_)
    module_ = np.array(module_)
    board_ = np.array(board_)
    outFile["hitTree"] = {"plane":plane_, "chip":chip_, "module":module_, "board":board_, "grow":yg_, "gcol": xg_, "gPosX": gPosX_, "gPosY": gPosY_, "sc": sc_, "sp": sp_, "row": row_, "col": col_, "adc": adc_, "run": run_, "eventID": eventID_}
    
    
    eventIDs = ak.Array(eventIDs)
    runs = ak.Array(runs)
    Track_X_atPreshower1 = ak.Array(Track_X_atPreshower1) 
    Track_X_atPreshower2 = ak.Array(Track_X_atPreshower2)
    Track_Y_atPreshower1 = ak.Array(Track_Y_atPreshower1)
    Track_Y_atPreshower2 = ak.Array(Track_Y_atPreshower2)
    Track_ThetaX_atPreshower1 = ak.Array(Track_ThetaX_atPreshower1)
    Track_ThetaY_atPreshower1 = ak.Array(Track_ThetaY_atPreshower1)
    Track_p1 = ak.Array(Track_p1)
    exg = ak.Array(exg)
    eyg = ak.Array(eyg)
    egPosX = ak.Array(egPosX)
    egPosY = ak.Array(egPosY)
    esc = ak.Array(esc)
    esp = ak.Array(esp)
    erow = ak.Array(erow)
    ecol = ak.Array(ecol)
    eplane = ak.Array(eplane)
    echip = ak.Array(echip)
    emodule = ak.Array(emodule)
    eboard = ak.Array(eboard)
    outFile["eventTree"] = {"plane":eplane, "chip":echip, "module":emodule, "board":eboard, "grow":eyg, "gcol": exg, "gPosX": egPosX, "gPosY": egPosY, "sc": esc, "sp": esp, "row": erow, "col": ecol, "adc": eadc, "run": runs, "eventID": eventIDs, "Track_X_atPreshower1": Track_X_atPreshower1, "Track_X_atPreshower2": Track_X_atPreshower2, "Track_Y_atPreshower1": Track_Y_atPreshower1, "Track_Y_atPreshower2": Track_Y_atPreshower2, "Track_ThetaX_atPreshower1": Track_ThetaX_atPreshower1, "Track_ThetaY_atPreshower1": Track_ThetaY_atPreshower1, "Track_p1": Track_p1}
    
    