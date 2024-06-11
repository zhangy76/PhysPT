# -*- coding: utf-8 -*-
"""
"""
import os 
import sys
import json
import time 
import trimesh
import argparse
import cv2 
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets

import torch 

import constants

from models.smpl_phys import SMPL

class Terrain(object):
    def __init__(self, vid_output_path, vis_contact_force=True):
        """
        Initialize the graphics window and mesh surface
        """

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('3d Motion and Forces')
        self.window.setGeometry(1000, 0, 1920, 1080)
        self.window.setCameraPosition(distance=1.5, elevation=1.8)
        self.window.setBackgroundColor(0.8)
        self.window.show()

        zgrid = gl.GLGridItem(color=(255, 255, 255, 226))
        self.window.addItem(zgrid)

        device = torch.device('cpu')
        smpl = SMPL(device=device)
        self.faces = smpl.faces.astype(int)

        data = np.load(vid_output_path)
        # 3D mesh
        self.vertices = data['physics_vertices_world']
        self.N_frames = self.vertices.shape[0]
        meshdata = gl.MeshData(vertexes=self.vertices[0], faces=self.faces)
        meshcolors = (60, 150, 60)
        self.mesh = gl.GLMeshItem(meshdata=meshdata, color=pg.glColor(meshcolors))
        self.window.addItem(self.mesh)

        # contact forces
        self.vis_contact_force = vis_contact_force
        if self.vis_contact_force:
            contact = self.vertices[:-2, smpl.contact_vertexidxall]
            grf = data['pred_grf'][0]
            vis_contact_scale = 200
            contact_end = contact + grf / vis_contact_scale
            self.contact_force = np.stack([contact, contact_end], axis=2)

            self.line_contact = {}
            for i in range(29):
                self.line_contact[str(i)] = gl.GLLinePlotItem(pos = np.zeros((2,3)),
                                     color=pg.glColor((0, 255, 0)),
                                     width = 3,
                                     antialias = True
                                    )
                self.window.addItem(self.line_contact[str(i)])

        # joint torque
        joint_actuation = data['pred_torques'][:, :, constants.q2smpl[6:]-6].reshape([self.N_frames-2, 23, 3])

        joint_actuation_mag = np.sqrt((joint_actuation**2).sum(2))
        joint_actuation_mag_root = joint_actuation_mag[:, :3].mean(axis=1, keepdims=True)
        joint_actuation_mag = np.concatenate([joint_actuation_mag_root, joint_actuation_mag], axis=1)

        vertex_colorweights_data = smpl.weights.detach().numpy()
        vertex_colorweights = np.zeros_like(vertex_colorweights_data)
        vertex_colorweights[np.arange(6890),np.argmax(vertex_colorweights_data,axis=1)] = 1
        self.joint_actuation = np.zeros([self.N_frames-2, 6890, 4])
        for frame_idx in range(self.N_frames-2):
            joint_actuation_color = trimesh.visual.interpolate(joint_actuation_mag[frame_idx], 'plasma') / 255. 
            self.joint_actuation[frame_idx] = vertex_colorweights.copy() @ joint_actuation_color.copy()

        # images
        frame_idx = data['frame_idx']
        with open(vid_output_path.split('_output.npz')[0] + '_CLIFF.json', 'rb') as f:
            frame_info = json.load(f)
        data_imgpath = frame_info['impath']
        anno_imgpath = frame_info['datapath']

        self.frame_paths = []
        for i in frame_idx:
            self.frame_paths.append(os.path.join(anno_imgpath, data_imgpath[i]))

        self.frame_idx = 0

    def update(self):

        step = 1
        print('visualizing motion and forces for frame: %d' % (self.frame_idx*step))
        self.mesh.setMeshData(vertexes=self.vertices[self.frame_idx*step], faces=self.faces, vertexColors=self.joint_actuation[self.frame_idx*step])

        if self.vis_contact_force:
            for i in range(29):
                self.line_contact[str(i)].setData(pos=self.contact_force[self.frame_idx*step][i])

        frame = cv2.imread(self.frame_paths[self.frame_idx*step])
        cv2.imshow('video', frame)
        cv2.waitKey(0)

        self.frame_idx = self.frame_idx + 1
        # time.sleep(1)

    def start(self):
        """
        get the graphics window open and setup
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def animation(self, frametime=1/60):
        """
        calls the update method to run in a loop
        """
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        
        self.start()      

parser = argparse.ArgumentParser()
parser.add_argument('--vid_output_path', type=str, required=True, help='Path to the inferred motion and forces data')

args = parser.parse_args()

if __name__ == '__main__':
    t = Terrain(args.vid_output_path)
    t.animation()
