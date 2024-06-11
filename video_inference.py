import os
import json
import argparse
from tqdm import tqdm

import numpy as np

import torch

import config
import constants

from models import PhysPT
from models import GlobalTrajPredictor
from models.smpl_phys import SMPL

from assets.utils import rotmat2eulerSMPL, rot6d_to_rotmat, batch_roteulerSMPL

def video_inference(PhysPT, GlobalTrajPredictor, vid_processed_path, batch_size, device):

    # set SMPL model
    smpl = SMPL(device=device)

    # read estimated pose and shape data
    with open(vid_processed_path) as f:
        vid_processed_data = json.load(f)

    vid_start = np.array(vid_processed_data['vid_start'])
    with torch.no_grad():
        for vid_i, vid_i_start in enumerate(tqdm(vid_start, desc='PhysPT Eval', total=(len(vid_start)-1))):
            if vid_i == len(vid_start) - 1:
                break
            vid_i_end = vid_start[vid_i + 1]
            if (vid_i_end - vid_i_start) < config.seqlen:
                print('the video length is too short, ignored')
                continue

            kinematics_deltatrans_list = []
            physics_deltatrans_list = []
            kinematics_vertices_list = []
            physics_vertices_list = []

            pred_springmass_list = []
            pred_torques_list = []
            pred_grf_list = []



            frame_idx_list = []
            num_frames = 0
            # inference on sequence starting from vid_i_start
            for frame_idx in range(vid_i_start, vid_i_end - config.seqlen, batch_size):
                print('frame:%d start:%d end:%d' % (frame_idx, vid_i_start, vid_i_end))
                batch = {}
                batch['pred_beta'] = []
                batch['pred_pose'] = []
                curr_batch_size = min(batch_size, vid_i_end - frame_idx - config.seqlen)
                # prepare the pose and shape estimate with inference step 1
                for b in range(curr_batch_size):
                    batch['pred_beta'].append(torch.from_numpy(
                            np.array(vid_processed_data['pred_beta'][frame_idx + b:frame_idx + b + config.seqlen])).float().to(
                            device).unsqueeze(0))
                    batch['pred_pose'].append(torch.from_numpy(
                            np.array(vid_processed_data['pred_pose'][frame_idx + b:frame_idx + b + config.seqlen])).float().to(
                            device).unsqueeze(0))

                pred_beta = torch.cat(batch['pred_beta'], dim=0).type(torch.float32)
                pred_beta = pred_beta.mean(dim=1, keepdims=True).expand(-1, config.seqlen, -1)  # N*T, 10
                pred_pose = torch.cat(batch['pred_pose'], dim=0).type(torch.float32)

                pred_beta_t = pred_beta.reshape([-1, 10])  # N*T, 10
                pred_pose_t = pred_pose.view([-1, 144])  # N*T, 144

                pred_rotmat = rot6d_to_rotmat(pred_pose_t).reshape([-1, 24, 3, 3])  # N*T, 24, 3, 3
                cam_R = torch.zeros([16 * curr_batch_size, 3, 3]).type(torch.float32).to(device)
                cam_R[:, 0, 0], cam_R[:, 1, 2], cam_R[:, 2, 1] = 1, -1, 1
                pred_rotmat[:, 0] = cam_R.clone().transpose(1, 2) @ pred_rotmat[:, 0]  # N*T, 24, 3, 3

                pred_output_smpl = smpl.forward(betas=pred_beta_t, rotmat=pred_rotmat)
                pred_joints_smpl_aligned = pred_output_smpl.joints_smpl - pred_output_smpl.joints_smpl[:, :1]

                # global trajector estimation
                _, _, pred_R, pred_delta_trans = GlobalTrajPredictor.forward(
                        pred_joints_smpl_aligned.reshape([-1, config.seqlen, 24, 3]))
                pred_T = GlobalTrajPredictor.forward_T(pred_delta_trans.clone(), config.seqlen)

                # kinematics-based prediction
                kinematics_rotmat = pred_rotmat.clone()
                kinematics_rotmat[:, 0] = pred_R.transpose(1, 2) @ kinematics_rotmat[:, 0]
                pred_pose6d = kinematics_rotmat[:, :, :, :2].reshape([-1, config.seqlen, 144])

                kinematics_q = torch.cat([pred_T, pred_pose6d], dim=2)  # N, T, 75
                kinematics_q[:, :, :2] = kinematics_q[:, :, :2] - kinematics_q[:, :1, :2]

                kinematics_output_smpl = smpl.forward(betas=pred_beta_t, rotmat=kinematics_rotmat)
                kinematics_joints = kinematics_output_smpl.joints[:, :17].detach().cpu().numpy()
                kinematics_joints_smpl = kinematics_output_smpl.joints_smpl.detach().cpu().numpy()
                kinematics_vertices = kinematics_output_smpl.vertices.detach().cpu().numpy()

                kinematics_dynamicinput = torch.cat([
                        kinematics_q, kinematics_output_smpl.joints_smpl.reshape([-1, config.seqlen, 72]),
                        kinematics_output_smpl.joints[:, constants.target].reshape([-1, config.seqlen, 60])
                        ], dim=2)

                # physics-based prediction
                physics_q, pred_springmass = PhysPT.forward(kinematics_dynamicinput.transpose(0, 1),
                                                            kinematics_dynamicinput.transpose(0, 1),
                                                            pred_beta_t[::config.seqlen], None, None, None, smpl)

                physics_pose6d = physics_q[:, :, 3:].clone()
                physics_trans = physics_q[:, :, :3].clone()
                pred_delta_trans_physics = physics_trans.clone()
                pred_delta_trans_physics[:, 1:, :2] = pred_delta_trans_physics[:, 1:, :2] - pred_delta_trans_physics[:, :-1, :2]

                physics_rotmat = rot6d_to_rotmat(physics_pose6d.reshape([-1, 144])).reshape([-1, 24, 3, 3])
                physics_output_smpl = smpl.forward(betas=pred_beta_t, rotmat=physics_rotmat)
                physics_joints = physics_output_smpl.joints[:, :17].detach().cpu().numpy()
                physics_joints_smpl = physics_output_smpl.joints_smpl.detach().cpu().numpy()
                physics_vertices = physics_output_smpl.vertices.detach().cpu().numpy()

                physics_pose = rotmat2eulerSMPL(physics_rotmat).reshape([-1, config.seqlen, 72])
                _, physics_rotmat_individual = batch_roteulerSMPL(physics_pose.reshape([-1, 72]))

                for b in range(curr_batch_size):
                    if (frame_idx + b) == vid_i_start:
                        idx_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
                        num_frames = num_frames + 9
                    elif curr_batch_size < batch_size and b == (curr_batch_size - 1):
                        idx_list = b * 16 + np.array([8, 9, 10, 11, 12, 13, 14, 15])
                        num_frames = num_frames + 8
                    else:
                        idx_list = b * 16 + np.array([8])
                        num_frames = num_frames + 1

                    kinematics_vertices_list.append(kinematics_vertices[idx_list])
                    physics_vertices_list.append(physics_vertices[idx_list])

                    kinematics_deltatrans_list.append(pred_delta_trans[idx_list].detach().cpu().numpy())
                    physics_deltatrans_list.append(pred_delta_trans_physics[b, idx_list-b*16].detach().cpu().numpy())

                    pred_springmass_list.append(pred_springmass[b, idx_list - b * 16])

            kinematics_vertices_list = np.concatenate(kinematics_vertices_list, axis=0)
            physics_vertices_list = np.concatenate(physics_vertices_list, axis=0)

            kinematics_deltatrans_list = np.concatenate(kinematics_deltatrans_list, axis=0)
            physics_deltatrans_list = np.concatenate(physics_deltatrans_list, axis=0)

            pred_springmass = torch.cat(pred_springmass_list, dim=0).unsqueeze(0)

            ## root position and motion in world frame
            kinematics_T = kinematics_deltatrans_list[:, None, :]
            kinematics_T[:, :, :2] = np.cumsum(kinematics_T[:, :, :2], axis=0)
            physics_T = physics_deltatrans_list[:, None, :]
            physics_T[:, :, :2] = np.cumsum(physics_T[:, :, :2], axis=0)

            kinematics_vertices_world = kinematics_vertices_list + kinematics_T
            physics_vertices_world = physics_vertices_list + physics_T

            ## forces
            # joint torques
            pred_torques = pred_springmass[:, :-2, 87:156]
            # root residual
            pred_residual = pred_springmass[:, :-2, 156:]

            # ground reaction forces
            # contact status
            seqlen_vid = physics_vertices_world.shape[0]
            contact_normal, contact_z, contact_v_T = smpl.compute_contact_status(
                    torch.from_numpy(physics_vertices_world).float().to(device), constants.dt, seqlen_vid)

            contact_normal = contact_normal.reshape([-1, seqlen_vid, 29, 3])[:, :-2]
            contact_z = contact_z + torch.from_numpy(constants.z_rec).float().to(device).clone().unsqueeze(0)

            # spring-mass model
            contact_z_T = contact_z.reshape([-1, seqlen_vid, 29])[:, :-2]
            z_scaling_T = torch.sigmoid(-constants.sigma_z * contact_z_T).unsqueeze(-1)
            v_scaling_T = 2 * torch.sigmoid(-constants.sigma_v * torch.linalg.norm(contact_v_T[:, :-1], dim=3)).unsqueeze(-1)

            pred_grf_spring_parameters = pred_springmass[:, :-2, :29 * 3].reshape([-1, seqlen_vid - 2, 29, 3])
            
            pred_grf = torch.zeros_like(pred_grf_spring_parameters).float().to(device)
            pred_grf[:, :, :, 0] = (contact_z_T-0.5) * contact_normal[:, :, :, 0].clone() * pred_grf_spring_parameters[:, :, :, 1].clone() +\
                                                contact_v_T[:, :-1, :, 0].clone() * pred_grf_spring_parameters[:, :, :, 2].clone()
            pred_grf[:, :, :, 1] = (contact_z_T-0.5) * contact_normal[:, :, :, 1].clone() * pred_grf_spring_parameters[:, :, :, 1].clone() +\
                                                contact_v_T[:, :-1, :, 1].clone() * pred_grf_spring_parameters[:, :, :, 2].clone()
            pred_grf[:, :, :, 2] = (2-contact_z_T) * pred_grf_spring_parameters[:, :, :, 0].clone() +\
                                                contact_v_T[:, :-1, :, 2].clone() * pred_grf_spring_parameters[:, :, :, 2].clone()
            pred_grf = z_scaling_T * v_scaling_T * pred_grf

            # store the frame idx information for visualization
            frame_idx_list.extend([i for i in range(vid_i_start, vid_i_start + num_frames)])

            np.savez(vid_processed_path.split('_CLIFF.json')[0] + '_output.npz', frame_idx=frame_idx_list,
                             kinematics_vertices_world=kinematics_vertices_world,
                             physics_vertices_world=physics_vertices_world,
                             pred_torques=pred_torques.detach().cpu().numpy(),
                             pred_grf=pred_grf.detach().cpu().numpy())


# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--vid_processed_path', type=str, required=True, help='Path to the processed data path')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size when performing inference')
parser.add_argument('--gpu', type=str, default='0', help='GPU to be used')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    PhysPT = PhysPT(device=device,
                    seqlen=config.seqlen,
                    mode='test',
                    f_dim=144 + 3 + 72 + 60,
                    d_model=1024,
                    nhead=8,
                    d_hid=1024,
                    nlayers=6,
                    dropout=0.1).to(device)
    PhysPT.load_state_dict(torch.load('./assets/checkpoint/PhysPT.pt'), strict=True)
    PhysPT.eval()

    GlobalTrajPredictor = GlobalTrajPredictor(device=device).to(device)
    GlobalTrajPredictor.load_state_dict(torch.load('./assets/checkpoint/GlobalTrajPredictor.pt'), strict=True)
    GlobalTrajPredictor.eval()

    print('loaded PhysPT and GlobalTrajPredictor')

    # Run Inference
    video_inference(PhysPT, GlobalTrajPredictor, args.vid_processed_path, args.batch_size, device)
