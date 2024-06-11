from __future__ import division

import pickle
from collections import namedtuple

import numpy as np
import torch

import config
import constants
from assets.utils import batch_global_rigid_transformation

ModelOutput = namedtuple('ModelOutput',
                         ['v_shaped', 'joint_smpl_shaped', 'vertices', 'joints', 'joints_smpl', 'physics_parameters'
                          ])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


# %%
class SMPL(object):
    def __init__(self, model_path=config.SMPL_MODEL_PATH, device=None):
        super(SMPL, self).__init__()
        """
        Build SMPL model
        """
        self.device = device if device is not None else torch.device('cpu')

        # -- Load SMPL params --        
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        # Mean template vertices: [6890, 3]
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.faces = params['f']
        # joint regressor
        self.regressor = torch.from_numpy(params['V_regressor']).type(torch.float32).transpose(1, 0)
        self.J_regressor = torch.from_numpy(params['J_regressor']).type(torch.float32).transpose(1, 0)
        # Parent for 24 and 37
        self.parents = params['kintree_table'].astype(np.int32)
        self.J_parents = params['kintree_table_J'].astype(np.int32)
        # Shape blend shape basis: [6890, 3, 10]
        # transposed to [10, 6890, 3]
        self.shapedirs = torch.from_numpy(params['shapedirs'].transpose(2, 0, 1)).type(torch.float32)
        # Pose blend shape basis: [6890, 3, 207]
        # transposed to [207, 6890, 3]
        self.posedirs = torch.from_numpy(params['posedirs'].transpose(2, 0, 1)).type(torch.float32)
        # LBS weights [6890, 24]
        self.weights = torch.from_numpy(params['weights']).type(torch.float32)
        self.joints_num = 24
        self.verts_num = 6890

        # physics
        with open(config.PHYSICS_PATH, 'rb') as f:
            physics_data = pickle.load(f)

        self.bodypart_faceid = physics_data['bodypart_faceid']
        self.bodypart_vertexid = physics_data['bodypart_vertexid']

        self.mass = torch.from_numpy(physics_data['mass']).type(torch.float32).unsqueeze(0).to(self.device)
        self.mass_scaling = torch.from_numpy(physics_data['mass_scaling']).type(torch.float32).unsqueeze(0).to(
            self.device)
        self.volumn = torch.from_numpy(physics_data['volumn']).type(torch.float32).unsqueeze(0).to(self.device)
        self.inertia = torch.from_numpy(physics_data['inertia']).type(torch.float32).unsqueeze(0).to(self.device)

        self.axis = torch.from_numpy(physics_data['h']).type(torch.float32).unsqueeze(0).to(self.device)  # 1,3,75
        self.rotation_parent = physics_data['h_rotation_parents']  # 71

        self.parent_chain_part = physics_data['parent_chain_part']

        self.gravitational_force = torch.tensor([[[0, 0, -9.81]]]).type(torch.float32).to(self.device)  # 1, 1, 3

        self.contact_vertexidx = physics_data['contact_vertexidx']
        self.contact_vertexidxall = physics_data['contact_vertexidxall']

        self.contact_facevertex_idx0 = self.contact_vertexidxall
        self.contact_facevertex_idx1 = physics_data['contact_facevertex_idx1']
        self.contact_facevertex_idx2 = physics_data['contact_facevertex_idx2']

        for name in ['v_template', 'J_regressor', 'regressor', 'weights', 'posedirs', 'shapedirs',
                     ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(self.device))

    def forward(self, betas, rotmat):

        N = betas.size()[0]
        # 1. Add shape blend shapes
        # (N, 10) x (10, 6890, 3) = [N, 6890, 3]
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [0])) + self.v_template.unsqueeze(0)
        joint_smpl_shaped = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)

        # 2. Add pose blend shapes
        # 2.1 Infer shape-dependent joint locations.
        # transpose [N, 6890, 3] to [N, 3, 6890] and perform multiplication
        # transpose results [N, 3, J] to [N, J, 3]
        J = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)
        # 2.2 add pose blend shapes 
        # rotation matrix [N,24,3,3]
        Rs = rotmat
        # ignore global rotation [N,23,3,3]
        pose = Rs[:, 1:, :, :]
        # rotation of T-pose
        pose_I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # weight matrix [N, 207]
        lrotmin = (pose - pose_I).view(-1, 207)
        # blended model [N,6890,3]
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [0]))

        # 3. Do LBS
        # obtain the transformed transformation matrix
        _, A = batch_global_rigid_transformation(Rs, J, self.parents)
        # repeat the weight matrix to [N,6890,24]
        W = self.weights.repeat(N, 1, 1)
        # calculate the blended transformation matrix 
        # [N,6890,24] * [N,24,16] = [N,6890,16] > [N,6890,4,4]
        T = torch.matmul(W, A.view(N, 24, 16)).view(N, 6890, 4, 4)
        # homegeous form of blended model [N,6890,4]
        v_posed_homo = torch.cat([v_posed,
                                  torch.ones([N, self.verts_num, 1]).type(torch.float32).to(self.device)], dim=2)
        # calculate the transformed 3D vertices position
        # [N,6890,4,4] * [N,6890,4,1] = [N,6890,4,1]
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:, :, :3, 0]  # [N,6890,3]

        # estimate 3D joint locations
        joint_regressed = torch.matmul(verts.transpose(1, 2), self.J_regressor).transpose(1, 2)
        # estimate 3D joint locations
        joint_regressed_smpl = torch.matmul(verts.transpose(1, 2), self.regressor).transpose(1, 2)

        output = ModelOutput(v_shaped=v_shaped,
                             joint_smpl_shaped=joint_smpl_shaped,
                             vertices=verts,
                             joints=joint_regressed,
                             joints_smpl=joint_regressed_smpl)
        return output

    def compute_massinertia(self, v_shaped, joints_smpl):
        N = v_shaped.size()[0]
        mass = self.mass.clone().repeat(N, 1)
        volume = self.volumn.clone().repeat(N, 1)
        inertia = self.inertia.clone().repeat(N, 1, 1, 1)

        vertices_all = torch.cat([v_shaped, joints_smpl], dim=1)
        for n_b in range(24):
            vertex_list = self.bodypart_vertexid[n_b]
            centroid = torch.mean(vertices_all[:, vertex_list], dim=1, keepdims=True)

            # update volumn
            triangle_list = np.concatenate(self.bodypart_faceid[n_b])
            triangles = (vertices_all[:, triangle_list] - centroid).reshape([-1, 3, 3])  # N*N_face, 3, 3
            volume_triangle = (torch.abs(torch.det(triangles)) / 6).reshape(N, len(triangle_list) // 3)
            volume[:, n_b] = torch.sum(volume_triangle, dim=1)  # in m^3

            # update mass
            mass[:, n_b] = (volume[:, n_b] / self.volumn[:, n_b].clone() - 1) * self.mass[:,
                                                                                n_b].clone() * self.mass_scaling[:,
                                                                                               n_b].clone() + self.mass[
                                                                                                              :,
                                                                                                              n_b].clone()

            # update inertia
            triangles_batch = triangles.reshape([N, len(triangle_list) // 3, 3, 3])  # N, N_face, 3, 3
            triangles_x = triangles_batch[:, :, :, 0]  # N, N_face, 3
            triangles_y = triangles_batch[:, :, :, 1]  # N, N_face, 3
            triangles_z = triangles_batch[:, :, :, 2]  # N, N_face, 3
            inertia[:, n_b, 2, 2] = torch.sum((triangles_x[:, :, 0] ** 2 + triangles_y[:, :, 0] ** 2 + \
                                               triangles_x[:, :, 1] ** 2 + triangles_y[:, :, 1] ** 2 + \
                                               triangles_x[:, :, 2] ** 2 + triangles_y[:, :, 2] ** 2 + \
                                               (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :,
                                                                                              2]) ** 2 + (
                                                           triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:,
                                                                                                         :, 2]) ** 2) * \
                                              volume_triangle, dim=1)

            inertia[:, n_b, 1, 1] = torch.sum((triangles_x[:, :, 0] ** 2 + triangles_z[:, :, 0] ** 2 + \
                                               triangles_x[:, :, 1] ** 2 + triangles_z[:, :, 1] ** 2 + \
                                               triangles_x[:, :, 2] ** 2 + triangles_z[:, :, 2] ** 2 + \
                                               (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :,
                                                                                              2]) ** 2 + (
                                                           triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                         :, 2]) ** 2) * \
                                              volume_triangle, dim=1)

            inertia[:, n_b, 0, 0] = torch.sum((triangles_y[:, :, 0] ** 2 + triangles_z[:, :, 0] ** 2 + \
                                               triangles_y[:, :, 1] ** 2 + triangles_z[:, :, 1] ** 2 + \
                                               triangles_y[:, :, 2] ** 2 + triangles_z[:, :, 2] ** 2 + \
                                               (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :,
                                                                                              2]) ** 2 + (
                                                           triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                         :, 2]) ** 2) * \
                                              volume_triangle, dim=1)

            inertia[:, n_b, 0, 1] = -torch.sum((triangles_x[:, :, 0] * triangles_y[:, :, 0] + \
                                                triangles_x[:, :, 1] * triangles_y[:, :, 1] + \
                                                triangles_x[:, :, 2] * triangles_y[:, :, 2] + \
                                                (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) * (
                                                            triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:,
                                                                                                          :, 2])) * \
                                               volume_triangle, dim=1)
            inertia[:, n_b, 1, 0] = inertia[:, n_b, 0, 1]

            inertia[:, n_b, 0, 2] = -torch.sum((triangles_x[:, :, 0] * triangles_z[:, :, 0] + \
                                                triangles_x[:, :, 1] * triangles_z[:, :, 1] + \
                                                triangles_x[:, :, 2] * triangles_z[:, :, 2] + \
                                                (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) * (
                                                            triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                          :, 2])) * \
                                               volume_triangle, dim=1)
            inertia[:, n_b, 2, 0] = inertia[:, n_b, 0, 2]

            inertia[:, n_b, 1, 2] = -torch.sum((triangles_y[:, :, 0] * triangles_z[:, :, 0] + \
                                                triangles_y[:, :, 1] * triangles_z[:, :, 1] + \
                                                triangles_y[:, :, 2] * triangles_z[:, :, 2] + \
                                                (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :, 2]) * (
                                                            triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                          :, 2])) * \
                                               volume_triangle, dim=1)
            inertia[:, n_b, 2, 1] = inertia[:, n_b, 1, 2]

        inertia = inertia / 20 * mass[:, :, None, None] / volume[:, :, None, None]

        return mass, volume, inertia

    def compute_contact_status(self, vertices, dt, T=config.seqlen):
        contact_normal = torch.cross(
            vertices[:, self.contact_facevertex_idx1] - vertices[:, self.contact_facevertex_idx0],
            vertices[:, self.contact_facevertex_idx2] - vertices[:, self.contact_facevertex_idx0],
            dim=2)
        contact_normal = contact_normal / torch.norm(contact_normal, dim=2, keepdim=True)

        contact_z = vertices[:, self.contact_facevertex_idx0, 2]

        if T is None:
            vertices_T = vertices[:, self.contact_facevertex_idx0]
            contact_v = (vertices_T[1:] - vertices_T[:-1]) / dt
        else:
            vertices_T = vertices[:, self.contact_facevertex_idx0].reshape([-1, T, 29, 3])
            contact_v = (vertices_T[:, 1:] - vertices_T[:, :-1]) / dt
        return contact_normal, contact_z, contact_v

    def compute_contactJacobian(self, vertices, joints, rotMat_individual, q_dot, dt, T=None):
        # rotMat_individual, B, 72, 3, 3

        N = vertices.shape[0]
        vertices_all = torch.cat([vertices.clone(), joints.clone()], dim=1)
        # compute the axes in world 
        world_n = self.axis.clone().repeat(N, 1, 1)

        world_rotation = [rotMat_individual[:, 0]]
        for i in range(71):
            world_rotation.append(rotMat_individual[:, i + 1] @ world_rotation[self.rotation_parent[i]])
            world_n[:, :, i + 4:i + 5] = world_rotation[i] @ world_n[:, :, i + 4:i + 5].clone()

        JcT = []
        Jp_com = torch.zeros([N, 24, 3, 75]).float().to(self.device)
        Jp_com[:, :, 0, 0], Jp_com[:, :, 1, 1], Jp_com[:, :, 2, 2] = 1., 1., 1.
        Jr_com = torch.zeros([N, 24, 3, 75]).float().to(self.device)
        for n_b in range(24):
            # body joint chain
            parent_chain_joint = np.repeat(self.parent_chain_part[n_b], 3)
            parent_chain_j = parent_chain_joint.copy() * 3 + 3
            parent_chain_j[1::3] = parent_chain_j[1::3] + 1
            parent_chain_j[2::3] = parent_chain_j[2::3] + 2

            # compute the contact Jacobian
            for vertex_i in self.contact_vertexidx[n_b]:
                contact_v_i = vertices[:, vertex_i:vertex_i + 1]
                r_v_i = contact_v_i - joints[:, parent_chain_joint]

                JcT_v_i = torch.zeros([N, 3, 75]).float().to(self.device)
                JcT_v_i[:, 0, 0], JcT_v_i[:, 1, 1], JcT_v_i[:, 2, 2] = 1, 1, 1

                JcT_v_i[:, :, parent_chain_j] = torch.cross(world_n[:, :, parent_chain_j], r_v_i.transpose(1, 2), dim=1)
                JcT.append(JcT_v_i)

            # compute the Generalized mass
            # linear term 
            com_n_b = torch.mean(vertices_all[:, self.bodypart_vertexid[n_b]], dim=1, keepdims=True)  # N, 1, 3
            r_com_n_b = com_n_b - joints[:, parent_chain_joint]
            Jp_com[:, n_b, :, parent_chain_j] = torch.cross(world_n[:, :, parent_chain_j], r_com_n_b.transpose(1, 2),
                                                            dim=1)
            # angular term
            Jr_com[:, n_b, :, parent_chain_j] = world_n[:, :, parent_chain_j]

        JcT = torch.stack(JcT, dim=1)
        return JcT

    def compute_Euler_Lagrange(self, mass, inertia, vertices, joints, rotMat_individual, q_dot, dt, T=None):
        # rotMat_individual, B, 72, 3, 3

        N = vertices.shape[0]
        vertices_all = torch.cat([vertices.clone(), joints.clone()], dim=1)
        # compute the axes in world 
        world_n = self.axis.clone().repeat(N, 1, 1)

        world_rotation = [rotMat_individual[:, 0]]
        for i in range(71):
            world_rotation.append(rotMat_individual[:, i + 1] @ world_rotation[self.rotation_parent[i]])
            world_n[:, :, i + 4:i + 5] = world_rotation[i] @ world_n[:, :, i + 4:i + 5].clone()

        JcT = []
        Jp_com = torch.zeros([N, 24, 3, 75]).float().to(self.device)
        Jp_com[:, :, 0, 0], Jp_com[:, :, 1, 1], Jp_com[:, :, 2, 2] = 1., 1., 1.
        Jr_com = torch.zeros([N, 24, 3, 75]).float().to(self.device)
        for n_b in range(24):
            # body joint chain
            parent_chain_joint = np.repeat(self.parent_chain_part[n_b], 3)
            parent_chain_j = parent_chain_joint.copy() * 3 + 3
            parent_chain_j[1::3] = parent_chain_j[1::3] + 1
            parent_chain_j[2::3] = parent_chain_j[2::3] + 2

            # compute the contact Jacobian
            for vertex_i in self.contact_vertexidx[n_b]:
                contact_v_i = vertices[:, vertex_i:vertex_i + 1]
                r_v_i = contact_v_i - joints[:, parent_chain_joint]

                JcT_v_i = torch.zeros([N, 3, 75]).float().to(self.device)
                JcT_v_i[:, 0, 0], JcT_v_i[:, 1, 1], JcT_v_i[:, 2, 2] = 1, 1, 1

                JcT_v_i[:, :, parent_chain_j] = torch.cross(world_n[:, :, parent_chain_j], r_v_i.transpose(1, 2), dim=1)
                JcT.append(JcT_v_i)

            # compute the Generalized mass
            # linear term 
            com_n_b = torch.mean(vertices_all[:, self.bodypart_vertexid[n_b]], dim=1, keepdims=True)  # N, 1, 3
            r_com_n_b = com_n_b - joints[:, parent_chain_joint]
            Jp_com[:, n_b, :, parent_chain_j] = torch.cross(world_n[:, :, parent_chain_j], r_com_n_b.transpose(1, 2),
                                                            dim=1)
            # angular term
            Jr_com[:, n_b, :, parent_chain_j] = world_n[:, :, parent_chain_j]

        Mq_linear = torch.einsum('nbij,nbjk->nbik', Jp_com.transpose(2, 3), Jp_com) * mass.clone()[:, :, None, None]
        world_rotation_joint = torch.stack(world_rotation[2::3], dim=1)
        world_inertia = torch.einsum('nbij,nbjk,nbkm->nbim', world_rotation_joint, inertia,
                                     world_rotation_joint.transpose(2, 3))
        Mq_angular = torch.einsum('nbij,nbjk,nbkm->nbim', Jr_com.transpose(2, 3), world_inertia, Jr_com)

        Mq = Mq_linear.sum(1) + Mq_angular.sum(1)

        # gravitational force
        G = -(torch.einsum('nbij,nbj->nbi', Jp_com.transpose(2, 3),
                           self.gravitational_force.repeat(N, 24, 1)) * mass.clone()[:, :, None]).sum(1)

        # compute C
        # q_dot: N,T-1,75
        # lienar
        if T is None:
            Jp_com_T = Jp_com
            Jp_com_T_dot = (Jp_com_T[1:] - Jp_com_T[:-1]) / dt

            C_linear = torch.einsum('nbij,nbjk->nbik', Jp_com_T[:-1].transpose(2, 3), Jp_com_T_dot) * mass.clone()[
                                                                                                      :-1].unsqueeze(
                2).unsqueeze(3)

            # angular
            Jr_com_T = Jr_com
            Jr_com_T_dot = (Jr_com_T[1:] - Jr_com_T[:-1]) / dt

            C_angular_term1 = torch.einsum('nbij,nbjk,nbkm->nbim', Jr_com_T[:-1].transpose(2, 3), world_inertia[:-1],
                                           Jr_com_T_dot)

            world_angularvelocity_joint_T = torch.einsum('nbij,nbj->nbi', Jr_com_T[:-1],
                                                         q_dot[:, constants.smpl2q].unsqueeze(1).repeat(1, 24, 1))
            world_angularvelocity_joint_skew_T = torch.zeros([N - 1, 24, 3, 3]).float().to(self.device)
            world_angularvelocity_joint_skew_T[:, :, 0, 1], world_angularvelocity_joint_skew_T[:, :, 0,
                                                            2] = -world_angularvelocity_joint_T[:, :,
                                                                  2], world_angularvelocity_joint_T[:, :, 1]
            world_angularvelocity_joint_skew_T[:, :, 1, 0], world_angularvelocity_joint_skew_T[:, :, 1,
                                                            2] = world_angularvelocity_joint_T[:, :,
                                                                 2], -world_angularvelocity_joint_T[:, :, 0]
            world_angularvelocity_joint_skew_T[:, :, 2, 0], world_angularvelocity_joint_skew_T[:, :, 2,
                                                            1] = -world_angularvelocity_joint_T[:, :,
                                                                  1], world_angularvelocity_joint_T[:, :, 0]

            C_angular_term2 = torch.einsum('nbij,nbjk,nbkm,nbm->nbi', Jr_com_T[:-1].transpose(2, 3),
                                           world_angularvelocity_joint_skew_T, world_inertia[:-1],
                                           world_angularvelocity_joint_T)

            C = torch.einsum('nij,nj->ni', (C_linear + C_angular_term1).sum(1),
                             q_dot[:, constants.smpl2q]) + C_angular_term2.sum(1)

        else:
            world_inertia_T = world_inertia.reshape([-1, T, 24, 3, 3])
            Jp_com_T = Jp_com.reshape([-1, T, 24, 3, 75])
            Jp_com_T_dot = (Jp_com_T[:, 1:] - Jp_com_T[:, :-1]) / dt

            C_linear = torch.einsum('ntbij,ntbjk->ntbik', Jp_com_T[:, :-1].transpose(3, 4),
                                    Jp_com_T_dot) * mass.clone().reshape([-1, T, 24, 1, 1])[:, :-1]

            # angular
            Jr_com_T = Jr_com.reshape([-1, T, 24, 3, 75])
            Jr_com_T_dot = (Jr_com_T[:, 1:] - Jr_com_T[:, :-1]) / dt

            C_angular_term1 = torch.einsum('ntbij,ntbjk,ntbkm->ntbim', Jr_com_T[:, :-1].transpose(3, 4),
                                           world_inertia_T[:, :-1], Jr_com_T_dot)

            world_angularvelocity_joint_T = torch.einsum('ntbij,ntbj->ntbi', Jr_com_T[:, :-1],
                                                         q_dot[:, :, constants.smpl2q].unsqueeze(2).repeat(1, 1, 24, 1))
            world_angularvelocity_joint_skew_T = torch.zeros([N // T, T - 1, 24, 3, 3]).float().to(self.device)
            world_angularvelocity_joint_skew_T[:, :, :, 0, 1], world_angularvelocity_joint_skew_T[:, :, :, 0,
                                                               2] = -world_angularvelocity_joint_T[:, :, :,
                                                                     2], world_angularvelocity_joint_T[:, :, :, 1]
            world_angularvelocity_joint_skew_T[:, :, :, 1, 0], world_angularvelocity_joint_skew_T[:, :, :, 1,
                                                               2] = world_angularvelocity_joint_T[:, :, :,
                                                                    2], -world_angularvelocity_joint_T[:, :, :, 0]
            world_angularvelocity_joint_skew_T[:, :, :, 2, 0], world_angularvelocity_joint_skew_T[:, :, :, 2,
                                                               1] = -world_angularvelocity_joint_T[:, :, :,
                                                                     1], world_angularvelocity_joint_T[:, :, :, 0]

            C_angular_term2 = torch.einsum('ntbij,ntbjk,ntbkm,ntbm->ntbi', Jr_com_T[:, :-1].transpose(3, 4),
                                           world_angularvelocity_joint_skew_T, world_inertia_T[:, :-1],
                                           world_angularvelocity_joint_T)

            C = torch.einsum('ntij,ntj->nti', (C_linear + C_angular_term1).sum(2),
                             q_dot[:, :, constants.smpl2q]) + C_angular_term2.sum(2)

        JcT = torch.stack(JcT, dim=1)
        return JcT, Mq, G, C, world_rotation_joint


class SMPLH(object):
    def __init__(self, gender='neutral', num_betas=10, device=None):
        super(SMPLH, self).__init__()
        """
        Build SMPLH model
        """
        # -- Load SMPL params -- 
        self.device = device if device is not None else torch.device('cpu')

        if gender == 'male':
            params = np.load(config.SMPLH_M_PATH)
        elif gender == 'female':
            params = np.load(config.SMPLH_F_PATH)
        else:
            params = np.load(config.SMPLH_N_PATH)
        # Mean template vertices: [6890, 3]
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.faces = params['f']
        # joint regressor of the official model
        with open(config.SMPL_MODEL_PATH, 'rb') as f:
            params_smpl = pickle.load(f)
        # 24 and 37
        self.regressor = torch.from_numpy(params_smpl['V_regressor']).type(torch.float32).transpose(1, 0)
        self.J_regressor = torch.from_numpy(params_smpl['J_regressor']).type(torch.float32).transpose(1, 0)
        # Parent for 24 and 37
        self.parents = params_smpl['kintree_table'].astype(np.int32)
        self.J_parents = params_smpl['kintree_table_J'].astype(np.int32)

        # transposed to [10, 6890, 3]
        self.shapedirs = torch.from_numpy(params['shapedirs'].transpose(2, 0, 1)).type(torch.float32)[:num_betas]
        # Pose blend shape basis: [6890, 3, 207]
        # transposed to [207, 6890, 3]
        self.posedirs = torch.from_numpy(params['posedirs'].transpose(2, 0, 1)).type(torch.float32)[:9 * 21]
        # LBS weights [6890, 24]
        self.weights = torch.from_numpy(params['weights_prior']).type(torch.float32)
        self.joints_num = 24
        self.verts_num = 6890

        # physics
        with open(config.PHYSICS_PATH, 'rb') as f:
            physics_data = pickle.load(f)

        self.bodypart_faceid = physics_data['bodypart_faceid']
        self.bodypart_vertexid = physics_data['bodypart_vertexid']

        self.mass = torch.from_numpy(physics_data['mass']).type(torch.float32).unsqueeze(0).to(self.device)
        self.mass_scaling = torch.from_numpy(physics_data['mass_scaling']).type(torch.float32).unsqueeze(0).to(
            self.device)
        self.volumn = torch.from_numpy(physics_data['volumn']).type(torch.float32).unsqueeze(0).to(self.device)
        self.inertia = torch.from_numpy(physics_data['inertia']).type(torch.float32).unsqueeze(0).to(self.device)

        self.axis = torch.from_numpy(physics_data['h']).type(torch.float32).unsqueeze(0).to(self.device)  # 1,3,75
        self.rotation_parent = physics_data['h_rotation_parents']  # 71

        self.parent_chain_part = physics_data['parent_chain_part']

        self.gravitational_force = torch.tensor([[[0, 0, -9.81]]]).type(torch.float32).to(self.device)  # 1, 1, 3

        self.contact_vertexidx = physics_data['contact_vertexidx']
        self.contact_vertexidxall = physics_data['contact_vertexidxall']

        self.contact_facevertex_idx0 = self.contact_vertexidxall
        self.contact_facevertex_idx1 = physics_data['contact_facevertex_idx1']
        self.contact_facevertex_idx2 = physics_data['contact_facevertex_idx2']

        for name in ['v_template', 'J_regressor', 'regressor', 'weights',
                     'posedirs', 'shapedirs',
                     ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(self.device))

    def forward(self, betas, rotmat):
        """
        Obtain SMPL 3D vertices and joints.
        Args:
            thetas: [N, 6], pose parameters, represented in a axis-angle format. 
            root joint it's global orientation (first three elements).

            betas: [N, 10] shape parameters, as coefficients of
            PCA components.
         Returns:
            verts: [N, 6890, 3], 3D vertices position in camera frame,
            joints: [N, J, 3], 3D joints positions in camera frame. The value 
            of J depends on the joint regressor type.
        """

        N = betas.size()[0]
        v_shape_mean = self.v_template.clone().repeat(N, 1, 1)
        # 1. Add shape blend shapes
        # (N, 10) x (10, 6890, 3) = [N, 6890, 3]
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [0])) + self.v_template.unsqueeze(0)
        joint_smpl_shaped = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)

        # 2. Add pose blend shapes
        # 2.1 Infer shape-dependent joint locations.
        # transpose [N, 6890, 3] to [N, 3, 6890] and perform multiplication
        # transpose results [N, 3, J] to [N, J, 3]
        J = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)
        # 2.2 add pose blend shapes 
        # rotation matrix [N,24,3,3]
        Rs = rotmat
        # ignore global rotation [N,23,3,3]
        pose = Rs[:, 1:-2, :, :]
        # rotation of T-pose
        pose_I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # weight matrix [N, 207]
        lrotmin = (pose - pose_I).view(-1, 9 * 21)
        # blended model [N,6890,3]
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [0]))

        # 3. Do LBS
        # obtain the transformed transformation matrix
        _, A = batch_global_rigid_transformation(Rs, J, self.parents)
        # repeat the weight matrix to [N,6890,24]
        W = self.weights.repeat(N, 1, 1)
        # calculate the blended transformation matrix 
        # [N,6890,24] * [N,24,16] = [N,6890,16] > [N,6890,4,4]
        T = torch.matmul(W, A.view(N, 24, 16)).view(N, 6890, 4, 4)
        # homegeous form of blended model [N,6890,4]
        v_posed_homo = torch.cat([v_posed,
                                  torch.ones([N, self.verts_num, 1]).type(torch.float32).to(self.device)], dim=2)
        # calculate the transformed 3D vertices position
        # [N,6890,4,4] * [N,6890,4,1] = [N,6890,4,1]
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:, :, :3, 0]  # [N,6890,3]

        # estimate 3D joint locations
        joint_regressed = torch.matmul(verts.transpose(1, 2), self.J_regressor).transpose(1, 2)
        # estimate 3D joint locations
        joint_regressed_smpl = torch.matmul(verts.transpose(1, 2), self.regressor).transpose(1, 2)

        output = ModelOutput(v_shaped=v_shaped,
                             joint_smpl_shaped=joint_smpl_shaped,
                             vertices=verts,
                             joints=joint_regressed,
                             joints_smpl=joint_regressed_smpl)
        return output

    def compute_massinertia(self, v_shaped, joints_smpl):
        N = v_shaped.size()[0]
        mass = self.mass.clone().repeat(N, 1)
        volume = self.volumn.clone().repeat(N, 1)
        inertia = self.inertia.clone().repeat(N, 1, 1, 1)

        vertices_all = torch.cat([v_shaped, joints_smpl], dim=1)
        for n_b in range(24):
            vertex_list = self.bodypart_vertexid[n_b]
            centroid = torch.mean(vertices_all[:, vertex_list], dim=1, keepdims=True)

            # update volumn
            triangle_list = np.concatenate(self.bodypart_faceid[n_b])
            triangles = (vertices_all[:, triangle_list] - centroid).reshape([-1, 3, 3])  # N*N_face, 3, 3
            volume_triangle = (torch.abs(torch.det(triangles)) / 6).reshape(N, len(triangle_list) // 3)
            volume[:, n_b] = torch.sum(volume_triangle, dim=1)  # in m^3

            # update mass
            mass[:, n_b] = (volume[:, n_b] / self.volumn[:, n_b].clone() - 1) * self.mass[:,
                                                                                n_b].clone() * self.mass_scaling[:,
                                                                                               n_b].clone() + self.mass[
                                                                                                              :,
                                                                                                              n_b].clone()

            # update inertia
            triangles_batch = triangles.reshape([N, len(triangle_list) // 3, 3, 3])  # N, N_face, 3, 3
            triangles_x = triangles_batch[:, :, :, 0]  # N, N_face, 3
            triangles_y = triangles_batch[:, :, :, 1]  # N, N_face, 3
            triangles_z = triangles_batch[:, :, :, 2]  # N, N_face, 3
            inertia[:, n_b, 2, 2] = torch.sum((triangles_x[:, :, 0] ** 2 + triangles_y[:, :, 0] ** 2 + \
                                               triangles_x[:, :, 1] ** 2 + triangles_y[:, :, 1] ** 2 + \
                                               triangles_x[:, :, 2] ** 2 + triangles_y[:, :, 2] ** 2 + \
                                               (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :,
                                                                                              2]) ** 2 + (
                                                           triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:,
                                                                                                         :, 2]) ** 2) * \
                                              volume_triangle, dim=1)

            inertia[:, n_b, 1, 1] = torch.sum((triangles_x[:, :, 0] ** 2 + triangles_z[:, :, 0] ** 2 + \
                                               triangles_x[:, :, 1] ** 2 + triangles_z[:, :, 1] ** 2 + \
                                               triangles_x[:, :, 2] ** 2 + triangles_z[:, :, 2] ** 2 + \
                                               (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :,
                                                                                              2]) ** 2 + (
                                                           triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                         :, 2]) ** 2) * \
                                              volume_triangle, dim=1)

            inertia[:, n_b, 0, 0] = torch.sum((triangles_y[:, :, 0] ** 2 + triangles_z[:, :, 0] ** 2 + \
                                               triangles_y[:, :, 1] ** 2 + triangles_z[:, :, 1] ** 2 + \
                                               triangles_y[:, :, 2] ** 2 + triangles_z[:, :, 2] ** 2 + \
                                               (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :,
                                                                                              2]) ** 2 + (
                                                           triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                         :, 2]) ** 2) * \
                                              volume_triangle, dim=1)

            inertia[:, n_b, 0, 1] = -torch.sum((triangles_x[:, :, 0] * triangles_y[:, :, 0] + \
                                                triangles_x[:, :, 1] * triangles_y[:, :, 1] + \
                                                triangles_x[:, :, 2] * triangles_y[:, :, 2] + \
                                                (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) * (
                                                            triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:,
                                                                                                          :, 2])) * \
                                               volume_triangle, dim=1)
            inertia[:, n_b, 1, 0] = inertia[:, n_b, 0, 1]

            inertia[:, n_b, 0, 2] = -torch.sum((triangles_x[:, :, 0] * triangles_z[:, :, 0] + \
                                                triangles_x[:, :, 1] * triangles_z[:, :, 1] + \
                                                triangles_x[:, :, 2] * triangles_z[:, :, 2] + \
                                                (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) * (
                                                            triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                          :, 2])) * \
                                               volume_triangle, dim=1)
            inertia[:, n_b, 2, 0] = inertia[:, n_b, 0, 2]

            inertia[:, n_b, 1, 2] = -torch.sum((triangles_y[:, :, 0] * triangles_z[:, :, 0] + \
                                                triangles_y[:, :, 1] * triangles_z[:, :, 1] + \
                                                triangles_y[:, :, 2] * triangles_z[:, :, 2] + \
                                                (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :, 2]) * (
                                                            triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:,
                                                                                                          :, 2])) * \
                                               volume_triangle, dim=1)
            inertia[:, n_b, 2, 1] = inertia[:, n_b, 1, 2]

        inertia = inertia / 20 * mass[:, :, None, None] / volume[:, :, None, None]
        return mass, volume, inertia

    def compute_contact_status(self, vertices, dt, T=config.seqlen):
        contact_normal = torch.cross(
            vertices[:, self.contact_facevertex_idx1] - vertices[:, self.contact_facevertex_idx0],
            vertices[:, self.contact_facevertex_idx2] - vertices[:, self.contact_facevertex_idx0], dim=2)
        contact_normal = contact_normal / torch.norm(contact_normal, dim=2, keepdim=True)

        contact_z = vertices[:, self.contact_facevertex_idx0, 2]

        if T is None:
            vertices_T = vertices[:, self.contact_facevertex_idx0]
            contact_v = (vertices_T[1:] - vertices_T[:-1]) / dt
        else:
            vertices_T = vertices[:, self.contact_facevertex_idx0].reshape([-1, T, 29, 3])
            contact_v = (vertices_T[:, 1:] - vertices_T[:, :-1]) / dt
        return contact_normal, contact_z, contact_v

    def compute_Euler_Lagrange(self, mass, inertia, vertices, joints, rotMat_individual, q_dot, dt, T=None):
        # rotMat_individual, B, 72, 3, 3

        N = vertices.shape[0]
        vertices_all = torch.cat([vertices.clone(), joints.clone()], dim=1)
        # compute the axes in world 
        world_n = self.axis.clone().repeat(N, 1, 1)

        world_rotation = [rotMat_individual[:, 0]]
        for i in range(71):
            world_rotation.append(rotMat_individual[:, i + 1] @ world_rotation[self.rotation_parent[i]])
            world_n[:, :, i + 4:i + 5] = world_rotation[i] @ world_n[:, :, i + 4:i + 5].clone()

        JcT = []
        Jp_com = torch.zeros([N, 24, 3, 75]).float().to(self.device)
        Jp_com[:, :, 0, 0], Jp_com[:, :, 1, 1], Jp_com[:, :, 2, 2] = 1., 1., 1.
        Jr_com = torch.zeros([N, 24, 3, 75]).float().to(self.device)
        for n_b in range(24):
            # body joint chain
            parent_chain_joint = np.repeat(self.parent_chain_part[n_b], 3)
            parent_chain_j = parent_chain_joint.copy() * 3 + 3
            parent_chain_j[1::3] = parent_chain_j[1::3] + 1
            parent_chain_j[2::3] = parent_chain_j[2::3] + 2
            # compute the contact Jacobian
            for vertex_i in self.contact_vertexidx[n_b]:
                contact_v_i = vertices[:, vertex_i:vertex_i + 1]
                r_v_i = contact_v_i - joints[:, parent_chain_joint]

                JcT_v_i = torch.zeros([N, 3, 75]).float().to(self.device)
                JcT_v_i[:, 0, 0], JcT_v_i[:, 1, 1], JcT_v_i[:, 2, 2] = 1, 1, 1

                JcT_v_i[:, :, parent_chain_j] = torch.cross(world_n[:, :, parent_chain_j], r_v_i.transpose(1, 2), dim=1)
                JcT.append(JcT_v_i)

            # compute the Generalized mass
            # linear term 
            com_n_b = torch.mean(vertices_all[:, self.bodypart_vertexid[n_b]], dim=1, keepdims=True)  # N, 1, 3
            r_com_n_b = com_n_b - joints[:, parent_chain_joint]
            Jp_com[:, n_b, :, parent_chain_j] = torch.cross(world_n[:, :, parent_chain_j], r_com_n_b.transpose(1, 2),
                                                            dim=1)
            # angular term
            Jr_com[:, n_b, :, parent_chain_j] = world_n[:, :, parent_chain_j]

        Mq_linear = torch.einsum('nbij,nbjk->nbik', Jp_com.transpose(2, 3), Jp_com) * mass.clone()[:, :, None, None]
        world_rotation_joint = torch.stack(world_rotation[2::3], dim=1)
        world_inertia = torch.einsum('nbij,nbjk,nbkm->nbim', world_rotation_joint, inertia,
                                     world_rotation_joint.transpose(2, 3))
        Mq_angular = torch.einsum('nbij,nbjk,nbkm->nbim', Jr_com.transpose(2, 3), world_inertia, Jr_com)

        Mq = Mq_linear.sum(1) + Mq_angular.sum(1)

        # gravitational force
        G = -(torch.einsum('nbij,nbj->nbi', Jp_com.transpose(2, 3),
                           self.gravitational_force.repeat(N, 24, 1)) * mass.clone()[:, :, None]).sum(1)

        # compute C
        # q_dot: N,T-1,75
        # lienar
        if T is None:
            Jp_com_T = Jp_com
            Jp_com_T_dot = (Jp_com_T[1:] - Jp_com_T[:-1]) / dt

            C_linear = torch.einsum('nbij,nbjk->nbik', Jp_com_T[:-1].transpose(2, 3), Jp_com_T_dot) * mass.clone()[
                                                                                                      :-1].unsqueeze(
                2).unsqueeze(3)

            # angular
            Jr_com_T = Jr_com
            Jr_com_T_dot = (Jr_com_T[1:] - Jr_com_T[:-1]) / dt

            C_angular_term1 = torch.einsum('nbij,nbjk,nbkm->nbim', Jr_com_T[:-1].transpose(2, 3), world_inertia[:-1],
                                           Jr_com_T_dot)

            world_angularvelocity_joint_T = torch.einsum('nbij,nbj->nbi', Jr_com_T[:-1],
                                                         q_dot[:, constants.smpl2q].unsqueeze(1).repeat(1, 24, 1))
            world_angularvelocity_joint_skew_T = torch.zeros([N - 1, 24, 3, 3]).float().to(self.device)
            world_angularvelocity_joint_skew_T[:, :, 0, 1], world_angularvelocity_joint_skew_T[:, :, 0,
                                                            2] = -world_angularvelocity_joint_T[:, :,
                                                                  2], world_angularvelocity_joint_T[:, :, 1]
            world_angularvelocity_joint_skew_T[:, :, 1, 0], world_angularvelocity_joint_skew_T[:, :, 1,
                                                            2] = world_angularvelocity_joint_T[:, :,
                                                                 2], -world_angularvelocity_joint_T[:, :, 0]
            world_angularvelocity_joint_skew_T[:, :, 2, 0], world_angularvelocity_joint_skew_T[:, :, 2,
                                                            1] = -world_angularvelocity_joint_T[:, :,
                                                                  1], world_angularvelocity_joint_T[:, :, 0]

            C_angular_term2 = torch.einsum('nbij,nbjk,nbkm,nbm->nbi', Jr_com_T[:-1].transpose(2, 3),
                                           world_angularvelocity_joint_skew_T, world_inertia[:-1],
                                           world_angularvelocity_joint_T)

            C = torch.einsum('nij,nj->ni', (C_linear + C_angular_term1).sum(1),
                             q_dot[:, constants.smpl2q]) + C_angular_term2.sum(1)

        else:
            world_inertia_T = world_inertia.reshape([-1, T, 24, 3, 3])
            Jp_com_T = Jp_com.reshape([-1, T, 24, 3, 75])
            Jp_com_T_dot = (Jp_com_T[:, 1:] - Jp_com_T[:, :-1]) / dt

            C_linear = torch.einsum('nbtij,nbtjk->nbtik', Jp_com_T[:, :-1].transpose(3, 4),
                                    Jp_com_T_dot) * mass.clone().reshape([-1, T, 24, 1, 1])[:, :-1]

            # angular
            Jr_com_T = Jr_com.reshape([-1, T, 24, 3, 75])
            Jr_com_T_dot = (Jr_com_T[:, 1:] - Jr_com_T[:, :-1]) / dt

            C_angular_term1 = torch.einsum('ntbij,ntbjk,ntbkm->ntbim', Jr_com_T[:, :-1].transpose(3, 4),
                                           world_inertia_T[:, :-1], Jr_com_T_dot)

            world_angularvelocity_joint_T = torch.einsum('ntbij,ntbj->ntbi', Jr_com_T[:, :-1],
                                                         q_dot[:, :, constants.smpl2q].unsqueeze(2).repeat(1, 1, 24, 1))
            world_angularvelocity_joint_skew_T = torch.zeros([N // T, T - 1, 24, 3, 3]).float().to(self.device)
            world_angularvelocity_joint_skew_T[:, :, :, 0, 1], world_angularvelocity_joint_skew_T[:, :, :, 0,
                                                               2] = -world_angularvelocity_joint_T[:, :, :,
                                                                     2], world_angularvelocity_joint_T[:, :, :, 1]
            world_angularvelocity_joint_skew_T[:, :, :, 1, 0], world_angularvelocity_joint_skew_T[:, :, :, 1,
                                                               2] = world_angularvelocity_joint_T[:, :, :,
                                                                    2], -world_angularvelocity_joint_T[:, :, :, 0]
            world_angularvelocity_joint_skew_T[:, :, :, 2, 0], world_angularvelocity_joint_skew_T[:, :, :, 2,
                                                               1] = -world_angularvelocity_joint_T[:, :, :,
                                                                     1], world_angularvelocity_joint_T[:, :, :, 0]

            C_angular_term2 = torch.einsum('ntbij,ntbjk,ntbkm,ntbm->ntbi', Jr_com_T[:, :-1].transpose(3, 4),
                                           world_angularvelocity_joint_skew_T, world_inertia_T[:, :-1],
                                           world_angularvelocity_joint_T)

            C = torch.einsum('ntij,ntj->nti', (C_linear + C_angular_term1).sum(2),
                             q_dot[:, :, constants.smpl2q]) + C_angular_term2.sum(2)

        JcT = torch.stack(JcT, dim=1)
        return JcT, Mq, G, C, world_rotation_joint
