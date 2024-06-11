import torch
import torch.nn.functional as F

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat2eulerzyx(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    N = rotmat.size()[0]
    cy_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N, 3).type(torch.float32).to(rotmat.device)

    r11, r12, r13 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    r21, r22, r23 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    r31, r32, r33 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]

    # cy: sqrt((cos(y)*sin(x))**2 + (cos(x)*cos(y))**2) = cos(y)
    cy = torch.sqrt(r33 * r33 + r23 * r23)
    # y = atan(sin(y),con(y))
    eulerangle[:, 1] = torch.atan2(r13, cy)  # [-pi,pi]

    # c>cy_thresh
    # -cos(y)*sin(z) / cos(y)*cos(z) = tanz, z = atan(sin(z),con(z))
    eulerangle[cy > cy_thresh, 2] = torch.atan2(-r12[cy > cy_thresh], r11[cy > cy_thresh])
    # -cos(y)*sin(x)] / cos(y)*cos(x) = tanx, x = atan(sin(x),con(x))
    eulerangle[cy > cy_thresh, 0] = torch.atan2(-r23[cy > cy_thresh], r33[cy > cy_thresh])

    # cy<=cy_thresh
    # r21 = sin(z), r22 = cos(z)
    eulerangle[cy <= cy_thresh, 2] = torch.atan2(r21[cy <= cy_thresh], r22[cy <= cy_thresh])

    return eulerangle


def rotmat2euleryzx(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),      - sin(z),                         cos(z)sin(y)],
      [sin(x)*sin(y) + cos(x)*cos(y)*sin(z), cos(x)*cos(z), cos(x)*sin(y)*sin(z) - cos(y)*sin(x)],
      [cos(y)*sin(z)*sin(x)-cos(x)*sin(y),   cos(z)*sin(x), cos(x)*cos(y) + sin(x)*sin(y)*sin(z)]
    with the obvious derivations for z, y, and x
       z = asin(r12)
       y = atan2(r13, r11)
       x = atan2(r32, r22)
    Problems arise when cos(z) is close to zero, because both of::
    '''
    N = rotmat.size()[0]
    cz_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N, 3).type(torch.float32).to(rotmat.device)

    r11, r12, r13 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    r21, r22, r23 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    r31, r32, r33 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]

    # cz: sqrt((cos(y)*cos(z))**2 + (cos(z)sin(y))**2) = cos(y)
    cz = torch.sqrt(r11 * r11 + r13 * r13)
    # z = atan(sin(z),con(z))
    eulerangle[:, 2] = torch.atan2(-r12, cz)  # [-pi,pi]

    # c>cz_thresh
    eulerangle[cz > cz_thresh, 1] = torch.atan2(r13[cz > cz_thresh], r11[cz > cz_thresh])
    eulerangle[cz > cz_thresh, 0] = torch.atan2(r32[cz > cz_thresh], r22[cz > cz_thresh])

    # cy<=cy_thresh
    eulerangle[cz <= cz_thresh, 0] = torch.atan2(-r23[cz <= cz_thresh], r33[cz <= cz_thresh])

    return eulerangle


def rotmat2eulerzxy(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, x, y axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [cos(y)*cos(z) + sin(x)*sin(y)*sin(z), cos(z)*sin(x)*sin(y) - cos(y)*sin(z), cos(x)*sin(y)],
      [                       cos(x)*sin(z),                        cos(x)*cos(z),      - sin(x)],
      [cos(y)*sin(x)*sin(z) - cos(z)*sin(y), sin(y)*sin(z) + cos(y)*cos(z)*sin(x), cos(x)*cos(y)]
    '''
    N = rotmat.size()[0]
    cx_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N, 3).type(torch.float32).to(rotmat.device)

    r11, r12, r13 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    r21, r22, r23 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    r31, r32, r33 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]

    # cx
    cx = torch.sqrt(r21 * r21 + r22 * r22)
    # x = atan(sin(x),con(x))
    eulerangle[:, 0] = torch.atan2(-r23, cx)  # [-pi,pi]

    # c>cx_thresh
    eulerangle[cx > cx_thresh, 2] = torch.atan2(r21[cx > cx_thresh], r22[cx > cx_thresh])
    eulerangle[cx > cx_thresh, 1] = torch.atan2(r13[cx > cx_thresh], r33[cx > cx_thresh])

    # cy<=cy_thresh
    eulerangle[cx <= cx_thresh, 2] = torch.atan2(-r12[cx <= cx_thresh], r11[cx <= cx_thresh])

    return eulerangle


def rotmat2eulerSMPL(rotmat):
    euler_root = rotmat2euleryzx(rotmat[:, :1, :, :].clone().view(-1, 3, 3)).view(-1, 1, 3)
    euler_s = rotmat2eulerzyx(rotmat[:, 1:16, :, :].clone().view(-1, 3, 3)).view(-1, 15, 3)
    euler_shoulder = rotmat2eulerzxy(rotmat[:, 16:18, :, :].clone().view(-1, 3, 3)).view(-1, 2, 3)
    euler_elbow = rotmat2euleryzx(rotmat[:, 18:20, :, :].clone().view(-1, 3, 3)).view(-1, 2, 3)
    euler_e = rotmat2eulerzyx(rotmat[:, 20:, :, :].clone().view(-1, 3, 3)).view(-1, 4, 3)

    euler = torch.cat((euler_root, euler_s, euler_shoulder, euler_elbow, euler_e), dim=1)
    return euler


def batch_euler2matyxz(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = zmat @ xmat @ ymat
    rotMat_individual = torch.stack([ymat, xmat, zmat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    rotMat_individual = torch.stack([zmat, xmat, ymat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matzyx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    rotMat_individual = torch.stack([zmat, ymat, xmat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matyzx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ zmat @ ymat
    rotMat_individual = torch.stack([ymat, zmat, xmat], dim=1)
    return rotMat, rotMat_individual


def batch_roteulerSMPL(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 72], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)

    rotMat_root, rotMat_root_individual = batch_euler2matyzx(angle[:, :3].reshape(-1, 3))
    rotMat_s, rotMat_s_individual = batch_euler2matzyx(angle[:, 3:48].reshape(-1, 3))
    rotMat_shoulder, rotMat_shoulder_individual = batch_euler2matzxy(angle[:, 48:54].reshape(-1, 3))
    rotMat_elbow, rotMat_elbow_individual = batch_euler2matyzx(angle[:, 54:60].reshape(-1, 3))
    rotMat_e, rotMat_e_individual = batch_euler2matzyx(angle[:, 60:].reshape(-1, 3))

    rotMat_root = rotMat_root.reshape(B, 1, 3, 3)
    rotMat_s = rotMat_s.reshape(B, 15, 3, 3)
    rotMat_shoulder = rotMat_shoulder.reshape(B, 2, 3, 3)
    rotMat_elbow = rotMat_elbow.reshape(B, 2, 3, 3)
    rotMat_e = rotMat_e.reshape(B, 4, 3, 3)
    rotMat = torch.cat((rotMat_root, rotMat_s, rotMat_shoulder, rotMat_elbow, rotMat_e), dim=1)

    rotMat_root_individual = rotMat_root_individual.reshape(B, 1, 3, 3, 3)
    rotMat_s_individual = rotMat_s_individual.reshape(B, 15, 3, 3, 3)
    rotMat_shoulder_individual = rotMat_shoulder_individual.reshape(B, 2, 3, 3, 3)
    rotMat_elbow_individual = rotMat_elbow_individual.reshape(B, 2, 3, 3, 3)
    rotMat_e_individual = rotMat_e_individual.reshape(B, 4, 3, 3, 3)
    rotMat_individual = torch.cat((rotMat_root_individual,
                                   rotMat_s_individual,
                                   rotMat_shoulder_individual,
                                   rotMat_elbow_individual,
                                   rotMat_e_individual),
                                  dim=1).reshape(B, -1, 3, 3)
    return rotMat, rotMat_individual


def batch_global_rigid_transformation(Rs, Js, parent):
    """
    Computes 3D joint locations given pose. J_child = A_parent * A_child[:, :, :3, 3]
    Args:
      Rs: N x 24 x 3 x 3, rotation matrix of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24, holding the parent id for each joint
    Returns
      J_transformed : N x 24 x 3 location of absolute joints
      A_relative: N x 24 4 x 4 relative transformation matrix for LBS.
    """

    def make_A(R, t, N):
        """
        construct transformation matrix for a joint
            Args: 
                R: N x 3 x 3, rotation matrix 
                t: N x 3 x 1, bone vector (child-parent)
            Returns:
                A: N x 4 x 4, transformation matrix
        """
        # N x 4 x 3
        R_homo = F.pad(R, (0, 0, 0, 1))
        # N x 4 x 1
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).type(torch.float32).to(R.device)], 1)
        # N x 4 x 4
        return torch.cat([R_homo, t_homo], 2)

    # obtain the batch size
    N = Rs.size()[0]
    # unsqueeze Js to N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)

    # rot_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).type(torch.float64).to(Rs.device)
    # rot_x = rot_x.repeat([N, 1]).view([N, 3, 3])
    # root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    root_rotation = Rs[:, 0, :, :]
    # transformation matrix of the root
    A0 = make_A(root_rotation, Js[:, 0], N)
    A = [A0]
    # caculate transformed matrix of each joint
    for i in range(1, parent.shape[0]):
        # transformation matrix
        t_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], t_here, N)
        # transformation given parent matrix
        A_here_tran = torch.matmul(A[parent[i]], A_here)
        A.append(A_here_tran)

    # N x 24 x 4 x 4, transformation matrix for each joint
    A = torch.stack(A, dim=1)
    # recover transformed joints from the transformed transformation matrix
    J_transformed = A[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    # N x 24 x 3 x 1 to N x 24 x 4 x 1, homo with zeros
    Js_homo = torch.cat([Js, torch.zeros([N, 24, 1, 1]).type(torch.float32).to(Rs.device)], 2)
    # N x 24 x 4 x 1
    init_bone = torch.matmul(A, Js_homo)
    # N x 24 x 4 x 4, For each 4 x 4, last column is the joints position, and otherwise 0. 
    init_bone = F.pad(init_bone, (3, 0))
    A_relative = A - init_bone
    return J_transformed, A_relative


def combine_output(smplh_m_output, smplh_f_output, smpl_output, gender_id, size, device):
    output = torch.zeros(size).float().to(device)
    output[gender_id == 0] = smplh_m_output
    output[gender_id == 1] = smplh_f_output
    output[gender_id == 2] = smpl_output
    return output
