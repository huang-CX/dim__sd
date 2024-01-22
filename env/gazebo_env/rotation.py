import numpy as np
# from autolab_core import RigidTransform
# from const import *
# #
# #
# def transform_matrix(orientation):
#
#     rotation_quaternion = np.asarray([orientation[3], orientation[0], orientation[1], orientation[2]])
#     norm = np.linalg.norm(rotation_quaternion)
#     rotation_quaternion = rotation_quaternion/norm
#     pose = np.asarray([position[0], position[1], position[2]])
#     # https://www.cnblogs.com/flyinggod/p/8144100.html
#     r_matrix = RigidTransform(rotation_quaternion, pose)
#     t_matrix = r_matrix.rotation
#     euler = r_matrix.euler
#     v = np.dot(t_matrix, pose)
#     return v, euler
#
#
# def cos_angle(orientation1, step_vector):
#
#     vec1, euler = transform_matrix(orientation1)
#     vec1 = vec1[0:2]
#     cos = np.dot(vec1, step_vector) / (np.linalg.norm(vec1) * np.linalg.norm(step_vector))
#     if -1.0 <= cos <= 1.0:
#         cos = cos
#     else:
#         cos = 0
#     return cos, euler


if __name__ == '__main__':
    quaternion = (0, 0, 0, 1)
    # step_v = (2, 2)
    # cos_alpha = cos_angle(quaternion, step_v)
    # print(cos_alpha)
    # A = [1, 2, 3]
    # B = [4, 5, 6]
    # C = A + B
    # C = np.asarray(C)
    # print(C)
    # C[1:3] = 2*C[1:3]
    # print(C)
    # Aa = [-0.02, -0.496, -0.01, 0.471, 0.018, 0.0, 0.213, 0.101, -0.173, -0.08, -0.014, -0.185]
    # Aa = 0.125*np.asarray(Aa)
    # print(Aa)

    # q_orig = transformations.quaternion_from_euler(0, 0, 0)
    # q_rot = transformations.quaternion_from_euler(pi, 0, 0)
    # q_new = transformations.quaternion_multiply(q_rot, q_orig)
    # print(q_new)

