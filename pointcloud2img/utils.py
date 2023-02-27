import numpy as np
import decimal

def getRT(Quaternion, translation):

    rt_mat = np.zeros((3,4)).astype(float)

    rt_mat[0][0] = 1 - 2 * (Quaternion[2] * Quaternion[2]) - 2 * (Quaternion[3] * Quaternion[3])
    rt_mat[0][1] = 2 * Quaternion[1] * Quaternion[2] - 2 * Quaternion[0] * Quaternion[3]
    rt_mat[0][2] = 2 * Quaternion[1] * Quaternion[3] + 2 * Quaternion[0] * Quaternion[2]
    rt_mat[1][0] = 2 * Quaternion[1] * Quaternion[2] + 2 * Quaternion[0] * Quaternion[3]
    rt_mat[1][1] = 1 - 2 * (Quaternion[1] * Quaternion[1]) - 2 * (Quaternion[3] * Quaternion[3])
    rt_mat[1][2] = 2 * Quaternion[2] * Quaternion[3] - 2 * Quaternion[0] * Quaternion[1]
    rt_mat[2][0] = 2 * Quaternion[1] * Quaternion[3] - 2 * Quaternion[0] * Quaternion[2]
    rt_mat[2][1] = 2 * Quaternion[2] * Quaternion[3] + 2 * Quaternion[0] * Quaternion[1]
    rt_mat[2][2] = 1 - 2 * (Quaternion[1] *
    Quaternion[1]) - 2 * (Quaternion[2] * Quaternion[2])
    
    rt_mat[0][3] = translation[0]
    rt_mat[1][3] = translation[1]
    rt_mat[2][3] = translation[2]

    return rt_mat

def drange(x, y, jump):
    yield float(y)
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)
