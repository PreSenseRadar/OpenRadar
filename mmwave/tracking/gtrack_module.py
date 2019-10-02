import numpy as np
import sys

from . import ekf_utils
from . import gtrack_unit


# This is a MODULE level predict function. The function is called by external
# step function to perform unit level kalman filter predictions
def module_predict(inst):
    for i in inst.activeList:
        uid = i.data
        if uid > inst.maxNumTracks:
            raise ValueError('module_predict')
        gtrack_unit.unit_predict(inst.hTrack[uid])


# This is a MODULE level associatiation function. The function is called
#  by external step function to associate measurement points with known targets
def module_associate(inst, point, num):
    for i in inst.activeList:
        uid = i.data
        gtrack_unit.unit_score(inst.hTrack[uid], point, inst.bestScore, inst.bestIndex, num)


# This is a MODULE level allocation function. The function is called by
# external step function to allocate new targets for the non-associated 
# measurement points
def module_allocate(inst, point, num):
    un = np.zeros(shape=(3,), dtype=np.float32)
    uk = np.zeros(shape=(3,), dtype=np.float32)
    un_sum = np.zeros(shape=(3,), dtype=np.float32)
    for n in range(num):
        if inst.bestIndex[n] == ekf_utils.gtrack_ID_POINT_NOT_ASSOCIATED:
            t_elem = inst.freeList[0] if inst.freeList else None
            if t_elem is None:
                if (inst.verbose & ekf_utils.VERBOSE_WARNING_INFO) != 0:
                    raise ValueError('Maximum number of tracks reached!, module_allocate')
                return
            inst.allocIndex[0] = n
            alloc_num = 1
            alloc_snr = point[n].snr

            un[0] = un_sum[0] = np.float32(point[n].range)
            un[1] = un_sum[1] = np.float32(point[n].angle)
            un[2] = un_sum[2] = np.float32(point[n].doppler)
            # print(un[0], un[1], un[2])

            for k in range(n + 1, num):
                if inst.bestIndex[k] == ekf_utils.gtrack_ID_POINT_NOT_ASSOCIATED:
                    uk[0] = np.float32(point[k].range)
                    uk[1] = np.float32(point[k].angle)
                    uk[2] = ekf_utils.gtrack_unrollRadialVelocity(inst.params.maxRadialVelocity, un[2],
                                                                  point[k].doppler)

                    if np.abs(uk[2] - un[2]) < inst.params.allocationParams.maxVelThre:
                        dist = np.float32(un[0] * un[0] + uk[0] * uk[0] - 2 * un[0] * uk[0] * np.cos(un[1] - uk[1]))
                        if dist < inst.params.allocationParams.maxDistanceThre:
                            inst.allocIndex[alloc_num] = k

                            un_sum[0] += uk[0]
                            un_sum[1] += uk[1]
                            un_sum[2] += uk[2]

                            alloc_num += 1
                            alloc_snr += point[k].snr

                            un[0] = np.float32(un_sum[0] / alloc_num)
                            un[1] = np.float32(un_sum[1] / alloc_num)
                            un[2] = np.float32(un_sum[2] / alloc_num)
            if (alloc_num > inst.params.allocationParams.pointsThre) and \
                    (alloc_snr > inst.params.allocationParams.snrThre) and \
                    (np.abs(un[2]) > inst.params.allocationParams.velocityThre):
                for k in range(alloc_num):
                    inst.bestIndex[inst.allocIndex[k]] = t_elem.data

                inst.targetNumTotal += 1
                inst.targetNumCurrent += 1

                t_elem = inst.freeList.pop(0)

                gtrack_unit.unit_start(inst.hTrack[t_elem.data], inst.heartBeat, inst.targetNumTotal, un)

                inst.activeList.append(t_elem)


# This is a MODULE level update function. The function is called by external 
# step function to perform unit level kalman filter updates
def module_update(inst, point, var, num):
    # NotImplemented

    # first create a list of elements that need to be removed
    need_removal = []
    for i in inst.activeList:
        uid = i.data
        state = gtrack_unit.unit_update(inst.hTrack[uid], point, var, inst.bestIndex, num)
        if state == ekf_utils.TrackState().TRACK_STATE_FREE:
            need_removal.append(i)
            inst.targetNumCurrent -= 1

    inst.freeList += need_removal

    for index, item in enumerate(inst.activeList):
        if item in need_removal:
            inst.activeList.pop(index)


# This is a MODULE level report function. The function is called by
# external step function to obtain unit level data
def module_report(inst, t, t_num):
    num = 0
    for i in inst.activeList:
        uid = i.data
        gtrack_unit.unit_report(inst.hTrack[uid], t[num])
        num += 1
    t_num[0] = num


#      Algorithm level step funtion
#      Application shall call this function to process one frame of measurements with a given instance of the algorithm
#
#  @param[in]  handle
#      Handle to GTRACK module
#  @param[in]  point
#      Pointer to an array of input measurments. Each measurement has range/angle/radial velocity information
#  @param[in]  var
#      Pointer to an array of input measurment variances. Shall be set to NULL if variances are unknown
#  @param[in]  mNum
#      Number of input measurements
#  @param[out]  t
#      Pointer to an array of \ref gtrack_targetDesc. Application shall provide
#      sufficient space for the expected number of targets.
#      This function populates the descritions for each of the tracked target
#  @param[out]  t_num
#      Pointer to a uint16_t value.
#      Function returns a number of populated target descriptos
#  @param[out]  mIndex
#      Pointer to an array of uint8_t indices. Application shall provide sufficient
#      space to index all measurment points.
#      This function populates target indices, indicating which tracking ID was assigned to each measurment.
#      See Target ID defeinitions, example \ref gtrack_ID_POINT_NOT_ASSOCIATED
#      Shall be set to NULL when indices aren't required.
#  @param[out]  bench
#      Pointer to an array of benchmarking results. Each result is a 32bit timestamp.
#      The array size shall be \ref gtrack_BENCHMARK_SIZE.
#      This function populates the array with the timestamps of free runing CPU cycles count.
#      Shall be set to NULL when benchmarking isn't required.

def step(handle, point, var, m_num, t, t_num, m_index):
    inst = handle
    inst.heartBeat += 1

    if m_num > inst.maxNumPoints:
        m_num = inst.maxNumPoints

    for n in range(m_num):
        inst.bestScore[n] = sys.float_info.max

        x_pos = point[n].range * np.sin(point[n].angle)
        y_pos = point[n].range * np.cos(point[n].angle)
        if inst.params.sceneryParams.numBoundaryBoxes != 0:
            inst.bestIndex[n] = ekf_utils.gtrack_ID_POINT_BEHIND_THE_WALL
            for numBoxes in range(inst.params.sceneryParams.numBoundaryBoxes):
                if ekf_utils.isPointInsideBox(x_pos, y_pos, inst.params.sceneryParams.boundaryBox[numBoxes]) == 1:
                    inst.bestIndex[n] = ekf_utils.gtrack_ID_POINT_NOT_ASSOCIATED
                    break
        else:
            inst.bestIndex[n] = ekf_utils.gtrack_ID_POINT_NOT_ASSOCIATED

    module_predict(inst)
    module_associate(inst, point, m_num)
    module_allocate(inst, point, m_num)
    module_update(inst, point, var, m_num)
    module_report(inst, t, t_num)

    if m_index != 0:
        for n in range(m_num):
            m_index[n] = inst.bestIndex[n]
