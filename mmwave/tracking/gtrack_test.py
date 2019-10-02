from . import ekf_utils
from . import gtrack_unit

import numpy as np
import copy


class ListElem:
    """Stores a single piece of data

    Attributes:
        data (float or int): Current element data

    """
    def __init__(self):
        self.data = 0


#
#  @b Description
#  @n
#     Algorithm level create funtion.
#       Application calls this function to create an instance of GTRACK algorithm with desired configuration parameters.
#     Function returns a handle, which shall be used to execute a single frame step function, or a delete function
#
#  @param[in]  config
#     This is a pointer to the configuration structure.
#     The structure contains all parameters that are exposed by GTRACK alrorithm.
#     The configuration does not need to persist.
#     Advanced configuration structure can be set to NULL to use the default one.
#     Any field within Advanced configuration can also be set to NULL to use the default values for the field.
#  @param[out] errCode
#      Error code populated on error, see \ref gtrack_ERROR_CODE
#
#  \ingroup gtrack_ALG_EXTERNAL_FUNCTION
#
#  @retval
#      Handle to GTRACK module
#

def create(config):
    if config.maxNumPoints > ekf_utils.gtrack_NUM_POINTS_MAX:
        raise ValueError('maxNumPoints exceeded, create')
    if config.maxNumTracks > ekf_utils.gtrack_NUM_TRACKS_MAX:
        raise ValueError('maxNumTracks exceeded, create')

    inst = ekf_utils.GtrackModuleInstance()

    inst.maxNumPoints = config.maxNumPoints
    inst.maxNumTracks = config.maxNumTracks

    inst.heartBeat = 0

    # default parameters
    inst.params.gatingParams = ekf_utils.gtrack_gatingParams(volume=2., params=[(3., 2., 0.)])
    inst.params.stateParams = ekf_utils.gtrack_stateParams(det2actThre=3, det2freeThre=3, active2freeThre=5,
                                                           static2freeThre=5, exit2freeThre=5)
    inst.params.unrollingParams = ekf_utils.gtrack_unrollingParams(alpha=0.5, confidence=0.1)
    inst.params.allocationParams = ekf_utils.gtrack_allocationParams(snrThre=100., velocityThre=0.5, pointsThre=5,
                                                                     maxDistanceThre=1., maxVelThre=2.)
    inst.params.variationParams = ekf_utils.gtrack_varParams(lengthStd=np.float32(1. / 3.46),
                                                             widthStd=np.float32(1. / 3.46), dopplerStd=2.)
    inst.params.sceneryParams = ekf_utils.gtrack_sceneryParams(numBoundaryBoxes=0, numStaticBoxes=0,
                                                               bound_box=[(0., 0., 0., 0.), (0., 0., 0., 0.)],
                                                               static_box=[(0., 0., 0., 0.), (0., 0., 0., 0.)])

    # user overwrites default parameters
    if config.advParams is not None:
        if config.advParams.gatingParams is not None:
            inst.params.gatingParams = copy.deepcopy(config.advParams.gatingParams)
        if config.advParams.stateParams is not None:
            inst.params.stateParams = copy.deepcopy(config.advParams.stateParams)
        if config.advParams.unrollingParams is not None:
            inst.params.unrollingParams = copy.deepcopy(config.advParams.unrollingParams)
        if config.advParams.allocationParams is not None:
            inst.params.allocationParams = copy.deepcopy(config.advParams.allocationParams)
        if config.advParams.variationParams is not None:
            inst.params.variationParams = copy.deepcopy(config.advParams.variationParams)
        if config.advParams.sceneryParams is not None:
            inst.params.sceneryParams = copy.deepcopy(config.advParams.sceneryParams)

    # pre-configured parameters
    inst.params.stateVectorType = config.stateVectorType
    inst.params.deltaT = config.deltaT
    inst.params.maxAcceleration = config.maxAcceleration
    inst.params.maxRadialVelocity = config.maxRadialVelocity
    inst.params.radialVelocityResolution = config.radialVelocityResolution
    inst.params.initialRadialVelocity = config.initialRadialVelocity

    if config.verbose == ekf_utils.gtrack_VERBOSE_TYPE().gtrack_VERBOSE_NONE:
        inst.params.verbose = 0
    elif config.verbose == ekf_utils.gtrack_VERBOSE_TYPE().gtrack_VERBOSE_ERROR:
        inst.params.verbose = ekf_utils.VERBOSE_ERROR_INFO
    elif config.verbose == ekf_utils.gtrack_VERBOSE_TYPE().gtrack_VERBOSE_WARNING:
        inst.params.verbose = (ekf_utils.VERBOSE_ERROR_INFO) | (ekf_utils.VERBOSE_WARNING_INFO)
    else:
        if config.verbose == ekf_utils.gtrack_VERBOSE_TYPE().gtrack_VERBOSE_DEBUG:
            inst.params.verbose = (ekf_utils.VERBOSE_ERROR_INFO) | (ekf_utils.VERBOSE_WARNING_INFO) | (
                ekf_utils.VERBOSE_DEBUG_INFO) | (ekf_utils.VERBOSE_UNROLL_INFO) | (ekf_utils.VERBOSE_STATE_INFO)
        elif config.verbose == ekf_utils.gtrack_VERBOSE_TYPE().gtrack_VERBOSE_MATRIX:
            inst.params.verbose = (ekf_utils.VERBOSE_ERROR_INFO) | (ekf_utils.VERBOSE_WARNING_INFO) | (
                ekf_utils.VERBOSE_DEBUG_INFO) | (ekf_utils.VERBOSE_MATRIX_INFO)
        elif config.verbose == ekf_utils.gtrack_VERBOSE_TYPE().gtrack_VERBOSE_MAXIMUM:
            inst.params.verbose = (ekf_utils.VERBOSE_ERROR_INFO) | (ekf_utils.VERBOSE_WARNING_INFO) | (
                ekf_utils.VERBOSE_DEBUG_INFO) | (ekf_utils.VERBOSE_MATRIX_INFO) | (ekf_utils.VERBOSE_UNROLL_INFO) | (
                                      ekf_utils.VERBOSE_STATE_INFO) | (ekf_utils.VERBOSE_ASSOSIATION_INFO)

    dt = np.float32(config.deltaT)
    dt2 = np.float32(np.power(dt, 2))
    dt3 = np.float32(np.power(dt, 3))
    dt4 = np.float32(np.power(dt, 4))

    f4 = np.array([1., 0., dt, 0.,
                   0., 1., 0., dt,
                   0., 0., 1., 0.,
                   0., 0., 0., 1.], dtype=np.float32)

    q4 = np.array([dt4 / 4, 0., dt3 / 2, 0.,
                   0., dt4 / 4, 0., dt3 / 2,
                   dt3 / 2, 0., dt2, 0.,
                   0., dt3 / 2, 0., dt2], dtype=np.float32)

    f6 = np.array([1., 0., dt, 0., dt2 / 2, 0.,
                   0., 1., 0., dt, 0., dt2 / 2,
                   0., 0., 1., 0., dt, 0.,
                   0., 0., 0., 1., 0., dt,
                   0., 0., 0., 0., 1., 0.,
                   0., 0., 0., 0., 0., 1.], dtype=np.float32)

    q6 = np.array([dt4 / 4, 0., dt3 / 2, 0., dt2 / 2, 0.,
                   0., dt4 / 4, 0., dt3 / 2, 0., dt2 / 2,
                   dt3 / 2, 0., dt2, 0., dt, 0.,
                   0., dt3 / 2, 0., dt2, 0., dt,
                   dt2 / 2, 0., dt, 0., 1., 0.,
                   0., dt2 / 2, 0., dt, 0., 1.], dtype=np.float32)

    inst.params.F4 = copy.deepcopy(f4)
    inst.params.Q4 = copy.deepcopy(q4)
    inst.params.F6 = copy.deepcopy(f6)
    inst.params.Q6 = copy.deepcopy(q6)

    inst.hTrack = [ekf_utils.GtrackUnitInstance() for _ in range(inst.maxNumTracks)]

    inst.bestScore = np.array([0. for _ in range(inst.maxNumPoints)], dtype=np.float32)

    inst.bestIndex = np.array([0 for _ in range(inst.maxNumPoints)], dtype=np.uint8)

    inst.allocIndex = np.array([0 for _ in range(inst.maxNumPoints)], dtype=np.uint16)

    inst.uidElem = [ListElem() for _ in range(inst.maxNumTracks)]

    inst.targetNumTotal = 0
    inst.targetNumCurrent = 0

    for uid in range(inst.maxNumTracks):
        inst.uidElem[uid].data = uid
        inst.freeList.append(inst.uidElem[uid])

        inst.params.uid = uid
        inst.hTrack[uid] = gtrack_unit.unit_create(inst.params)

    return inst
