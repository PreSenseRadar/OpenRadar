import numpy as np
import copy

from . import ekf_utils

gtrack_MIN_DISPERSION_ALPHA = 0.1
gtrack_EST_POINTS = 10
gtrack_MIN_POINTS_TO_UPDATE_DISPERSION = 3
gtrack_KNOWN_TARGET_POINTS_THRESHOLD = 50


# GTRACK Module calls this function to instantiate GTRACK Unit with desired configuration parameters. 
# Function returns a handle, which is used my module to call units' methods

def unit_create(params):
    inst = ekf_utils.GtrackUnitInstance()

    inst.gatingParams = params.gatingParams
    inst.stateParams = params.stateParams
    inst.allocationParams = params.allocationParams
    inst.unrollingParams = params.unrollingParams
    inst.variationParams = params.variationParams
    inst.sceneryParams = params.sceneryParams

    inst.uid = params.uid
    inst.maxAcceleration = params.maxAcceleration
    inst.maxRadialVelocity = params.maxRadialVelocity
    inst.radialVelocityResolution = params.radialVelocityResolution
    inst.verbose = params.verbose
    inst.initialRadialVelocity = params.initialRadialVelocity

    inst.F4 = params.F4
    inst.Q4 = params.Q4
    inst.F6 = params.F6
    inst.Q6 = params.Q6

    if params.stateVectorType == ekf_utils.gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2DA:
        inst.stateVectorType = ekf_utils.gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2DA
        inst.stateVectorLength = 6
        inst.measurementVectorLength = 3
    else:
        raise ValueError('not supported, unit_create')

    inst.dt = params.deltaT
    inst.state = ekf_utils.TrackState().TRACK_STATE_FREE

    return inst


# GTRACK Module calls this function to run GTRACK unit prediction step 
def unit_predict(handle):
    inst = handle
    inst.heartBeatCount += 1
    temp1 = np.zeros(shape=(36,), dtype=np.float32)
    temp2 = np.zeros(shape=(36,), dtype=np.float32)
    temp3 = np.zeros(shape=(36,), dtype=np.float32)

    # Current state vector length
    sLen = inst.stateVectorLength

    if inst.processVariance != 0:
        inst.S_apriori_hat = ekf_utils.gtrack_matrixMultiply(sLen, sLen, 1, inst.F, inst.S_hat)
        temp1 = ekf_utils.gtrack_matrixMultiply(6, 6, 6, inst.F, inst.P_hat)
        temp2 = ekf_utils.gtrack_matrixTransposeMultiply(6, 6, 6, temp1, inst.F)
        temp1 = ekf_utils.gtrack_matrixScalerMultiply(sLen, sLen, inst.Q, inst.processVariance)
        temp3 = ekf_utils.gtrack_matrixAdd(sLen, sLen, temp1, temp2)

        inst.P_apriori_hat = ekf_utils.gtrack_matrixMakeSymmetrical(sLen, temp3)
    else:
        inst.S_apriori_hat = copy.deepcopy(inst.S_hat)
        inst.P_apriori_hat = copy.deepcopy(inst.P_hat)

    ekf_utils.gtrack_cartesian2spherical(inst.stateVectorType, inst.S_apriori_hat, inst.H_s)


# GTRACK Module calls this function to obtain the measurement vector scoring from the GTRACK unit perspective
def unit_score(handle, point, best_score, best_ind, num):
    limits = np.zeros(shape=(3,), dtype=np.float32)
    u_tilda = np.zeros(shape=(3,), dtype=np.float32)

    inst = handle

    limits[0] = inst.gatingParams.limits[0].length
    limits[1] = inst.gatingParams.limits[0].width
    limits[2] = inst.gatingParams.limits[0].vel

    if inst.processVariance == 0:
        inst.G = 1
    else:
        inst.G = ekf_utils.gtrack_gateCreateLim(inst.gatingParams.volume, inst.gC_inv, inst.H_s[0], limits)

    det = ekf_utils.gtrack_matrixDet3(inst.gC)

    log_det = np.float32(np.log(det))

    for n in range(num):
        if best_ind[n] == ekf_utils.gtrack_ID_POINT_BEHIND_THE_WALL:
            continue

        u_tilda[0] = np.float32(point[n].range - inst.H_s[0])
        u_tilda[1] = np.float32(point[n].angle - inst.H_s[1])

        if inst.velocityHandling < ekf_utils.VelocityHandlingState().VELOCITY_LOCKED:
            # Radial velocity estimation is not yet known, unroll based on velocity measured at allocation time
            rv_out = ekf_utils.gtrack_unrollRadialVelocity(inst.maxRadialVelocity, inst.allocationVelocity,
                                                          point[n].doppler)
            u_tilda[2] = np.float32(rv_out - inst.allocationVelocity)
        else:
            # Radial velocity estimation is known 
            rv_out = ekf_utils.gtrack_unrollRadialVelocity(inst.maxRadialVelocity, inst.H_s[2], point[n].doppler)
            u_tilda[2] = np.float32(rv_out - inst.H_s[2])

        chi2 = ekf_utils.gtrack_computeMahalanobis3(u_tilda, inst.gC_inv)
        # print(inst.gC_inv)

        if chi2 < inst.G:
            score = np.float32(log_det + chi2)
            if score < best_score[n]:
                best_score[n] = score
                best_ind[n] = np.uint8(inst.uid)
                point[n].doppler = rv_out


# GTRACK Module calls this function to start target tracking. This function is called during modules' allocation step,
# once new set of points passes allocation thresholds 
def unit_start(handle, time_stamp, tid, um):
    inst = handle

    m = np.zeros(shape=(3,), dtype=np.float32)

    inst.tid = tid
    inst.heartBeatCount = time_stamp
    inst.allocationTime = time_stamp
    inst.allocationRange = um[0]
    inst.allocationVelocity = um[2]
    inst.associatedPoints = 0

    inst.state = ekf_utils.TrackState().TRACK_STATE_DETECTION
    inst.currentStateVectorType = ekf_utils.gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2DA
    inst.stateVectorLength = 6

    inst.processVariance = (0.5 * inst.maxAcceleration) * (0.5 * inst.maxAcceleration)

    inst.F = inst.F6
    inst.Q = inst.Q6

    inst.velocityHandling = ekf_utils.VelocityHandlingState().VELOCITY_INIT

    m[2] = ekf_utils.gtrack_unrollRadialVelocity(inst.maxRadialVelocity, inst.initialRadialVelocity, um[2])

    inst.rangeRate = m[2]

    m[0] = um[0]
    m[1] = um[1]

    ekf_utils.gtrack_spherical2cartesian(inst.currentStateVectorType, m, inst.S_apriori_hat)
    inst.H_s = copy.deepcopy(m)

    inst.P_apriori_hat = copy.deepcopy(ekf_utils.pinit6x6)
    inst.gD = copy.deepcopy(ekf_utils.zero3x3)
    inst.G = 1.


# GTRACK Module calls this function to perform an update step for the tracking unit. 
def unit_update(handle, point, var, pInd, num):
    J = np.zeros(shape=(18,), dtype=np.float32)  # 3x6
    PJ = np.zeros(shape=(18,), dtype=np.float32)  # 6x3
    JPJ = np.zeros(shape=(9,), dtype=np.float32)  # 3x3
    U = np.zeros(shape=(3,), dtype=np.float32)
    u_tilda = np.zeros(shape=(3,), dtype=np.float32)
    cC = np.zeros(shape=(9,), dtype=np.float32)
    cC_inv = np.zeros(shape=(9,), dtype=np.float32)
    K = np.zeros(shape=(18,), dtype=np.float32)  # 6x3

    u_mean = ekf_utils.gtrack_measurementPoint()

    D = np.zeros(shape=(9,), dtype=np.float32)
    Rm = np.zeros(shape=(9,), dtype=np.float32)
    Rc = np.zeros(shape=(9,), dtype=np.float32)

    temp1 = np.zeros(shape=(36,), dtype=np.float32)

    inst = handle
    mlen = inst.measurementVectorLength
    slen = inst.stateVectorLength

    myPointNum = 0

    for n in range(num):
        if pInd[n] == inst.uid:
            myPointNum += 1
            u_mean.range += point[n].range
            u_mean.angle += point[n].angle

            if var != None:
                Rm[0] += var[n].rangeVar
                Rm[4] += var[n].angleVar
                Rm[8] += var[n].dopplerVar

            if myPointNum == 1:
                rvPilot = point[n].doppler
                u_mean.doppler = rvPilot
            else:
                rvCurrent = ekf_utils.gtrack_unrollRadialVelocity(inst.maxRadialVelocity, rvPilot, point[n].doppler)
                point[n].doppler = rvCurrent
                u_mean.doppler += rvCurrent

    if myPointNum == 0:
        # INACTIVE
        if (np.abs(inst.S_hat[2]) < inst.radialVelocityResolution) and \
                (np.abs(inst.S_hat[3]) < inst.radialVelocityResolution):
            inst.S_hat = np.zeros(shape=(inst.S_hat.shape), dtype=np.float32)

            inst.S_hat[0] = inst.S_apriori_hat[0]
            inst.S_hat[1] = inst.S_apriori_hat[1]

            inst.P_hat = copy.deepcopy(inst.P_apriori_hat)

            inst.processVariance = 0
        else:
            inst.S_hat = copy.deepcopy(inst.S_apriori_hat)
            inst.P_hat = copy.deepcopy(inst.P_apriori_hat)

        unit_event(inst, myPointNum)
        return inst.state

    inst.associatedPoints += myPointNum

    if inst.processVariance == 0:
        inst.processVariance = np.float32((0.5 * (inst.maxAcceleration)) * (0.5 * (inst.maxAcceleration)))

    u_mean.range = np.float32(u_mean.range / myPointNum)
    u_mean.angle = np.float32(u_mean.angle / myPointNum)
    u_mean.doppler = np.float32(u_mean.doppler / myPointNum)

    if var != None:
        Rm[0] = np.float32(Rm[0] / myPointNum)
        Rm[4] = np.float32(Rm[4] / myPointNum)
        Rm[8] = np.float32(Rm[8] / myPointNum)
    else:
        dRangeVar = np.float32((inst.variationParams.lengthStd) * (inst.variationParams.lengthStd))
        dDopplerVar = np.float32((inst.variationParams.dopplerStd) * (inst.variationParams.dopplerStd))

        Rm[0] = dRangeVar
        angleStd = np.float32(2 * np.float32(np.arctan(0.5 * (inst.variationParams.widthStd) / inst.H_s[0])))
        Rm[4] = angleStd * angleStd
        Rm[8] = dDopplerVar

    U[0] = u_mean.range
    U[1] = u_mean.angle
    U[2] = u_mean.doppler

    velocity_state_handling(inst, U)

    if myPointNum > gtrack_MIN_POINTS_TO_UPDATE_DISPERSION:
        for n in range(num):
            if pInd[n] == inst.uid:
                D[0] += np.float32((point[n].range - u_mean.range) * (point[n].range - u_mean.range))
                D[4] += np.float32((point[n].angle - u_mean.angle) * (point[n].angle - u_mean.angle))
                D[8] += np.float32((point[n].doppler - u_mean.doppler) * (point[n].doppler - u_mean.doppler))
                D[1] += np.float32((point[n].range - u_mean.range) * (point[n].angle - u_mean.angle))
                D[2] += np.float32((point[n].range - u_mean.range) * (point[n].doppler - u_mean.doppler))
                D[5] += np.float32((point[n].angle - u_mean.angle) * (point[n].doppler - u_mean.doppler))

        D[0] = np.float32(D[0] / myPointNum)
        D[4] = np.float32(D[4] / myPointNum)
        D[8] = np.float32(D[8] / myPointNum)
        D[1] = np.float32(D[1] / myPointNum)
        D[2] = np.float32(D[2] / myPointNum)
        D[5] = np.float32(D[5] / myPointNum)

        alpha = np.float32(myPointNum / (inst.associatedPoints))
        # print(alpha)
        if alpha < gtrack_MIN_DISPERSION_ALPHA:
            alpha = gtrack_MIN_DISPERSION_ALPHA

        inst.gD[0] = np.float32((1. - alpha) * inst.gD[0] + alpha * D[0])
        inst.gD[1] = np.float32((1. - alpha) * inst.gD[1] + alpha * D[1])
        inst.gD[2] = np.float32((1. - alpha) * inst.gD[2] + alpha * D[2])
        inst.gD[3] = np.float32(inst.gD[1])
        inst.gD[4] = np.float32((1. - alpha) * inst.gD[4] + alpha * D[4])
        inst.gD[5] = np.float32((1. - alpha) * inst.gD[5] + alpha * D[5])
        inst.gD[6] = np.float32(inst.gD[2])
        inst.gD[7] = np.float32(inst.gD[5])
        inst.gD[8] = np.float32((1. - alpha) * inst.gD[8] + alpha * D[8])

    if myPointNum > gtrack_EST_POINTS:
        alpha = 0
    else:
        alpha = np.float32((gtrack_EST_POINTS - myPointNum) / ((gtrack_EST_POINTS - 1) * myPointNum))

    Rc[0] = np.float32((Rm[0] / myPointNum) + alpha * (inst.gD[0]))
    Rc[4] = np.float32((Rm[4] / myPointNum) + alpha * (inst.gD[4]))
    Rc[8] = np.float32((Rm[8] / myPointNum) + alpha * (inst.gD[8]))

    ekf_utils.gtrack_computeJacobian(inst.currentStateVectorType, inst.S_apriori_hat, J)

    u_tilda = ekf_utils.gtrack_matrixSub(mlen, 1, U, inst.H_s)
    PJ = ekf_utils.gtrack_matrixComputePJT(inst.P_apriori_hat, J)
    JPJ = ekf_utils.gtrack_matrixMultiply(mlen, slen, mlen, J, PJ)
    cC = ekf_utils.gtrack_matrixAdd(mlen, mlen, JPJ, Rc)

    cC_inv = ekf_utils.gtrack_matrixInv3(cC)

    K = ekf_utils.gtrack_matrixMultiply(slen, mlen, mlen, PJ, cC_inv)

    temp1 = ekf_utils.gtrack_matrixMultiply(slen, mlen, 1, K, u_tilda)
    inst.S_hat = ekf_utils.gtrack_matrixAdd(slen, 1, inst.S_apriori_hat, temp1)
    # print(temp1)

    temp1 = ekf_utils.gtrack_matrixTransposeMultiply(slen, mlen, slen, K, PJ)
    inst.P_hat = ekf_utils.gtrack_matrixSub(slen, slen, inst.P_apriori_hat, temp1)

    temp1 = ekf_utils.gtrack_matrixAdd(mlen, mlen, JPJ, Rm)
    inst.gC = ekf_utils.gtrack_matrixAdd(mlen, mlen, temp1, inst.gD)

    inst.gC_inv = ekf_utils.gtrack_matrixInv3(inst.gC)

    unit_event(inst, myPointNum)
    return inst.state


# this is the helper function for GTRACK unit update
def velocity_state_handling(handle, um):
    inst = handle
    rvIn = um[2]
    # print(inst.velocityHandling)

    if inst.velocityHandling == ekf_utils.VelocityHandlingState().VELOCITY_INIT:
        um[2] = inst.rangeRate
        inst.velocityHandling = ekf_utils.VelocityHandlingState().VELOCITY_RATE_FILTER
    elif inst.velocityHandling == ekf_utils.VelocityHandlingState().VELOCITY_RATE_FILTER:
        instanteneousRangeRate = np.float32(
            (um[0] - inst.allocationRange) / ((inst.heartBeatCount - inst.allocationTime) * (inst.dt)))
        inst.rangeRate = np.float32((inst.unrollingParams.alpha) * (inst.rangeRate) + (
                1 - (inst.unrollingParams.alpha)) * instanteneousRangeRate)
        um[2] = ekf_utils.gtrack_unrollRadialVelocity(inst.maxRadialVelocity, inst.rangeRate, rvIn)

        rrError = np.float32((instanteneousRangeRate - inst.rangeRate) / inst.rangeRate)

        if np.abs(rrError) < inst.unrollingParams.confidence:
            inst.velocityHandling = ekf_utils.VelocityHandlingState().VELOCITY_TRACKING
    elif inst.velocityHandling == ekf_utils.VelocityHandlingState().VELOCITY_TRACKING:
        instanteneousRangeRate = np.float32(
            (um[0] - inst.allocationRange) / ((inst.heartBeatCount - inst.allocationTime) * inst.dt))

        inst.rangeRate = np.float32(
            (inst.unrollingParams.alpha) * inst.rangeRate + (1 - inst.unrollingParams.alpha) * instanteneousRangeRate)
        um[2] = ekf_utils.gtrack_unrollRadialVelocity(inst.maxRadialVelocity, inst.rangeRate, rvIn)
        rvError = np.float32((inst.H_s[2] - um[2]) / um[2])
        if np.abs(rvError) < 0.1:
            inst.velocityHandling = ekf_utils.VelocityHandlingState().VELOCITY_LOCKED
    elif inst.velocityHandling == ekf_utils.VelocityHandlingState().VELOCITY_LOCKED:
        um[2] = ekf_utils.gtrack_unrollRadialVelocity(inst.maxRadialVelocity, inst.H_s[2], um[2])


# GTRACK Module calls this function to run GTRACK unit level state machine
def unit_event(handle, num):
    inst = handle

    if inst.state == ekf_utils.TrackState().TRACK_STATE_DETECTION:
        if num > inst.allocationParams.pointsThre:
            inst.detect2freeCount = 0
            inst.detect2activeCount += 1
            if inst.detect2activeCount > inst.stateParams.det2actThre:
                inst.state = ekf_utils.TrackState().TRACK_STATE_ACTIVE
        else:
            if num == 0:
                inst.detect2freeCount += 1
                if inst.detect2activeCount > 0:
                    inst.detect2activeCount -= 1
                if inst.detect2freeCount > inst.stateParams.det2freeThre:
                    inst.state = ekf_utils.TrackState().TRACK_STATE_FREE
    elif inst.state == ekf_utils.TrackState().TRACK_STATE_ACTIVE:
        if num != 0:
            inst.active2freeCount = 0
        else:
            inst.active2freeCount += 1

            if inst.sceneryParams.numStaticBoxes != 0:
                thre = inst.stateParams.exit2freeThre
                for numBoxes in range(inst.sceneryParams.numStaticBoxes):
                    if ekf_utils.isPointInsideBox(inst.S_hat[0], inst.S_hat[1],
                                                  inst.sceneryParams.boundaryBox[numBoxes]) == 1:
                        if inst.processVariance == 0:
                            thre = inst.stateParams.static2freeThre
                        else:
                            thre = inst.stateParams.active2freeThre
                        break
            else:
                thre = inst.stateParams.active2freeThre

            if thre > inst.heartBeatCount:
                thre = np.uint16(inst.heartBeatCount)

            if inst.active2freeCount > thre:
                inst.state = ekf_utils.TrackState().TRACK_STATE_FREE


# GTRACK Module calls this function to report GTRACK unit results to the target descriptor
def unit_report(handle, target):
    inst = handle

    target.uid = inst.uid
    target.tid = inst.tid

    target.S = copy.deepcopy(inst.S_hat)
    target.EC = copy.deepcopy(inst.gC_inv)
    target.G = inst.G
