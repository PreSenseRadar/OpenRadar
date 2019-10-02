import numpy as np

'''global constants'''

# Maximum supported configurations
gtrack_NUM_POINTS_MAX = 1000
gtrack_NUM_TRACKS_MAX = 250

# Target ID definitions
gtrack_ID_POINT_TOO_WEAK = 253
gtrack_ID_POINT_BEHIND_THE_WALL = 254
gtrack_ID_POINT_NOT_ASSOCIATED = 255

# Boundary boxes
gtrack_MAX_BOUNDARY_BOXES = 2
gtrack_MAX_STATIC_BOXES = 2

MAXNUMBERMEASUREMENTS = 800
MAXNUMBERTRACKERS = 20

zero3x3 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)

pinit6x6 = np.array([0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., 0., 0.5, 0., 0., 0.,
                     0., 0., 0., 0.5, 0., 0.,
                     0., 0., 0., 0., 1., 0.,
                     0., 0., 0., 0., 0., 1.], dtype=np.float32)

VERBOSE_ERROR_INFO = 0x00000001  # /*!< Report Errors */
VERBOSE_WARNING_INFO = 0x00000002  # /*!< Report Warnings */
VERBOSE_DEBUG_INFO = 0x00000004  # /*!< Report Debuging information */
VERBOSE_MATRIX_INFO = 0x00000008  # /*!< Report Matrix math computations */
VERBOSE_UNROLL_INFO = 0x00000010  # /*!< Report velocity unrolling data */
VERBOSE_STATE_INFO = 0x00000020  # /*!< Report state transitions */
VERBOSE_ASSOSIATION_INFO = 0x00000040  # /*!< Report association data */
VERBOSE_GATEXY_INFO = 0x00000080  # /*!< Report gating in XY space */
VERBOSE_GATERA_INFO = 0x00000100  # /*!< Report gating in range/angle space */
VERBOSE_GATEG1_INFO = 0x00000200  # /*!< Report unitary gating */

'''below is the gtrack alg configuration params'''


# GTRACK Box Structure
class gtrack_boundaryBox():
    def __init__(self, left, right, bottom, top):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top


# GTRACK Scene Parameters
class gtrack_sceneryParams():
    def __init__(self, numBoundaryBoxes=0, numStaticBoxes=0, bound_box=[(0, 0, 0, 0), (0, 0, 0, 0)],
                 static_box=[(0, 0, 0, 0), (0, 0, 0, 0)]):
        self.numBoundaryBoxes = numBoundaryBoxes
        self.boundaryBox = [gtrack_boundaryBox(*bound) for bound, _ in zip(bound_box, range(gtrack_MAX_BOUNDARY_BOXES))]
        self.numStaticBoxes = numStaticBoxes
        self.staticBox = [gtrack_boundaryBox(*bound) for bound, _ in zip(static_box, range(gtrack_MAX_STATIC_BOXES))]


# GTRACK Gate Limits
class gtrack_gateLimits():
    def __init__(self, length, width, vel):
        self.length = length
        self.width = width
        self.vel = vel


# GTRACK Gating Function Parameters
class gtrack_gatingParams():
    def __init__(self, volume=2, params=[(3, 2, 0)]):
        self.volume = volume
        self.limits = [gtrack_gateLimits(i, j, k) for (i, j, k) in params]


# GTRACK Tracking Management Function Parameters
class gtrack_stateParams():
    def __init__(self, det2actThre=3, det2freeThre=3, active2freeThre=5, static2freeThre=5, exit2freeThre=5):
        self.det2actThre = det2actThre
        self.det2freeThre = det2freeThre
        self.active2freeThre = active2freeThre
        self.static2freeThre = static2freeThre
        self.exit2freeThre = exit2freeThre


# GTRACK Update Function Parameters
class gtrack_varParams():
    def __init__(self, lengthStd=np.float32(1 / 3.46), widthStd=np.float32(1 / 3.46), dopplerStd=2.):
        self.lengthStd = lengthStd
        self.widthStd = widthStd
        self.dopplerStd = dopplerStd


# GTRACK Allocation Function Parameters
class gtrack_allocationParams():
    def __init__(self, snrThre=100., velocityThre=0.5, pointsThre=5, maxDistanceThre=1., maxVelThre=2.):
        self.snrThre = snrThre
        self.velocityThre = velocityThre
        self.pointsThre = pointsThre
        self.maxDistanceThre = maxDistanceThre
        self.maxVelThre = maxVelThre


# GTRACK Unrolling Parameters
class gtrack_unrollingParams():
    def __init__(self, alpha=0.5, confidence=0.1):
        self.alpha = alpha
        self.confidence = confidence


# GTRACK State Vector
class gtrack_STATE_VECTOR_TYPE():
    def __init__(self):
        self.gtrack_STATE_VECTORS_2D = 0
        self.gtrack_STATE_VECTORS_2DA = 1
        self.gtrack_STATE_VECTORS_3D = 2
        self.gtrack_STATE_VECTORS_3DA = 3


# GTRACK Verbose Level
class gtrack_VERBOSE_TYPE():
    def __init__(self):
        self.gtrack_VERBOSE_NONE = 0
        self.gtrack_VERBOSE_ERROR = 1
        self.gtrack_VERBOSE_WARNING = 2
        self.gtrack_VERBOSE_DEBUG = 3
        self.gtrack_VERBOSE_MATRIX = 4
        self.gtrack_VERBOSE_MAXIMUM = 5


# GTRACK Advanced Parameters
class gtrack_advancedParameters():
    def __init__(self):
        self.gatingParams = gtrack_gatingParams()
        self.allocationParams = gtrack_allocationParams()
        self.unrollingParams = gtrack_unrollingParams()
        self.stateParams = gtrack_stateParams()
        self.variationParams = gtrack_varParams()
        self.sceneryParams = gtrack_sceneryParams()


# GTRACK Configuration
class gtrack_moduleConfig():
    def __init__(self):
        self.stateVectorType = gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2DA
        self.verbose = gtrack_VERBOSE_TYPE().gtrack_VERBOSE_NONE
        self.maxNumPoints = MAXNUMBERMEASUREMENTS
        self.maxNumTracks = MAXNUMBERTRACKERS
        self.initialRadialVelocity = 0
        self.maxRadialVelocity = 20
        self.radialVelocityResolution = 0
        self.maxAcceleration = 12
        self.deltaT = 0.4
        self.advParams = gtrack_advancedParameters()


# GTRACK Measurement point
class gtrack_measurementPoint():
    def __init__(self):
        self.range = 0.
        self.angle = 0.
        self.doppler = 0.
        self.snr = 0.


# GTRACK Measurement variances
class gtrack_measurementVariance():
    def __init__(self):
        self.rangeVar = 0
        self.angleVar = 0
        self.dopplerVar = 0


# GTRACK target descriptor
class gtrack_targetDesc():
    def __init__(self):
        self.uid = 0
        self.tid = 0
        self.S = np.zeros(shape=(6,), dtype=np.float32)
        self.EC = np.zeros(shape=(9,), dtype=np.float32)
        self.G = 0

# /**
# *  @b Description
# *  @n
# *     This function is used to force matrix symmetry by averaging off-diagonal elements
# *     Matrices are squared, real, single precision floating point.
# *     Matrices are in row-major order
# *
# *  @param[in]  m (m=rows=cols)
# *     Number of rows and cols
# *  @param[in]  A
# *     Matrix A
# *  @param[out]  B
# *     Matrix B
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixMakeSymmetrical(m, A, B):
    A = A.reshape(m, m)

    B = np.squeeze((1/2 * np.add(A, A.T)).reshape(1, -1))

    A = np.squeeze(A.reshape(1, -1))
'''


def gtrack_matrixMakeSymmetrical(m, A):
    A = A.reshape(m, m)

    B = np.squeeze((1 / 2 * np.add(A, A.T)).reshape(1, -1))

    return B
    # i = j = 0
    # B = np.zeros_like(A, dtype = np.float32)
    # i = j = 0
    # for i in range(0, m - 1):
    #     B[i*m + i] = A[i*m + i]
    #     for j in range(i+1, m):
    #         B[i*m+j] = B[j*m+i] = 0.5 * (A[i*m+j] + A[j*m+i])
    # B[(i+1)*m+(i+1)] = A[(i+1)*m+(i+1)]
    # return B


# /**
# *  @b Description
# *  @n
# *     This function is used to multiply two matrices. 
# *     Matrices are all real, single precision floating point.
# *     Matrices are in row-major order
# *
# *  @param[in]  rows
# *     Outer dimension, number of rows
# *  @param[in]  m
# *     Inner dimension
# *  @param[in]  cols
# *     Outer dimension, number of cols
# *  @param[in]  A
# *     Matrix A
# *  @param[in]  B
# *     Matrix B
# *  @param[out]  C
# *     Matrix C(rows,cols) = A(rows,m) X B(m,cols)
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixMultiply(rows, m, cols, A, B, C):
    A = A.reshape(rows, m)
    B = B.reshape(m, cols)

    C = np.squeeze(np.dot(A, B).reshape(1, -1))

    A = np.squeeze(A.reshape(1, -1))
    B = np.squeeze(B.reshape(1, -1))
'''


def gtrack_matrixMultiply(rows, m, cols, A, B):
    A = A.reshape(rows, m)
    B = B.reshape(m, cols)

    C = np.squeeze(np.dot(A, B).reshape(1, -1))

    # A = np.squeeze(A.reshape(1, -1))
    # B = np.squeeze(B.reshape(1, -1))
    return np.float32(C)


# /**
# *  @b Description
# *  @n
# *     This function is used to multiply two matrices. Second Matrix is getting transposed first
# *     Matrices are all real, single precision floating point.
# *     Matrices are in row-major order
# *
# *  @param[in]  rows
# *     Outer dimension, number of rows
# *  @param[in]  m
# *     Inner dimension
# *  @param[in]  cols
# *     Outer dimension, number of cols
# *  @param[in]  A
# *     Matrix A
# *  @param[in]  B
# *     Matrix B
# *  @param[out]  C
# *     Matrix C(rows,cols) = A(rows,m) X B(cols,m)T
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixTransposeMultiply(rows, m, cols, A, B, C):
    A = A.reshape(rows, m)
    B = B.reshape(cols, m)

    C = np.squeeze(np.dot(A, B.T).reshape(1, -1))

    A = np.squeeze(A.reshape(1, -1))
    B = np.squeeze(B.reshape(1, -1))
'''


def gtrack_matrixTransposeMultiply(rows, m, cols, A, B):
    A = A.reshape(rows, m)
    B = B.reshape(cols, m)

    C = np.squeeze(np.dot(A, B.T).reshape(1, -1))

    # A = np.squeeze(A.reshape(1, -1))
    # B = np.squeeze(B.reshape(1, -1))
    return np.float32(C)


# /**
# *  @b Description
# *  @n
# *     This function is used to multiply two matrices.
# *     First matrix P is of size 6x6, the second one is of the size 3x6.
# *     The second matrix is being transposed first.
# *     Matrices are all real, single precision floating point.
# *     Matrices are in row-major order
# *
# *  @param[in]  P
# *     Matrix P
# *  @param[in]  J
# *     Matrix J
# *  @param[out]  PJ
# *     Matrix PJ = P(6,6) X J(3,6)T
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixComputePJT(P, J, PJ):
    P = P.reshape(6, 6)
    J = J.reshape(3, 6)

    PJ = np.squeeze(np.dot(P, J.T).reshape(1, -1))

    P = np.squeeze(P.reshape(1, -1))
    J = np.squeeze(J.reshape(1, -1))
'''


def gtrack_matrixComputePJT(P, J):
    P = P.reshape(6, 6)
    J = J.reshape(3, 6)

    PJ = np.squeeze(np.dot(P, J.T).reshape(1, -1))

    # P = np.squeeze(P.reshape(1, -1))
    # J = np.squeeze(J.reshape(1, -1))
    return np.float32(PJ)


# /**
# *  @b Description
# *  @n
# *     This function is used to multiply matrix by a scaller.
# *     Matrices are all real, single precision floating point.
# *     Matrices are in row-major order
# *
# *  @param[in]  rows
# *     Number of rows
# *  @param[in]  cols
# *     Number of cols
# *  @param[in]  A
# *     Matrix A
# *  @param[in]  C
# *     Scaller C
# *  @param[out]  B
# *     Matrix B(rows,cols) = A(rows,cols) X C
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixScalerMultiply(rows, cols, A, C, B):
    A = A.reshape(rows, cols)

    B = np.squeeze(np.dot(C, A).reshape(1, -1))

    A = np.squeeze(A.reshape(1, -1))
'''


def gtrack_matrixScalerMultiply(rows, cols, A, C):
    A = A.reshape(rows, cols)

    B = np.squeeze(np.dot(C, A).reshape(1, -1))

    # A = np.squeeze(A.reshape(1, -1))
    return np.float32(B)


# /**
# *  @b Description
# *  @n
# *     This function is used to add two matrices.
# *     Matrices are all real, single precision floating point.
# *     Matrices are in row-major order
# *
# *  @param[in]  rows
# *     Number of rows
# *  @param[in]  cols
# *     Number of cols
# *  @param[in]  A
# *     Matrix A
# *  @param[in]  B
# *     Matrix B
# *  @param[out]  C
# *     Matrix C(rows,cols) = A(rows,cols) + B(rows,cols)
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixAdd(rows, cols, A, B, C):
    A = A.reshape(rows, cols)
    B = B.reshape(rows, cols)

    C = np.squeeze(np.add(A, B).reshape(1, -1))

    A = np.squeeze(A.reshape(1, -1))
    B = np.squeeze(B.reshape(1, -1))
'''


def gtrack_matrixAdd(rows, cols, A, B):
    A = A.reshape(rows, cols)
    B = B.reshape(rows, cols)

    C = np.squeeze(np.add(A, B).reshape(1, -1))

    # A = np.squeeze(A.reshape(1, -1))
    # B = np.squeeze(B.reshape(1, -1))
    return np.float32(C)


# /**
# *  @b Description
# *  @n
# *     This function is used to subtract two matrices.
# *     Matrices are all real, single precision floating point.
# *     Matrices are in row-major order
# *
# *  @param[in]  rows
# *     Number of rows
# *  @param[in]  cols
# *     Number of cols
# *  @param[in]  A
# *     Matrix A
# *  @param[in]  B
# *     Matrix B
# *  @param[out]  C
# *     Matrix C(rows,cols) = A(rows,cols) - B(rows,cols)
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixSub(rows, cols, A, B, C):
    A = A.reshape(rows, cols)
    B = B.reshape(rows, cols)

    C = np.squeeze(np.subtract(A, B).reshape(1, -1))

    A = np.squeeze(A.reshape(1, -1))
    B = np.squeeze(B.reshape(1, -1))
'''


def gtrack_matrixSub(rows, cols, A, B):
    A = A.reshape(rows, cols)
    B = B.reshape(rows, cols)

    C = np.squeeze(np.subtract(A, B).reshape(1, -1))

    # A = np.squeeze(A.reshape(1, -1))
    # B = np.squeeze(B.reshape(1, -1))
    return np.float32(C)


# /**
# *  @b Description
# *  @n
# *     This function performs cholesky decomposition of 3x3 matrix.
# *     Matrix are squared, real, single precision floating point.
# *     Matrix are in row-major order
# *
# *  @param[in]  A
# *     Matrix A
# *  @param[out]  G
# *     Matrix G = cholseky(A);
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
'''
def gtrack_matrixCholesky3(A, G):
    A = A.reshape(3, 3)

    G = np.squeeze(np.linalg.cholesky(A).reshape(1,-1))

    A = np.squeeze(A.reshape(1, -1))
'''


def gtrack_matrixCholesky3(A):
    A = A.reshape(3, 3)

    G = np.squeeze(np.linalg.cholesky(A).reshape(1, -1))

    return G

    # G = np.zeros_like(A, dtype = np.float32)
    # v = np.zeros([3,], dtype = np.float32)
    # dim = 3

    # for j in range(dim):
    #     for i in range(j, dim):
    #         v[i] = A[i*dim + j]

    #     for k in range(0, j):
    #         for i in range(j, dim):
    #             v[i] = v[i] - G[j*dim + k] * G[i*dim + k]

    #     temp = 1. / np.sqrt(v[j])
    #     for i in range(j, dim):
    #         G[i*dim + j] = v[i] * temp

    # G[1] = G[2] = G[5] = 0.
    # # G = np.squeeze(G.reshape(1,-1))
    # return np.float32(G)


# /**
# *  @b Description
# *  @n
# *     This function computes the determinant of 3x3 matrix.
# *     Matrix is real, single precision floating point.
# *     Matrix is in row-major order
# *
# *  @param[in]  A
# *     Matrix A
# *  @param[out]  det
# *     det = det(A);
# *
# *  \ingroup gtrack_ALG_MATH_FUNCTION
# *
# *  @retval
# *      None
# */
def gtrack_matrixDet3(A):
    # det = A[0] * (A[4]*A[8] - A[7]*A[5]) - \
    #     A[1] * (A[3]*A[8] - A[5]*A[6]) + \
    #     A[2] * (A[3]*A[7] - A[4]*A[6])
    A = A.reshape(3, 3)
    det = np.linalg.det(A)

    return np.float32(det)


def gtrack_matrixInv3(A):
    A = A.reshape(3, 3)
    if np.linalg.det(A) == 0:
        inv = np.zeros(shape=(9,), dtype=np.float32)
        # A = np.squeeze(A.reshape(1, -1))
        return inv

    inv = np.squeeze(np.linalg.inv(A).reshape(1, -1))

    # A = np.squeeze(A.reshape(1, -1))
    return np.float32(inv)



def gtrack_spherical2cartesian(format, sph, cart):
    range = np.float32(sph[0])
    azimuth = np.float32(sph[1])
    doppler = np.float32(sph[2])

    if format == gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2DA:
        cart[4] = 0
        cart[5] = 0
        cart[0] = np.float32(range * np.sin(azimuth))
        cart[1] = np.float32(range * np.cos(azimuth))
        cart[2] = np.float32(doppler * np.sin(azimuth))
        cart[3] = np.float32(doppler * np.cos(azimuth))
    elif format == gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2D:
        cart[0] = np.float32(range * np.sin(azimuth))
        cart[1] = np.float32(range * np.cos(azimuth))
        cart[2] = np.float32(doppler * np.sin(azimuth))
        cart[3] = np.float32(doppler * np.cos(azimuth))
    else:
        return


def gtrack_cartesian2spherical(format, cart, sph):
    if format == gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2D or format == gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2DA:
        posx = np.float32(cart[0])
        posy = np.float32(cart[1])
        velx = np.float32(cart[2])
        vely = np.float32(cart[3])

        sph[0] = np.float32(np.sqrt(posx * posx + posy * posy))

        if posy == 0:
            sph[1] = np.float32((np.pi) / 2)
        elif posy > 0:
            sph[1] = np.float32(np.arctan(posx / posy))
        else:
            sph[1] = np.float32(np.arctan(posx / posy) + np.pi)

        sph[2] = np.float32((posx * velx + posy * vely) / sph[0])
    else:
        return


def gtrack_computeJacobian(format, cart, jac):
    posx = np.float32(cart[0])
    posy = np.float32(cart[1])
    velx = np.float32(cart[2])
    vely = np.float32(cart[3])

    range2 = np.float32(posx * posx + posy * posy)
    range = np.float32(np.sqrt(range2))
    range3 = np.float32(range * range2)

    if format == gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2D:
        jac[0] = np.float32(posx / range)
        jac[1] = np.float32(posy / range)
        jac[2] = 0.
        jac[3] = 0.

        jac[4] = np.float32(posy / range2)
        jac[5] = np.float32(-posx / range2)
        jac[6] = 0.
        jac[7] = 0.

        jac[8] = np.float32((posy * (velx * posy - vely * posx)) / range3)
        jac[9] = np.float32((posx * (vely * posx - velx * posy)) / range3)
        jac[10] = np.float32(posx / range)
        jac[11] = np.float32(posy / range)

    elif format == gtrack_STATE_VECTOR_TYPE().gtrack_STATE_VECTORS_2DA:
        jac[0] = np.float32(posx / range)
        jac[1] = np.float32(posy / range)
        jac[2] = 0.
        jac[3] = 0.
        jac[4] = 0.
        jac[5] = 0.
        jac[6] = np.float32(posy / range2)
        jac[7] = np.float32(-posx / range2)
        jac[8] = 0.
        jac[9] = 0.
        jac[10] = 0.
        jac[11] = 0.
        jac[12] = np.float32((posy * (velx * posy - vely * posx)) / range3)
        jac[13] = np.float32((posx * (vely * posx - velx * posy)) / range3)
        jac[14] = np.float32(posx / range)
        jac[15] = np.float32(posy / range)
        jac[16] = 0.
        jac[17] = 0.

    return


def gtrack_unrollRadialVelocity(rvMax, rvExp, rvIn):
    distance = np.float32(rvExp - rvIn)
    if distance >= 0:
        factor = int((distance + rvMax) / (2 * rvMax))
        rvOut = np.float32(rvIn + 2 * rvMax * factor)
    else:
        factor = int((rvMax - distance) / (2 * rvMax))
        rvOut = np.float32(rvIn - 2 * rvMax * factor)
    return rvOut


def isPointInsideBox(x, y, box):
    if (x > box.left) and (x < box.right) and (y > box.bottom) and (y < box.top):
        return 1
    else:
        return 0


# Simplified Gate Construction (no need for SVD) 
# We build a gate based on a constant volume 
# In addition, we impose a limiter: under no circumstances the gate will 
# allow to reach points beyond gateLimits 
def gtrack_gateCreateLim(volume, EC, range, gateLim):
    # LQ = np.zeros(shape = (9,))
    # W = np.zeros(shape = (9,))

    a = np.float32(1 / (np.sqrt(EC[0])))
    b = np.float32(1 / (np.sqrt(EC[4])))
    c = np.float32(1 / (np.sqrt(EC[8])))
    v = np.float32(4 * ((np.pi) / 3) * a * b * c)

    gConst = np.float32(np.power((volume / v), 2 / 3))

    LQ = gtrack_matrixCholesky3(EC)
    # print(LQ)

    W = gtrack_matrixInv3(LQ)
    # W = W.reshape(9,) #########################
    # print(W)

    gMin = gConst

    if gateLim[0] != 0:
        gLimit = np.float32((gateLim[0] * gateLim[0]) / (4 * (W[0] * W[0] + W[3] * W[3] + W[6] * W[6])))
        if gMin > gLimit:
            gMin = gLimit

    if gateLim[1] != 0:
        sWidth = np.float32(2 * range * np.float32(np.tan(np.float32(np.sqrt(W[4] * W[4] + W[7] * W[7])))))
        gLimit = np.float32((gateLim[1] * gateLim[1]) / (sWidth * sWidth))
        if gMin > gLimit:
            gMin = gLimit

    if gateLim[2] != 0:
        gLimit = np.float32((gateLim[2] * gateLim[2]) / (4 * (W[8] * W[8])))
        if gMin > gLimit:
            gMin = gLimit

    return np.float32(gMin)


def gtrack_computeMahalanobis3(d, S):
    chi2 = np.float32(d[0] * (d[0] * S[0] + d[1] * S[3] + d[2] * S[6]) + \
                      d[1] * (d[0] * S[1] + d[1] * S[4] + d[2] * S[7]) + \
                      d[2] * (d[0] * S[2] + d[1] * S[5] + d[2] * S[8]))
    return chi2

# GTRACK Unit Parameters structure
class TrackingParams():
    def __init__(self):
        self.uid = 0
        self.stateVectorType = gtrack_STATE_VECTOR_TYPE()
        self.verbose = 0
        self.initialRadialVelocity = 0.
        self.maxRadialVelocity = 0.
        self.radialVelocityResolution = 0.
        self.maxAcceleration = 0.
        self.deltaT = 0.

        self.gatingParams = gtrack_gatingParams()
        self.allocationParams = gtrack_allocationParams()
        self.unrollingParams = gtrack_unrollingParams()
        self.stateParams = gtrack_stateParams()
        self.variationParams = gtrack_varParams()
        self.sceneryParams = gtrack_sceneryParams()

        self.F4 = np.zeros(shape=(16,), dtype=np.float32)
        self.F6 = np.zeros(shape=(36,), dtype=np.float32)
        self.Q4 = np.zeros(shape=(16,), dtype=np.float32)
        self.Q6 = np.zeros(shape=(36,), dtype=np.float32)


# GTRACK Unit State
class TrackState():
    def __init__(self):
        self.TRACK_STATE_FREE = 0
        self.TRACK_STATE_INIT = 1
        self.TRACK_STATE_DETECTION = 2
        self.TRACK_STATE_ACTIVE = 3


# GTRACK Unit Velocity Handling State
class VelocityHandlingState():
    def __init__(self):
        self.VELOCITY_INIT = 0
        self.VELOCITY_RATE_FILTER = 1
        self.VELOCITY_TRACKING = 2
        self.VELOCITY_LOCKED = 3


# GTRACK Unit instance structure
class GtrackUnitInstance():
    def __init__(self):
        self.uid = 0
        self.tid = 0
        self.heartBeatCount = 0
        self.allocationTime = 0
        self.allocationRange = 0.
        self.allocationVelocity = 0.
        self.associatedPoints = 0

        self.state = TrackState()
        self.stateVectorType = gtrack_STATE_VECTOR_TYPE()
        self.currentStateVectorType = gtrack_STATE_VECTOR_TYPE()
        self.stateVectorLength = 0
        self.measurementVectorLength = 0
        self.verbose = 0

        self.gatingParams = gtrack_gatingParams()
        self.stateParams = gtrack_stateParams()
        self.allocationParams = gtrack_allocationParams()
        self.unrollingParams = gtrack_unrollingParams()
        self.variationParams = gtrack_varParams()
        self.sceneryParams = gtrack_sceneryParams()

        self.velocityHandling = VelocityHandlingState()

        self.initialRadialVelocity = 0.
        self.maxRadialVelocity = 0.
        self.radialVelocityResolution = 0.
        self.rangeRate = 0.

        self.detect2activeCount = 0
        self.detect2freeCount = 0
        self.active2freeCount = 0

        self.maxAcceleration = 0.
        self.processVariance = 0.
        self.dt = 0.

        self.F4 = None
        self.F6 = None
        self.Q4 = None
        self.Q6 = None

        self.F = None
        self.Q = None

        self.S_hat = np.zeros(shape=(6,), dtype=np.float32)
        self.S_apriori_hat = np.zeros(shape=(6,), dtype=np.float32)
        self.P_hat = np.zeros(shape=(36,), dtype=np.float32)
        self.P_apriori_hat = np.zeros(shape=(36,), dtype=np.float32)
        self.H_s = np.zeros(shape=(3,), dtype=np.float32)
        self.gD = np.zeros(shape=(9,), dtype=np.float32)
        self.gC = np.zeros(shape=(9,), dtype=np.float32)
        self.gC_inv = np.zeros(shape=(9,), dtype=np.float32)
        self.G = 0.


# GTRACK Module instance structure
class GtrackModuleInstance():
    def __init__(self):
        self.maxNumPoints = 0
        self.maxNumTracks = 0
        self.params = TrackingParams()

        self.heartBeat = 0

        self.verbose = 0
        self.bestScore = None
        self.bestIndex = None
        self.allocIndex = None
        self.hTrack = None

        self.activeList = []
        self.freeList = []

        self.uidElem = None

        self.targetDesc = None
        self.targetNumTotal = 0
        self.targetNumCurrent = 0
