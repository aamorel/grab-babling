from time import sleep
import pybullet as p
import numpy as np


MAX_FORCE = 100


def setUpWorld(initialSimSteps=100):
    """
    Reset the simulation to the beginning and reload all models.

    Parameters
    ----------
    initialSimSteps : int

    Returns
    -------
    baxterId : int
    endEffectorId : int
    """
    p.resetSimulation()

    # load plane
    p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
    sleep(0.1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    # load Baxter
    # urdf_flags = p.URDF_USE_SELF_COLLISION    makes the simulation go crazy
    baxterId = p.loadURDF("baxter_common/baxter_description/urdf/toms_baxter.urdf", useFixedBase=True)
    p.resetBasePositionAndOrientation(baxterId, [0, -0.8, 0.0], [0., 0., -1., -1.])

    # table robot part shapes
    t_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.7, 0.7, 0.1])
    t_body_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.7, 0.7, 0.1], rgbaColor=[0.3, 0.3, 0, 1])

    t_legs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.4])
    t_legs_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.4], rgbaColor=[0.3, 0.3, 0, 1])

    body_Mass = 20
    visualShapeId = t_body_v
    link_Masses = [.1, .1, .1, .1]

    linkCollisionShapeIndices = [t_legs] * 4

    nlnk = len(link_Masses)
    linkVisualShapeIndices = [t_legs_v] * nlnk
    # link positions wrt the link they are attached to

    linkPositions = [[0.35, 0.35, -0.3], [-0.35, 0.35, -0.3], [0.35, -0.35, -0.3], [-0.35, -0.35, -0.3]]

    linkOrientations = [[0, 0, 0, 1]] * nlnk
    linkInertialFramePositions = [[0, 0, 0]] * nlnk
    # note the orientations are given in quaternions (4 params). There are function to convert of Euler angles and back
    linkInertialFrameOrientations = [[0, 0, 0, 1]] * nlnk
    # indices determine for each link which other link it is attached to
    indices = [0] * nlnk
    # most joint are revolving. The prismatic joints are kept fixed for now
    jointTypes = [p.JOINT_FIXED] * nlnk
    # revolution axis for each revolving joint
    axis = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]

    # drop the body in the scene at the following body coordinates
    basePosition = [0, 0.4, 0]
    baseOrientation = [0, 0, 0, 1]
    # main function that creates the table
    tab = p.createMultiBody(body_Mass, t_body, visualShapeId, basePosition, baseOrientation,
                            linkMasses=link_Masses,
                            linkCollisionShapeIndices=linkCollisionShapeIndices,
                            linkVisualShapeIndices=linkVisualShapeIndices,
                            linkPositions=linkPositions,
                            linkOrientations=linkOrientations,
                            linkInertialFramePositions=linkInertialFramePositions,
                            linkInertialFrameOrientations=linkInertialFrameOrientations,
                            linkParentIndices=indices,
                            linkJointTypes=jointTypes,
                            linkJointAxis=axis)
    print(tab)

    # create object to grab
    square_base = 0.02
    height = 0.08
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[square_base, square_base, height])
    viz_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[square_base, square_base, 0.1], rgbaColor=[1, 0, 0, 1])
    obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
    p.resetBasePositionAndOrientation(obj_to_grab_id, [0, 0.2, 3], [0, 0, 0, 1])

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # grab relevant joint IDs
    endEffectorId = 48  # (left gripper left finger)

    # set gravity
    p.setGravity(0., 0., -9.81)

    # let the world run for a bit
    for _ in range(initialSimSteps):
        p.stepSimulation()

    return baxterId, endEffectorId, obj_to_grab_id


def getJointRanges(bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

    numJoints = p.getNumJoints(bodyId)
    # loop through all joints
    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        print(jointInfo[0], jointInfo[1], jointInfo[2], jointInfo[3], jointInfo[8:10], jointInfo[12])
        if includeFixed or jointInfo[3] > -1:
            # jointInfo[3] > -1 means that the joint is not fixed
            ll, ul = jointInfo[8:10]
            jr = ul - ll

            # For simplicity, assume resting state == initial state
            rp = p.getJointState(bodyId, i)[0]

            lowerLimits.append(ll)
            upperLimits.append(ul)
            jointRanges.append(jr)
            restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses


def accurateIK(bodyId, endEffectorId, targetPosition, lowerLimits, upperLimits, jointRanges, restPoses,
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    bodyId : int
    endEffectorId : int
    targetPosition : [float, float, float]
    lowerLimits : [float]
    upperLimits : [float]
    jointRanges : [float]
    restPoses : [float]
    useNullSpace : bool
    maxIter : int
    threshold : float

    Returns
    -------
    jointPoses : [float] * numDofs
    """

    if useNullSpace:
        jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,
                                                  lowerLimits=lowerLimits, upperLimits=upperLimits,
                                                  jointRanges=jointRanges,
                                                  restPoses=restPoses)
    else:
        jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition)

    return jointPoses


def setMotors(bodyId, jointPoses):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """
    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        qIndex = jointInfo[3]
        if qIndex > -1:
            p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[qIndex - 7], force=MAX_FORCE)


if __name__ == "__main__":
    guiClient = p.connect(p.DIRECT)
    p.resetDebugVisualizerCamera(2., 180, 0., [0.52, 0.2, np.pi / 4.])

    targetPosXId = p.addUserDebugParameter("targetPosX", -1, 1, 0.2)
    targetPosYId = p.addUserDebugParameter("targetPosY", -1, 1, 0)
    targetPosZId = p.addUserDebugParameter("targetPosZ", -1, 1, -0.1)
    nullSpaceId = p.addUserDebugParameter("nullSpace", 0, 1, 1)

    target_gripper_pos_ID = p.addUserDebugParameter("target gripper pos", 0, 0.020833, 0.0)

    # set up the world, endEffector is the tip of the left finger
    baxterId, endEffectorId, obj_id = setUpWorld()

    lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(baxterId, includeFixed=False)
    targetPosition = [0.2, 0.0, -0.1]

    maxIters = 100000

    sleep(1.)

    d_inf = p.getDynamicsInfo(obj_id, -1)
    print('mass:', d_inf[0], ', lateral friction:', d_inf[1])

    # p.getCameraImage(320, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # start simu iteration
    for i in range(maxIters):
        p.stepSimulation()
        targetPosX = p.readUserDebugParameter(targetPosXId)
        targetPosY = p.readUserDebugParameter(targetPosYId)
        targetPosZ = p.readUserDebugParameter(targetPosZId)
        nullSpace = p.readUserDebugParameter(nullSpaceId)

        target_gripper_pos = p.readUserDebugParameter(target_gripper_pos_ID)

        targetPosition = [targetPosX, targetPosY, targetPosZ]
      
        useNullSpace = nullSpace > 0.5
        jointPoses = accurateIK(baxterId, endEffectorId, targetPosition, lowerLimits, upperLimits,
                                jointRanges, restPoses, useNullSpace=useNullSpace)

        setMotors(baxterId, jointPoses)

        if i % 2000 == 0:
            o = list(p.getBasePositionAndOrientation(obj_id)[0])
            g = list(p.getLinkState(baxterId, endEffectorId)[0])
            dist = ((o[0] - g[0])**2 + (o[1] - g[1])**2 + (o[2] - g[2])**2)**(1 / 2)
            print('Object position:', o)
            print('Gripper position:', g)
            print('Distance:', dist)

        # explicitly control the gripper
        p.setJointMotorControl2(bodyIndex=baxterId, jointIndex=49, controlMode=p.POSITION_CONTROL,
                                targetPosition=target_gripper_pos, force=MAX_FORCE)
        p.setJointMotorControl2(bodyIndex=baxterId, jointIndex=51, controlMode=p.POSITION_CONTROL,
                                targetPosition=-target_gripper_pos, force=MAX_FORCE)

        # sleep(0.1)
    p.disconnect()
