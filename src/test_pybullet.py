import pybullet as p
import time
import pybullet_data
import qibullet as q

MAX_FORCE = 100


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
        if includeFixed or jointInfo[3] > -1:
            print(jointInfo[0], jointInfo[1], jointInfo[2], jointInfo[3], jointInfo[8:10], jointInfo[12])
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
                                                  restPoses=restPoses, maxNumIterations=2)
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


physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

pepper = q.PepperVirtual()

pepper.loadRobot(translation=[0, 0, 0],
                 quaternion=[0, 0, 0, 1],
                 physicsClientId=physicsClient)  # experimentation

pepper_id = pepper.getRobotModel()

num_joints = p.getNumJoints(pepper_id)
info_joint = p.getJointInfo(pepper_id, 0)

joints = pepper.joint_dict
joint_keys = joints.keys()

interesting_joints = ['HipPitch', 'HipRoll', 'HeadYaw', 'HeadPitch',
                      'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll',
                      'LWristYaw', 'LHand', 'LFinger21', 'LFinger22', 'LFinger23',
                      'LFinger11', 'LFinger12', 'LFinger13', 'LFinger41', 'LFinger42',
                      'LFinger43', 'LFinger31', 'LFinger32', 'LFinger33', 'LThumb1', 'LThumb2']
interesting_joints_idx = [joints[joint].getIndex() for joint in interesting_joints]
end_effector_id = interesting_joints.index('LThumb2')
targetPosition = [0.2, 0.0, -0.1]

lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(pepper_id, includeFixed=False)

for i in range(10000):
    p.stepSimulation()
    jointPoses = accurateIK(pepper_id, end_effector_id, targetPosition, lowerLimits, upperLimits,
                            jointRanges, restPoses, useNullSpace=False)

    setMotors(pepper_id, jointPoses)


p.disconnect()
