import pybullet_data
import numpy as np
import gym
import os
import random
from pathlib import Path
from gym_grabbing.envs.robot_grasping import RobotGrasping
from gym_grabbing.envs.xacro import _process
import gym_grabbing



class KukaRobotiq(RobotGrasping):

    def __init__(self,
        gripper='2f_140',
        **kwargs
	):
        assert gripper in {'2f_140', '2f_85', '3f'}, f"The gripper you gave {gripper} must be one of them: 2f_140, 2f_85, 3f."

        cwd = Path(gym_grabbing.__file__).resolve().parent/"envs"

        urdf = Path(cwd/f"robots/generated/kuka_{gripper}.urdf")
        urdf.parent.mkdir(exist_ok=True)
        if True:#not urdf.is_file(): # create the file if doesn't exist
            _process(str(cwd/f"robots/LBR_iiwa/urdf/lbr_iiwa_robotiq.xacro"), dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1, mappings={'arg_gripper':gripper})) # convert xacro to urdf


        def load_kuka_robotiq():
            id = self.p.loadURDF(str(urdf), basePosition=[0, -0.5, -0.75], baseOrientation=[0., 0., 0., 1.], useFixedBase=False)#, flags=self.p.URDF_USE_SELF_COLLISION)
            #for i in range(self.p.getNumJoints(id)): print("joint info", self.p.getJointInfo(id, i))
            return id

        super().__init__(
            robot=load_kuka_robotiq,
            camera={'target':(0,0,0.3), 'distance':0.7, 'pitch':-20, 'fov':90},
            table_height=0.8,
            joint_ids=[1,2,3,4,5,6,7, 13],
            contact_ids=(12,13),
            n_control_gripper=1,
            end_effector_id = 8,
            center_workspace = 1,
            radius = 1,
            disable_collision_pair = [[16,12],[17,15], [11,12], [14,15]],
            change_dynamics = { # decrease jointLimitForce: we suppose the payload is light
                1:{'maxJointVelocity':0.5, 'jointLimitForce':50},
                2:{'maxJointVelocity':0.5, 'jointLimitForce':60},
                3:{'maxJointVelocity':0.5, 'jointLimitForce':40},
                4:{'maxJointVelocity':0.5, 'jointLimitForce':30},
                5:{'maxJointVelocity':0.5, 'jointLimitForce':20},
                6:{'maxJointVelocity':0.5, 'jointLimitForce':10},
                7:{'maxJointVelocity':0.5, 'jointLimitForce':5},
            }, # jointLimitForce
            **kwargs,
        )

        #self._id = 16
        #self.lines = [self.p.addUserDebugLine([0, 0, 0], end, color, parentObjectUniqueId=self.robot_id, parentLinkIndex=self._id) for end, color in zip(np.eye(3)*0.2, np.eye(3))]
        #self.lines1 = [self.p.addUserDebugLine([0, 0, 0], end, color, parentObjectUniqueId=self.robot_id, parentLinkIndex=self._id) for end, color in zip(np.eye(3)*0.2, np.eye(3))]
        # change colors
        for i in (12,15,): #8, 9, 12, 14, 17,
            self.p.changeVisualShape(objectUniqueId=self.robot_id, linkIndex=i, rgbaColor=[1,1,1,1]) # white
        for i in (9, 11, 14, 16, 17, 18):
            self.p.changeVisualShape(objectUniqueId=self.robot_id, linkIndex=i, rgbaColor=[0.2,0.2,0.2,1]) # black

        # geometric constraints of the gripper
        for k, v in {16:12, 17:15}.items():#{13:11, 18:16}
            b = (k==16)*2-1
            c = self.p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=k,
                childBodyUniqueId=self.robot_id,
                childLinkIndex=v,
                jointType=self.p.JOINT_POINT2POINT,
                jointAxis=[0,0,0],
                parentFramePosition={"2f_140":[0.034*b, 0, 0.037], "2f_85":[0.0185*b,0,0.020]}[gripper],
                childFramePosition={"2f_140":[-0.002*b, 0, -0.018], "2f_85":[-0.004*b,0,-0.01]}[gripper]
                #parentFramePosition={"2f_140":[0,0.0485,-0.003], "2f_85":[0,0.037,0.0435]}[gripper],
                #childFramePosition={"2f_140":[0,-0.01, -0.004], "2f_85":[0,0,0]}[gripper]
            )
            self.p.changeConstraint(c, erp=0.1, maxForce=1000)
        for i in (10,12,13,15,16,17):#(11,13,14,16,18): # disable motor constrains
            self.p.setJointMotorControl2(self.robot_id, i, self.p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        # left right symmetric
        c = self.p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=10,
            childBodyUniqueId=self.robot_id,
            childLinkIndex=13,
            jointType=self.p.JOINT_GEAR,
            jointAxis=[1,0,0],
            parentFramePosition=[0,0,0],
            childFramePosition=[0,0,0]
        )
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=100)
        #self.x = self.p.addUserDebugParameter("x",-0.1, 0.1)
        #self.z = self.p.addUserDebugParameter("z",-0.1, 0.1)




    def get_object(self, obj=None):
        if obj == 'cuboid':
            return {"shape":'cuboid', "x":0.06, "y":0.06, "z":0.16}
        elif obj == 'cube':
            return {"shape": 'cube', "unit":0.055, "lateralFriction":0.8}
        elif obj == 'sphere':
            return {"shape":'sphere', "radius":0.055}
        elif obj == 'cylinder':
            return {"shape":'cylinder', "radius":0.032, "z":0.15}
        else:
            return obj



    def step(self, action=None):
        if action is None:
            return super().step()

        fingers = self.get_fingers(action[-1])
        if self.mode == 'joint positions':
            # we want one action per joint (gripper is composed by 4 joints but considered as one)
            assert(len(action) == self.n_actions)
            self.info['closed gripper'] = action[-1]<0
            # add the 3 commands for the 3 last gripper joints
            commands = np.hstack([action[:-1], fingers])
        elif self.mode == 'inverse kinematics':
            target_position = action[0:3]
            target_orientation = action[3:7] # xyzw
            gripper = action[7] # [-1(open), 1(close)]
            self.info['closed gripper'] = gripper<0

            commands = np.zeros(self.n_joints)
            commands[:-4] = self.p.calculateInverseKinematics(self.robot_id, self.end_effector_id, target_position, targetOrientation=target_orientation)[:-4]
            commands = 2*(commands-self.upperLimits)/(self.upperLimits-self.lowerLimits) + 1 # set between -1, 1
            commands[-4:] = fingers # add fingers
            commands = commands.tolist()

        elif self.mode in {'joint torques', 'joint velocities', 'inverse dynamics', 'pd stable'}:
            # control the gripper in positions
            for id, a, v, f, u, l in zip(self.joint_ids[-1:], fingers, self.maxVelocity[-1:], self.maxForce[-1:], self.upperLimits[-1:], self.lowerLimits[-1:]):
                self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=l+(a+1)/2*(u-l), maxVelocity=v, force=f)
            commands = action[:-1]


        #self.lines = [self.p.addUserDebugLine([0,0,0], end, color, replaceItemUniqueId=id, parentObjectUniqueId=self.robot_id, parentLinkIndex=self._id) for end, color, id in zip(np.eye(3)*0.5, np.eye(3), self.lines)]
        #pos = np.array([self.p.readUserDebugParameter(self.x), 0, self.p.readUserDebugParameter(self.z)])
        #self.lines1 = [self.p.addUserDebugLine(pos, pos+end, color, replaceItemUniqueId=id, parentObjectUniqueId=self.robot_id, parentLinkIndex=self._id) for end, color, id in zip(np.eye(3)*0.5, np.eye(3), self.lines1)]
        # apply the commands
        return super().step(commands)

    def get_fingers(self, x):
        return -np.array([x])

    def reset_robot(self):
        for i,j in zip(self.joint_ids,[0,0,0,0,0,0,0, 0,0,0,0,0,0]):
            self.p.resetJointState(self.robot_id, i, targetValue=j)


if __name__ == "__main__": # testing
    import time
    env = KukaRobotiq(display=True, gripper_display=False, gripper="2f_140", mode="pd stable")
    for i in range(env.p.getNumJoints(env.robot_id)): print("joint info", env.p.getJointInfo(env.robot_id, i))

    for i in range(10000000000):
        env.step([0,0,0,0,0,0,0, -1])#np.cos(i/100)])
        #print(env.info['applied joint motor torques'])
