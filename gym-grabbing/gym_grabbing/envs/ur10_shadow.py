import pybullet_data
import numpy as np
import gym
import os
import random
from pathlib import Path
from gym_grabbing.envs.robot_grasping import RobotGrasping
from gym_grabbing.envs.xacro import _process
import gym_grabbing



class UR10_shadow(RobotGrasping):

    def __init__(self,
        obstacle_pos=None,
        obstacle_size=0.1,
        object_position=[0, 0.1, 0],
        mode='joint positions',
        left=False, # the default is the right hand
        **kwargs
	):

        self.obstacle_size = obstacle_size
        cwd = Path(gym_grabbing.__file__).resolve().parent/"envs"

        urdf = Path(cwd/f"robots/generated/ur10_shadow.urdf")
        urdf.parent.mkdir(exist_ok=True)
        if not urdf.is_file(): # create the file if doesn't exist
            _process(str(cwd/"robots/ur_description/urdf/ur10_shadow.xacro"), dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1, mappings={})) # convert xacro to urdf


        def load_ur10_shadow():
            id = self.p.loadURDF(str(urdf), basePosition=[0, -0.5, -0.5], baseOrientation=[0., 0., 0., 1.], useFixedBase=True, flags=self.p.URDF_USE_SELF_COLLISION) # kuka_with_gripper2 gripper have a continuous joint (7)
            for i, pos in {1:0, 2:-np.pi/2, 3:0, 4:0, 5:0, 6:0}.items():
                self.p.resetJointState(id, i, targetValue=pos)
            #for i in range(self.p.getNumJoints(id)): print("joint info", self.p.getJointInfo(id, i))
            return id

        super().__init__(
            robot=load_ur10_shadow,
            camera={'target':(0,0,0.35), 'distance':0.7, 'pitch':-0, 'fov':90},
            mode=mode,
            object_position=object_position,
            table_height=0.8,
            joint_ids=[1, 2, 3, 4, 5, 6, 9,10, 13,14,15,16, 18,19,20,21, 23,24,25,26, 28,29,30,31,32, 34,35,36,37,38],
            contact_ids=list(range(9,39)),
            n_control_gripper=24,
            end_effector_id = 11,
            center_workspace = 0,
            radius = 1,
            disable_collision_pair = [],#[[8,11], [10,13]],
            change_dynamics = {**{ # change joints ranges for gripper and add jointLimitForce and maxJointVelocity, the default are 0 in the sdf and this produces very weird behaviours
#                id:{'lateralFriction':1, 'jointLowerLimit':l, 'jointUpperLimit':h, 'jointLimitForce':10, 'jointDamping':0.5, 'maxJointVelocity':0.5} for id,l,h in [ # , 'maxJointVelocity':1
#                    (8, -0.5, -0.05), # b'base_left_finger_joint
#                    (11, 0.05, 0.5), # b'base_right_finger_joint
#                    (10, -0.3, 0.1), # b'left_base_tip_joint
#                    (13, -0.1, 0.3)] # b'right_base_tip_joint
            # decrease max force to 50 & velocity to 0.5
            # J1 needs more torque to lift the arm so we set a higher torque
            # J6 is very unstable (the torque explodes) when using PDControllerStable so we set a low torque
            }, **{i:{'maxJointVelocity':0.5} for i in range(7)}}, # jointLimitForce
            **kwargs,
        )
        #self.p.loadURDF("duck_vhacd.urdf")




    def get_object(self, obj=None):
        if obj == 'cuboid':
            return {"shape":'cuboid', "x":0.06, "y":0.06, "z":0.16}
        elif obj == 'cube':
            return {"shape": 'cube', "unit":0.055, "lateralFriction":0.8}
        elif obj == 'sphere':
            return {"shape":'sphere', "radius":0.055}
        elif obj == 'cylinder':
            return {"shape":'cylinder', "radius":0.032, "z":0.15}
        elif obj == 'paper roll':
            return {"shape":'cylinder', "radius":0.021, "z":0.22}
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
            #fingers = np.array([-1, -1, 1, 1]) * -1
            for id, a, v, f, u, l in zip(self.joint_ids[-16:], fingers, self.maxVelocity[-16:], self.maxForce[-16:], self.upperLimits[-16:], self.lowerLimits[-16:]):
                self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=l+(a+1)/2*(u-l), maxVelocity=v, force=f)
            commands = action[:-1]

        # apply the commands
        return super().step(commands)

    def get_fingers(self, x):
        return -np.array([0,0,0,x,x,x,0,x,x,x,0,x,x,x,1,0,x,x,x,x,x,x,x,x])

    def reset_robot(self):
        for i, in zip(self.joint_ids,):
            p.resetJointState(self.robot_id, i, targetValue=-0.5 if i==2 else 0)


if __name__ == "__main__": # testing
    env = UR10_shadow(display=True, gripper_display=True, left=True)
    #for i in range(env.p.getNumJoints(env.robot_id)): print("joint info", p.getJointInfo(env.robot_id, i))

    for i in range(10000000000):
            env.step([0,-0.5,0,0,0,0, -1])
