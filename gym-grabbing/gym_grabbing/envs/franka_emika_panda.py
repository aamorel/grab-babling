import pybullet_data
import numpy as np
import gym
import os
import random
from pathlib import Path
from gym_grabbing.envs.robot_grasping import RobotGrasping
from gym_grabbing.envs.xacro import _process
import gym_grabbing



class Franka_emika_panda(RobotGrasping):

    def __init__(self,
        obstacle_pos=None,
        obstacle_size=0.1,
        object_position=[0, 0.1, 0],
        mode='joint positions',
        left=False, # the default is the right hand
        **kwargs
	):
        

        def load_panda():
            id = self.p.loadURDF("franka_panda/panda.urdf", basePosition=[0, -0.5, -0.5], baseOrientation=[0., 0., 0., 1.], useFixedBase=True)
            #for i in range(self.p.getNumJoints(id)): print("joint info", self.p.getJointInfo(id, i))
            for i, pos in {0:np.pi/2, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}.items():
                self.p.resetJointState(id, i, targetValue=pos)
            return id

        super().__init__(
            robot=load_panda,
            camera={'target':(0,0,0.1), 'distance':0.4, 'pitch':-10, 'fov':90},
            mode=mode,
            object_position=object_position,
            table_height=0.8,
            joint_ids=[0, 1, 2, 3, 4, 5, 6, 9,10],
            contact_ids=[9,10],
            n_control_gripper=2,
            end_effector_id = 11,
            center_workspace = 0,
            radius = 1,
            disable_collision_pair = [],
            **kwargs,
        )



    
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
            for id, a, v, f, u, l in zip(self.joint_ids[2:], fingers, self.maxVelocity[2:], self.maxForce[2:], self.upperLimits[2:], self.lowerLimits[2:]):
                self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=l+(a+1)/2*(u-l), maxVelocity=v, force=f)
            if self.mode in {'joint torques', 'joint velocities'}:
                commands = action[:-1]
            elif self.mode in {'inverse dynamics', 'pd stable'}:
                commands = np.hstack((action[:-1], fingers))

        # apply the commands
        return super().step(commands)

    def get_fingers(self, x):
        return np.array([x, x])
    
    def reset_robot(self):
        for i, in zip(self.joint_ids,):
            p.resetJointState(self.robot_id, i, targetValue=0)


if __name__ == "__main__": # testing
    env = Franka_emika_panda(display=True, gripper_display=True)
    #for i in range(env.p.getNumJoints(env.robot_id)): print("joint info", p.getJointInfo(env.robot_id, i))
    
    for i in range(10000000000):
            env.step([0,np.sin(i/1000),0,0,0,0,np.sin(i/1000), np.sin(i/1000)])
