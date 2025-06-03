#!/usr/bin/env python

from gym.envs.registration import register
from .CustomizedGrid import CustomizedGrid


register(
    id='ContinuousVecGridEnv-v0',
    entry_point='envs.vectorized_grid:ContinuousGridEnv',
)

register(
    id='GoalGrid-v0',
    entry_point='envs.goal_grid:GoalContinuousGrid',
)

register(
    id='ReacherDraw-v0',
    entry_point='envs.reacher_trace:ReacherTraceEnv',
)

register(
    id='HopperFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Hopper-v2'
    )
)

register(
    id='Walker2dFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Walker2d-v2'
    )
)

register(
    id='HalfCheetahFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='HalfCheetah-v2'
    )
)

register(
    id='AntFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Ant-v2'
    )
)

register(
    id='HumanoidFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Humanoid-v2'
    )
)


register(id='PointMazeRight-v0', entry_point='envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
register(id='PointMazeRightSparse-v0', entry_point='envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': True, 'direction': 1})
register(id='PointMazeLeft-v0', entry_point='envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 0})

# A modified ant which flips over less and learns faster via TRPO
register(id='CustomAnt-v0', entry_point='envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': False})
register(id='DisabledAnt-v0', entry_point='envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': True})

register(
    id='NonStHopper-v0',
    entry_point='envs.nonstationary_mujoco:NonstationaryMujoco',
    kwargs=dict(
        env_name='Hopper-v2'
    )
)

register(
    id='NonStWalker2d-v0',
    entry_point='envs.nonstationary_mujoco:NonstationaryMujoco',
    kwargs=dict(
        env_name='Walker2d-v2'
    )
)

register(
    id='NonStHalfCheetah-v0',
    entry_point='envs.nonstationary_mujoco:NonstationaryMujoco',
    kwargs=dict(
        env_name='HalfCheetah-v2'
    )
)

register(
    id='NonStAnt-v0',
    entry_point='envs.nonstationary_mujoco:NonstationaryMujoco',
    kwargs=dict(
        env_name='Ant-v2'
    )
)

register(
    id='NonStHumanoid-v0',
    entry_point='envs.nonstationary_mujoco:NonstationaryMujoco',
    kwargs=dict(
        env_name='Humanoid-v2'
    )
)

register(
    id='dmc_walker_walk-v0',
    entry_point='envs.dmc_env:DMCWrapper',
    kwargs=dict(
        domain_name="walker",
        task_name="walk",
    )
)

register(
    id='dmc_quadruped_walk-v0',
    entry_point='envs.dmc_env:DMCWrapper',
    kwargs=dict(
        domain_name="quadruped",
        task_name="walk",
    )
)

register(
    id='dmc_cheetah_run-v0',
    entry_point='envs.dmc_env:DMCWrapper',
    kwargs=dict(
        domain_name="cheetah",
        task_name="run",
    )
)
