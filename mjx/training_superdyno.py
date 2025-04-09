######################################
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8" # 0.9 causes too much lag. 
from datetime import datetime
import functools

# Math
import jax.numpy as jp
import numpy as np
import jax
from jax import config # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'high')
# config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)
from brax import math

# Sim
import mujoco
import mujoco.mjx as mjx

# Brax
from brax import envs
from brax.base import Motion, Transform
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
from brax.mjx.pipeline import _reformat_contact
from brax.training.acme import running_statistics
from brax.io import model

# Algorithms
from brax.training.agents.apg import train as apg
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.ppo import train as ppo

# Supporting
from etils import epath
import mediapy as media
import matplotlib.pyplot as plt
from ml_collections import config_dict
from typing import Any, Dict
from PIL import Image
import imageio

from superdyno_train import superdyno_train, extract_and_concat_state_info, make_neural_world_models, rotate_vec_batch, _get_self_obs
from superdyno_train_onedevice import superdyno_train_onedevice
# Load in the model
xml_path = epath.Path('mujoco_menagerie/anybotics_anymal_c/scene_mjx.xml').as_posix()

mj_model = mujoco.MjModel.from_xml_path(xml_path)

if 'renderer' not in dir():
    renderer = mujoco.Renderer(mj_model)

init_q = mj_model.keyframe('standing').qpos

mj_data = mujoco.MjData(mj_model)
mj_data.qpos = init_q
mujoco.mj_forward(mj_model, mj_data)

renderer.update_scene(mj_data)

# img = Image.fromarray((renderer.render()).astype(np.uint8))  # assuming float32 [0, 1]
# img.save("output1.png")

def save_video(frames, filename="output.mp4", fps=30):
    # Convert to uint8 if needed
    frames_uint8 = [(frame).astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in frames]
    
    imageio.mimsave(filename, frames_uint8, fps=fps)
    print(f"Video saved to: {filename}")

# Rendering Rollouts
def render_rollout(reset_fn, step_fn, 
                   inference_fn, env, filename,
                   n_steps = 200, camera=None,
                   seed=0):
    rng = jax.random.key(seed)
    render_every = 3
    state = reset_fn(rng)
    rollout = [state.pipeline_state]

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        if i % render_every == 0:
            rollout.append(state.pipeline_state)

    save_video(env.render(rollout, camera=camera),
               filename=filename, fps=1.0 / (env.dt*render_every))
    
    # media.show_video(env.render(rollout, camera=camera), 
    #                  fps=1.0 / (env.dt*render_every),
    #                  codec='gif')
    


### Designing Reference Kinematics
def cos_wave(t, step_period, scale):
    _cos_wave = -jp.cos(((2*jp.pi)/step_period)*t)
    return _cos_wave * (scale/2) + (scale/2)

def dcos_wave(t, step_period, scale):
    """ 
    Derivative of the cos wave, for reference velocity
    """
    return ((scale*jp.pi) / step_period) * jp.sin(((2*jp.pi)/step_period)*t)

def make_kinematic_ref(sinusoid, step_k, scale=0.3, dt=1/50):
    """ 
    Makes trotting kinematics for the 12 leg joints.
    step_k is the number of timesteps it takes to raise and lower a given foot.
    A gait cycle is 2 * step_k * dt seconds long.
    """
    
    _steps = jp.arange(step_k)
    step_period = step_k * dt
    t = _steps * dt
    
    wave = sinusoid(t, step_period, scale)
    # Commands for one step of an active front leg
    fleg_cmd_block = jp.concatenate(
        [jp.zeros((step_k, 1)),
        wave.reshape(step_k, 1),
        -2*wave.reshape(step_k, 1)],
        axis=1
    )
    # Our standing config reverses front and hind legs
    h_leg_cmd_bloc = -1 * fleg_cmd_block

    block1 = jp.concatenate([
        jp.zeros((step_k, 3)),
        fleg_cmd_block,
        h_leg_cmd_bloc,
        jp.zeros((step_k, 3))],
        axis=1
    )

    block2 = jp.concatenate([
        fleg_cmd_block,
        jp.zeros((step_k, 3)),
        jp.zeros((step_k, 3)),
        h_leg_cmd_bloc],
        axis=1
    )
    # In one step cycle, both pairs of active legs have inactive and active phases
    step_cycle = jp.concatenate([block1, block2], axis=0)
    return step_cycle


poses  = make_kinematic_ref(cos_wave, step_k=25) # 1 seconds

frames = []
init_q = mj_model.keyframe('standing').qpos
mj_data.qpos = init_q
default_ap = init_q[7:]

for i in range(len(poses)):
    mj_data.qpos[7:] = poses[i] + default_ap
    mujoco.mj_forward(mj_model, mj_data)
    renderer.update_scene(mj_data)
    frames.append(renderer.render())

# save_video(frames, filename="output_kinematics.mp4", fps=50)


# create RL environment
def get_config():
  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
              dict(
                min_reference_tracking = -2.5 * 3e-3, # to equalize the magnitude
                reference_tracking = -1.0,
                feet_height = -1.0
                )
              )
            )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(rewards=get_default_rewards_config(),))

  return default_config

# Math functions from (https://github.com/jiawei-ren/diffmimic)
def quaternion_to_matrix(quaternions):
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_rotation_6d(matrix):
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))

def quaternion_to_rotation_6d(quaternion):
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))

class TrotAnymal(PipelineEnv):

  def __init__(
      self,
      termination_height: float=0.25,
      **kwargs,
  ):
    step_k = kwargs.pop('step_k', 25)

    physics_steps_per_control_step = 10
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    kp = 230
    mj_model.actuator_gainprm[:, 0] = kp
    mj_model.actuator_biasprm[:, 1] = -kp

    sys = mjcf.load_model(mj_model)

    super().__init__(sys=sys, **kwargs)    
    
    self.termination_height = termination_height
    
    self._init_q = mj_model.keyframe('standing').qpos
    
    self.err_threshold = 0.4 # diffmimic; value from paper.
    
    self._default_ap_pose = mj_model.keyframe('standing').qpos[7:]
    self.reward_config = get_config()

    self.action_loc = self._default_ap_pose
    self.action_scale = jp.array([0.2, 0.8, 0.8] * 4)
    
    self.feet_inds = jp.array([21,28,35,42]) # LF, RF, LH, RH

    #### Imitation reference
    kinematic_ref_qpos = make_kinematic_ref(
      cos_wave, step_k, scale=0.3, dt=self.dt)
    kinematic_ref_qvel = make_kinematic_ref(
      dcos_wave, step_k, scale=0.3, dt=self.dt)
    
    self.l_cycle = jp.array(kinematic_ref_qpos.shape[0])
    
    # Expand to entire state space.

    kinematic_ref_qpos += self._default_ap_pose
    ref_qs = np.tile(self._init_q.reshape(1, 19), (self.l_cycle, 1))
    ref_qs[:, 7:] = kinematic_ref_qpos
    self.kinematic_ref_qpos = jp.array(ref_qs)
    
    ref_qvels = np.zeros((self.l_cycle, 18))
    ref_qvels[:, 6:] = kinematic_ref_qvel
    self.kinematic_ref_qvel = jp.array(ref_qvels)

    # Can decrease jit time and training wall-clock time significantly.
    self.pipeline_step = jax.checkpoint(self.pipeline_step, 
      policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    
  def reset(self, rng: jax.Array) -> State:
    # Deterministic init

    qpos = jp.array(self._init_q)
    qvel = jp.zeros(18)
    
    data = self.pipeline_init(qpos, qvel)

    # Position onto ground
    pen = jp.min(data.contact.dist)
    qpos = qpos.at[2].set(qpos[2] - pen)
    data = self.pipeline_init(qpos, qvel)

    state_info = {
        'rng': rng,
        'steps': 0.0,
        'reward_tuple': {
            'reference_tracking': 0.0,
            'min_reference_tracking': 0.0,
            'feet_height': 0.0
        },
        'last_action': jp.zeros(12), # from MJX tutorial.
        'kinematic_ref': jp.zeros(19),
    }

    x, xd = data.x, data.xd
    obs = self._get_obs(data.qpos, x, xd, state_info)
    reward, done = jp.zeros(2)
    metrics = {}
    for k in state_info['reward_tuple']:
      metrics[k] = state_info['reward_tuple'][k]
    state = State(data, obs, reward, done, metrics, state_info)
    return jax.lax.stop_gradient(state)
  
  def step(self, state: State, action: jax.Array) -> State:
    action = jp.clip(action, -1, 1) # Raw action

    action = self.action_loc + (action * self.action_scale)

    data = self.pipeline_step(state.pipeline_state, action)
    
    ref_qpos = self.kinematic_ref_qpos[jp.array(state.info['steps']%self.l_cycle, int)]
    ref_qvel = self.kinematic_ref_qvel[jp.array(state.info['steps']%self.l_cycle, int)]
    
    # Calculate maximal coordinates
    ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
    ref_data = mjx.forward(self.sys, ref_data)
    ref_x, ref_xd = ref_data.x, ref_data.xd

    state.info['kinematic_ref'] = ref_qpos

    # observation data
    x, xd = data.x, data.xd
    obs = self._get_obs(data.qpos, x, xd, state.info)

    # Terminate if flipped over or fallen down.
    done = 0.0
    done = jp.where(x.pos[0, 2] < self.termination_height, 1.0, done)
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0, 1.0, done)

    # reward
    reward_tuple = {
        'reference_tracking': (
          self._reward_reference_tracking(x, xd, ref_x, ref_xd)
          * self.reward_config.rewards.scales.reference_tracking
        ),
        'min_reference_tracking': (
          self._reward_min_reference_tracking(ref_qpos, ref_qvel, state)
          * self.reward_config.rewards.scales.min_reference_tracking
        ),
        'feet_height': (
          self._reward_feet_height(data.geom_xpos[self.feet_inds][:, 2]
                                   ,ref_data.geom_xpos[self.feet_inds][:, 2])
          * self.reward_config.rewards.scales.feet_height
        )
    }
    
    reward = sum(reward_tuple.values())

    # state management
    state.info['reward_tuple'] = reward_tuple
    state.info['last_action'] = action # used for observation. 

    for k in state.info['reward_tuple'].keys():
      state.metrics[k] = state.info['reward_tuple'][k]

    state = state.replace(
        pipeline_state=data, obs=obs, reward=reward,
        done=done)
    
    #### Reset state to reference if it gets too far
    # error = (((x.pos - ref_x.pos) ** 2).sum(-1)**0.5).mean()
    # to_reference = jp.where(error > self.err_threshold, 1.0, 0.0)

    # to_reference = jp.array(to_reference, dtype=int) # keeps output types same as input. 
    # ref_data = self.mjx_to_brax(ref_data)

    # data = jax.tree_util.tree_map(lambda x, y: 
    #                               jp.array((1-to_reference)*x + to_reference*y, x.dtype), data, ref_data)
    
    x, xd = data.x, data.xd # Data may have changed.
    obs = self._get_obs(data.qpos, x, xd, state.info)
    
    # wrap up the reference data
    ref_vector = extract_and_concat_state_info(ref_data)
    # breakpoint()
    return state.replace(pipeline_state=data, obs=obs), ref_vector, done, ref_qpos
    
  

  def _get_obs(self, qpos: jax.Array, x: Transform, xd: Motion,
               state_info: Dict[str, Any]) -> jax.Array:
    # breakpoint()
    inv_base_orientation = math.quat_inv(x.rot[0])
    local_rpyrate = math.rotate(xd.ang[0], inv_base_orientation)

    obs_list = []
    # yaw rate
    obs_list.append(jp.array([local_rpyrate[2]]) * 0.25)
    # projected gravity
    obs_list.append(
        math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation))
    # motor angles
    # angles = qpos[7:19]
    # obs_list.append(angles - self._default_ap_pose)
    
    # self proprioception
    self_obs = _get_self_obs(x, xd)[0]
    obs_list.append(self_obs) # 192
    
    # last action
    # obs_list.append(state_info['last_action'])
    # kinematic reference
    kin_ref = self.kinematic_ref_qpos[jp.array(state_info['steps']%self.l_cycle, int)]
    obs_list.append(kin_ref[7:]) # First 7 indicies are fixed

    obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

    return obs
  
  def mjx_to_brax(self, data):
    """ 
    Apply the brax wrapper on the core MJX data structure.
    """
    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[self.sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    data = _reformat_contact(self.sys, data)
    return data.replace(q=q, qd=qd, x=x, xd=xd)


  # ------------ reward functions----------------
  def _reward_reference_tracking(self, x, xd, ref_x, ref_xd):
    """
    Rewards based on inertial-frame body positions.
    Notably, we use a high-dimension representation of orientation.
    """

    f = lambda x, y: ((x - y) ** 2).sum(-1).mean()

    _mse_pos = f(x.pos,  ref_x.pos)
    _mse_rot = f(quaternion_to_rotation_6d(x.rot),
                 quaternion_to_rotation_6d(ref_x.rot))
    _mse_vel = f(xd.vel, ref_xd.vel)
    _mse_ang = f(xd.ang, ref_xd.ang)

    # Tuned to be about the same size.
    return _mse_pos      \
      + 0.1 * _mse_rot   \
      + 0.01 * _mse_vel  \
      + 0.001 * _mse_ang

  def _reward_min_reference_tracking(self, ref_qpos, ref_qvel, state):
    """ 
    Using minimal coordinates. Improves accuracy of joint angle tracking.
    """
    pos = jp.concatenate([
      state.pipeline_state.qpos[:3],
      state.pipeline_state.qpos[7:]])
    pos_targ = jp.concatenate([
      ref_qpos[:3],
      ref_qpos[7:]])
    pos_err = jp.linalg.norm(pos_targ - pos)
    vel_err = jp.linalg.norm(state.pipeline_state.qvel- ref_qvel)

    return pos_err + vel_err

  def _reward_feet_height(self, feet_pos, feet_pos_ref):
    return jp.sum(jp.abs(feet_pos - feet_pos_ref)) # try to drive it to 0 using the l1 norm.

envs.register_environment('trotting_anymal', TrotAnymal)

make_networks_factory = functools.partial(
    apg_networks.make_apg_networks,
    hidden_layer_sizes=(256, 128)
)

make_world_factory = functools.partial(
  make_neural_world_models,
  hidden_layer_sizes=(256,128)
)

epochs = 499

train_fn = functools.partial(superdyno_train,
                             episode_length=240,
                             policy_updates=epochs,
                             horizon_length=26,
                             world_updates_per_epoch=8,
                             policy_updates_per_epoch=4,
                             world_training_length=8,
                             policy_training_length=24,
                             batch_size=512,
                             num_envs=64,
                             learning_rate_policy=1e-5,
                             learning_rate_world=5e-4,
                             num_eval_envs=64,
                             num_evals=10 + 1,
                             use_float64=True,
                             normalize_observations=True,
                             policy_network_factory=make_networks_factory,
                             world_network_factory=make_world_factory)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

def progress(it, metrics):
  times.append(datetime.now())
  x_data.append(it)
  y_data.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])
  print(f"Step {it} - "
        f"Reward: {metrics['eval/episode_reward']:.2f} +/- {metrics['eval/episode_reward_std']:.2f} - ")

# Each foot contacts the ground twice/sec.
env = envs.get_environment("trotting_anymal", step_k = 13)
eval_env = envs.get_environment("trotting_anymal", step_k = 13)

make_inference_fn, policy_params, _= train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)

# plt.errorbar(x_data, y_data, yerr=ydataerr)

demo_env = envs.training.EpisodeWrapper(env, 
                                        episode_length=1000, 
                                        action_repeat=1)

model_path = '/home/yulei/mujoco/mjx/trotting_2hz_policy_superdyno_debug'
model.save_params(model_path, policy_params)

render_rollout(
  jax.jit(demo_env.reset),
  jax.jit(demo_env.step),
  jax.jit(make_inference_fn(policy_params)),
  demo_env,
  filename='output_trot_superdyno_debug.mp4',
  n_steps=200,
  seed=1
)

