"""SuperDyno training test -- Yu Lei"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union, Sequence, Tuple
import dataclasses

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.apg import networks as apg_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training import networks

from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import optax
from flax import linen

from brax import math
from replay_buffer import ReplayBufferSuper
from brax.base import Motion, Transform

import mujoco.mjx as mjx

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'

# Math functions from (https://github.com/jiawei-ren/diffmimic)
def quaternion_to_matrix(quaternions):
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jnp.stack(
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


def extract_and_concat_state_info(pipeline_state: Any) -> jnp.ndarray:
    """
    Extracts xpos (x.pos), xquat (x.rot), xvel(x.vel), xang(x.ang_vel) from MJX pipeline_state,
    and concatenates into a single [B, D] array.
    """
    # qpos = pipeline_state.q         # [B, nq] , specifically nq = 19 in this case
    # qvel = pipeline_state.qd        # [B, nv] , specifically nv = 18 in this case
    xpos = pipeline_state.x.pos     # [B, nb, 3], specifically nb = 13 in this case
    xquat = pipeline_state.x.rot    # [B, nb, 4]
    xvel = pipeline_state.xd.vel    # [B, nb, 3]
    xang = pipeline_state.xd.ang # [B, nb, 3]

    # Flatten xpos and xquat across bodies
    xpos_flat = jnp.reshape(xpos, (xpos.shape[0], -1))    # [B, nb*3]
    xquat_flat = jnp.reshape(xquat, (xquat.shape[0], -1)) # [B, nb*4]
    xvel_flat = jnp.reshape(xvel, (xvel.shape[0], -1))    # [B, nb*3]
    xang_flat = jnp.reshape(xang, (xang.shape[0], -1))    # [B, nb*3]

    # Concatenate everything along last dim
    state_vector = jnp.concatenate([xpos_flat, xquat_flat, xvel_flat, xang_flat], axis=-1)  # [B, D], D = 13 * (3 + 4 + 3 + 3) = 169

    return state_vector


def decompose_state(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Decomposes the state into its components: xpos, xquat, xvel, xang.
    """
    B = state.shape[0]
    N = 13
    xpos = state[..., :N * 3].reshape(-1, 3)
    xquat = state[..., N * 3: N * 7].reshape(-1, 4)
    xvel = state[..., N * 7: N * 10].reshape(-1, 3)
    xang = state[..., N * 10:].reshape(-1, 3)
    return xpos, xquat, xvel, xang

def compute_tracking_loss_from_state(pred_state: jnp.ndarray, target_state: jnp.ndarray):
    
    pred_pos, pred_quat, pred_vel, pred_ang = decompose_state(pred_state)
    target_pos, target_quat, target_vel, target_ang = decompose_state(target_state)
    
    f = lambda x, y: ((x - y) ** 2).sum(-1).mean()
    
    _mse_pos = f(pred_pos, target_pos)
    _mse_rot = f(quaternion_to_rotation_6d(pred_quat), quaternion_to_rotation_6d(target_quat)) * 0.1
    _mse_vel = f(pred_vel, target_vel)
    _mse_ang = f(pred_ang, target_ang)
    
    # jax.debug.print('mse_pos:{}', _mse_pos)
    # jax.debug.print('mse_rot:{}', _mse_rot)
    # jax.debug.print('mse_vel:{}', _mse_vel)
    # jax.debug.print('mse_ang:{}', _mse_ang)
    
    return _mse_pos \
        +  _mse_rot \
        +  _mse_vel \
        +  _mse_ang


    


def batchify(data: dict, batch_size: int):
    total = next(iter(data.values())).shape[0]
    num_batches = total // batch_size
    reshaped = {
        k: v[:num_batches * batch_size].reshape((num_batches, batch_size) + v.shape[1:])
        for k, v in data.items()
    }
    return reshaped


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    
    optimizer_state: optax.OptState
    network_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    # env_steps: jnp.ndarray

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


### Create world model networks and implement the step function and integration ###
@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

class MLP(linen.Module):
  """MLP module."""

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias,
      )(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = flax.linen.LayerNorm()(hidden)
    return hidden


def make_world_network(
    param_size: int, # here the param size should be the state size
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = flax.linen.silu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
) -> FeedForwardNetwork:
    """Creates a world model network."""
    world_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
    )
    # world_module = flax.linen.scan(
    #     MLP,
    #     variable_axes={'params': 0},
    #     split_rngs={'params': False},
    #     in_axes=1,
    #     out_axes=1,
    #     length=7,
    # )(layer_sizes=list(hidden_layer_sizes) + [param_size],activation=activation, kernel_init=kernel_init)
    
    # breakpoint()
    def apply(processor_params, world_params, obs):
        # here the observation should be the self proprioception + action size
        # and the output should be the delta state: x.vel, x.ang_vel
        assert isinstance(obs, jnp.ndarray)
        # breakpoint()
        # obs = preprocess_observations_fn(obs, processor_params) # it can still be considered
        return world_module.apply(world_params, obs)
    # breakpoint()
    # NOTE: should be careful that the obs_size of world model should be given
    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: world_module.init(key, dummy_obs), apply=apply
    )
    
def make_neural_world_models(
    input_size: int,
    output_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = flax.linen.silu,
    layer_norm: bool = True,
):
    world_network = make_world_network(
        output_size,
        input_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        kernel_init=flax.linen.initializers.orthogonal(0.01),
        layer_norm=layer_norm,
    )
    
    return world_network

def rotate_vec_batch(quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
        # quat = quat[:, jnp.newaxis, :]
        # return jax.vmap(math.rotate)(vec, quat)
        return jax.vmap(jax.vmap(math.rotate, in_axes=(0, None)), in_axes=(0, 0))(vec, quat)

def transform_from_state_world(states: jnp.ndarray) -> jnp.ndarray:
    B = states.shape[0]
    N = 13
    x_pos = states[:, :N * 3].reshape(B, -1, 3)
    x_quat = states[:, N * 3: N * 7].reshape(B, -1, 4)
    xd_vel = states[:, N * 7: N * 10].reshape(B, -1, 3)
    xd_ang = states[:, N * 10: N * 13].reshape(B, -1, 3)
    
    # turn into body frame
    base_pos = x_pos[:, 0, :]
    base_rot_inv = math.quat_inv(x_quat[:, 0])
    
    # base_rot_inv = base_rot_inv[:, jnp.newaxis, :]
    # breakpoint()
    # pos
    rel_pos = x_pos - base_pos[:, None, :]
    # rel_pos_body = jax.vmap(math.rotate)(rel_pos[:, 1:], base_rot_inv) # remove the base pos
    rel_pos_body = rotate_vec_batch(base_rot_inv, rel_pos[:, 1:])
    # rot:
    # rel_rot = math.quat_mul(base_rot_inv[:, None, :], x_quat)
    rel_rot = jax.vmap(jax.vmap(math.quat_mul, in_axes=(None, 0)), in_axes=(0, 0))(base_rot_inv, x_quat)
    # breakpoint()
    rel_rot6d = jax.vmap(quaternion_to_rotation_6d)(rel_rot.reshape(-1,4)).reshape(B, N, 6)
    
    # vel, ang_vel
    rel_vel_body = rotate_vec_batch(base_rot_inv, xd_vel)
    rel_ang_body = rotate_vec_batch(base_rot_inv, xd_ang)
    # rel_vel_body = jax.vmap(math.rotate)(xd_vel, base_rot_inv)
    # rel_ang_body = jax.vmap(math.rotate)(xd_ang, base_rot_inv)
    
    world_input = jnp.concatenate([
        rel_pos_body.reshape(B, -1),
        rel_rot6d.reshape(B, -1),
        rel_vel_body.reshape(B, -1),
        rel_ang_body.reshape(B, -1)
    ], axis=-1)
    
    # the shape should be N * (3 + 6 + 3 + 3) - 3 = 192
    
    return world_input

def _get_self_obs(x: Transform, xd: Motion) -> jax.Array:    
        if len(x.pos.shape) == 2:
            B = 1
            N = x.pos.shape[0]
            
            x_pos = x.pos[None,]
            x_quat = x.rot[None,]
            x_vel = xd.vel[None,]
            x_ang = xd.ang[None,]
            
        else:
            B, N, _ = x.pos.shape
        
        
            x_pos = x.pos
            x_quat = x.rot
            x_vel = xd.vel
            x_ang = xd.ang
        
        base_pos = x_pos[:, 0]
        base_rot_inv = math.quat_inv(x_quat[:, 0])
        
        rel_pos = x_pos - base_pos[:, None, :]
        # rel_pos_body = jax.vmap(math.rotate)(rel_pos[:, 1:], base_rot_inv) # remove the base pos
        rel_pos_body = rotate_vec_batch(base_rot_inv, rel_pos[:, 1:])
        # rot:
        # rel_rot = math.quat_mul(base_rot_inv[:, None, :], x_quat)
        rel_rot = jax.vmap(jax.vmap(math.quat_mul, in_axes=(None, 0)), in_axes=(0, 0))(base_rot_inv, x_quat)
        # breakpoint()
        rel_rot6d = jax.vmap(quaternion_to_rotation_6d)(rel_rot.reshape(-1,4)).reshape(B, N, 6)
        
        # vel, ang_vel
        rel_vel_body = rotate_vec_batch(base_rot_inv, x_vel)
        rel_ang_body = rotate_vec_batch(base_rot_inv, x_ang)
        # rel_vel_body = jax.vmap(math.rotate)(xd_vel, base_rot_inv)
        # rel_ang_body = jax.vmap(math.rotate)(xd_ang, base_rot_inv)
        
        self_obs = jnp.concatenate([
            rel_pos_body.reshape(B, -1),
            rel_rot6d.reshape(B, -1),
            rel_vel_body.reshape(B, -1),
            rel_ang_body.reshape(B, -1)
        ], axis=-1)

        return self_obs


def quat_integrate(quat: jnp.ndarray, ang_vel: jnp.ndarray, dt: float) -> jnp.ndarray:
    
    omega = jnp.concatenate([ang_vel, jnp.zeros(ang_vel.shape[:-1] + (1,), dtype=ang_vel.dtype)], axis=-1)

    delta_q = 0.5 * dt * jax.vmap(jax.vmap(math.quat_mul, in_axes=(0, 0)), in_axes=(0, 0))(quat, omega)
    new_q = quat + delta_q
    
    norm_new_q = jnp.linalg.norm(new_q, axis=-1, keepdims=True)
    new_q2 = new_q / norm_new_q
    
    return new_q2.reshape(quat.shape)

def make_world_predict_fn(
    world_network: FeedForwardNetwork
):
    def world_predict(params, deterministic: bool = False):
        def predict(states: jnp.ndarray, actions: types.Action):
            observation = transform_from_state_world(states)
            inputs = jnp.concatenate([observation, actions], axis=-1)
            # breakpoint() #(B, xxx)
            new_part_state = world_network.apply(*params, inputs) # composed of new_xd_vel, new_xd_ang
            #######################################################
            # TODO: integrate the delta state to get the next state
            B = states.shape[0]
            new_vel = new_part_state[:, :3*13].reshape(-1, 13, 3)
            new_ang = new_part_state[:, 3*13:].reshape(-1, 13, 3)

            last_x_pos = states[:, :3*13].reshape(-1, 13, 3)
            last_x_quat = states[:, 3*13: 7*13].reshape(-1, 13, 4)
            last_xd_vel = states[:, 7*13: 10*13].reshape(-1, 13, 3)
            last_xd_ang = states[:, 10*13:].reshape(-1, 13, 3)
            
            # integrate the delta state
            base_rot = last_x_quat[:, 0]
            new_xd_vel = rotate_vec_batch(base_rot, new_vel)
            new_xd_ang = rotate_vec_batch(base_rot, new_ang)
            
            # new_xd_vel = jax.vmap(math.rotate)(new_vel, base_rot).reshape(B, -1)
            # new_xd_ang = jax.vmap(math.rotate)(new_ang, base_rot).rehsape(B, -1)
            new_x_pos = last_x_pos + new_xd_vel * 1/50
            new_x_pos = new_x_pos.reshape(B, -1)
            new_xd_vel = new_xd_vel.reshape(B, -1)
            
            dt = 1/50
            new_x_quat = quat_integrate(last_x_quat, new_xd_ang, dt).reshape(B, -1)
            new_xd_ang = new_xd_ang.reshape(B, -1)
            # new_x_quat = math.integrate(last_x_quat, new_xd_ang * 1/50).reshape(B, -1)

            new_state = jnp.concatenate([
                new_x_pos,
                new_x_quat,
                new_xd_vel,
                new_xd_ang
            ], axis=-1)
            
            return new_state
            
        return predict
            
    return world_predict       
################################



def superdyno_train_onedevice(
    environment: Union[envs_v1.Env, envs.Env],
    episode_length: int,
    policy_updates: int,
    world_updates_per_epoch: int,
    policy_updates_per_epoch: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    horizon_length: int = 36,
    world_training_length: int = 8,
    policy_training_length: int = 32,
    batch_size: int = 256,
    num_envs: int = 1,
    num_evals: int = 1,
    action_repeat: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate_policy: float = 1e-5,
    learning_rate_world: float = 5e-4,
    adam_b: tuple[float, float] = (0.7, 0.95),
    use_schedule: bool = True,
    use_float64: bool = False,
    schedule_decay: float = 0.997,
    seed: int = 0,
    max_gradient_norm: float = 1e9,
    normalize_observations: bool = False,
    deterministic_eval: bool = False,
    policy_network_factory: types.NetworkFactory[
        apg_networks.APGNetworks
    ] = apg_networks.make_apg_networks,
    world_network_factory: types.NetworkFactory[
        MLP] = make_neural_world_models,
    # TODO: later we should write a world model network factory here.
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    xt = time.time()
    
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count
    
    num_updates_policy = policy_updates
    num_evals_after_init = max(num_evals - 1, 1)
    updates_per_epoch_policy = jnp.round(num_updates_policy / (num_evals_after_init))
    print("updates_per_epoch_policy", updates_per_epoch_policy)
    
    assert num_envs % device_count == 0
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    rng, global_key = jax.random.split(global_key, 2)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, eval_key = jax.random.split(local_key)
    
    env = environment
    if wrap_env:
        if wrap_env_fn is not None:
            wrap_for_training = wrap_env_fn
        elif isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(rng, num_envs // process_count)
            )
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )  # pytype: disable=wrong-keyword-args

    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    
    obs_size = env.observation_size
    if isinstance(obs_size, Dict):
        raise NotImplementedError('Dictionary observations not implemented in APG')
    action_size = env.action_size
    print("obs_size", obs_size)
    print("action_size", action_size)
    
    
    ####################################################
    # Create policy network and world network.
    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    policy_network = policy_network_factory(
        obs_size, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = apg_networks.make_inference_fn(policy_network)
    
    # NOTE: notice the policy forward
    # Define 'world_input_size' and 'world_output_size'
    world_input_size = 192 + 12
    world_output_size = 78
    world_network = world_network_factory(
        world_input_size, world_output_size, preprocess_observations_fn=normalize
    )
    # breakpoint()
    # %%
    world_predict = make_world_predict_fn(world_network)
    ######################################################
    if use_schedule:
        learning_rate_policy = optax.exponential_decay(
            init_value=learning_rate_policy, transition_steps=1, decay_rate=schedule_decay
        )
        
        learning_rate_world = optax.exponential_decay(
            init_value=learning_rate_world, transition_steps=1, decay_rate=schedule_decay
        )
    
    optimizer_policy = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=learning_rate_policy, b1=adam_b[0], b2=adam_b[1]),
    )
    
    optimizer_world = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=learning_rate_world, b1=adam_b[0], b2=adam_b[1]),
    )
    
    #### Create a buffer ####
    # The buffer size could be bigger
    buffer = ReplayBufferSuper(buffer_size=30000)
    
    def scramble_times(state, key):
        state.info['steps'] = jnp.round(
            jax.random.uniform(
                key,
                (
                    num_envs,
                ),
                maxval=episode_length,
            )
        )
        return state

    def env_step(
        carry: Tuple[Union[envs.State, envs_v1.State], PRNGKey],
        step_index: int,
        policy: types.Policy,
    ):
        # NOTE: we should restore the state, actions, next_state, and target.
        # But reward is also okay.
        env_state, key = carry
        key, key_sample = jax.random.split(key)
        # breakpoint()
        actions = policy(env_state.obs, key_sample)[0]
        # move the original action outside into the env step
        actions = jnp.clip(actions, -1.0, 1.0)
        # actions = env.action_loc + (actions * env.action_scale)
        # breakpoint()
        
        nstate, ref_state_vector, done, ref_qpos = env.step(env_state, actions)
        
        state_vector = extract_and_concat_state_info(env_state.pipeline_state)
        n_state_vector = extract_and_concat_state_info(nstate.pipeline_state)
        # breakpoint()
        transition_data = {
            'state': state_vector,
            'reference': ref_state_vector[0].reshape(ref_state_vector.shape[1], -1),
            'action': actions,
            'next_state': n_state_vector,
            'reward': nstate.reward[..., None],
            'done': done[...,None],
            'ref_qpos': ref_qpos[0],
        }
        # breakpoint()
        return (nstate, key), (transition_data, env_state.obs)
    
    ############## Loss Function Part ################
    ### Define how to calculate the loss or reward for world and policy model
    
    def get_policy_observation(state: jnp.ndarray, reference_state: jnp.ndarray, ref_qpos: jnp.ndarray):
        B = state.shape[0]
        xpos, xquat, xvel, xang = decompose_state(state)
        xpos = xpos.reshape(B, -1, 3)
        xquat = xquat.reshape(B, -1, 4)
        xvel = xvel.reshape(B, -1, 3)
        xang = xang.reshape(B, -1, 3)
        
        x = base.Transform(pos=xpos, rot=xquat)
        xd = base.Motion(vel=xvel, ang=xang)
        
        # qpos = mjx.inverse(env.sys, x)
        # breakpoint()
        inv_base_orientation = math.quat_inv(xquat[:,0])
        
        local_rpyrate = jax.vmap(math.rotate, in_axes=(0, 0))(xang[:,0],inv_base_orientation)
        # breakpoint()
        obs_list = []
        # yaw rate
        obs_list.append(jnp.array([local_rpyrate[:, 2]]).reshape(B, -1) * 0.25)
        # projected gravity
        v = jnp.array([0.0, 0.0, -1.0])
        v_batched = jnp.broadcast_to(v, (inv_base_orientation.shape[0], 3))
        obs_list.append(jax.vmap(math.rotate, in_axes=(0, 0))(v_batched,inv_base_orientation))

        # motor angles
        # angles = qpos[7:19]
        # obs_list.append(angles - env._default_ap_pose)
        # breakpoint()
        # self observation
        self_obs = _get_self_obs(x, xd)
        obs_list.append(self_obs)
        
        # obs_list.append(xd.vel.reshape(-1))
        # last action
        # obs_list.append(state_info['last_action']) # (12)
        # kinematic reference
        # ref_pos, ref_quat, ref_vel, ref_ang = decompose_state(reference_state)
        # ref_x = base.Transform(pos=ref_pos, rot=ref_quat)
        # ref_qpos = mjx.inverse(env.sys, ref_x)
        # kin_ref = env.kinematic_ref_qpos[jnp.array(state_info['steps']%self.l_cycle, int)] 
        obs_list.append(ref_qpos[:,7:]) # First 7 indicies are fixed # (12)

        obs = jnp.clip(jnp.concatenate(obs_list, axis=-1), -100.0, 100.0)

        return obs
        
    def world_model_training_step(carry, t, world_model):
        (curr_state, data_states, data_actions) = carry
        target_state = data_states[:, t]
        action = data_actions[:, t]
        
        pred_next_state = world_model(curr_state, action)
        
        # maybe here we need to observation transformation
        
        return (pred_next_state, data_states, data_actions), pred_next_state
            
        
    def policy_model_training_step(carry, t, policy, world_model):
        (curr_state, data_reference, ref_qpos_vector, key) = carry # we don't need env_state here
        key, key_sample = jax.random.split(key)
        
        reference_state = data_reference[:,t]
        ref_qpos = ref_qpos_vector[:,t]
        
        # TODO: transform the reference state into the observation
        obs = get_policy_observation(curr_state, reference_state, ref_qpos)
        actions = policy(obs, key_sample)[0]
        
        # observation transformation
        pred_next_state = world_model(curr_state, actions)
        
        # TODO: rewrite the world model into a class with step function... No more fn

        return (pred_next_state, data_reference, ref_qpos_vector, key), pred_next_state
    
    
    def policy_model_loss(policy_params, normalizer_policy_params, world_params, normalizer_world_params, policy_data, key):
        f = functools.partial(
            policy_model_training_step, policy=make_policy((normalizer_policy_params, policy_params)),
            world_model=world_predict(_unpmap((normalizer_world_params, world_params)))
        )
        (state_h, _, _, _), pred_states = jax.lax.scan(
            f, (policy_data['state'][:,0], policy_data['reference'], policy_data['ref_qpos'], key), (jnp.arange((policy_training_length - 1) // action_repeat))
        )

        # TODO: calculate the loss with pred_states and reference states
        policy_loss = compute_tracking_loss_from_state(pred_states, policy_data['next_state'][:,:-1])
        
        return policy_loss, policy_loss

    def world_model_loss(world_params, normalizer_world_params, world_data, key):
        # breakpoint()
        f = functools.partial(
            world_model_training_step, world_model=world_predict(_unpmap((normalizer_world_params, world_params)))
        )
        (state_h, _, _), pred_states = jax.lax.scan(
            f, (world_data['state'][:,0], world_data['state'], world_data['action']), (jnp.arange((world_training_length - 1) // action_repeat))
        )
        # breakpoint()
        # TODO: calculate the loss with pred_states and reference states
        world_loss = compute_tracking_loss_from_state(pred_states, world_data['next_state'][:,:-1])
        
        return world_loss, world_loss

    loss_grad_policy = jax.grad(policy_model_loss, has_aux=True)
    loss_grad_world = jax.grad(world_model_loss, has_aux=True)
    
    # NOTE: to use scan, I need to make sure the 'carry' interface is the same
    # for world model training, carry should be state, xs should be actions
    # for policy training, carry should be state, xs should be reference
    # Then in each world_training_step or policy_training_step, 
    # we need to customize the loss function
    
    
    ######################################
    
    def clip_by_global_norm(updates):
        g_norm = optax.global_norm(updates)
        trigger = g_norm < max_gradient_norm
        return jax.tree_util.tree_map(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm), updates
        )
        
    def minibatch_world_step(optimizer_world_state, normalizer_params, world_params, key, world_data):
        key, key_grad = jax.random.split(key)
        # breakpoint()
        world_start = time.time()
        grad, world_loss = loss_grad_world(
            world_params, normalizer_params, world_data, key_grad
        )
        
        grad = clip_by_global_norm(grad)
        grad = jax.lax.pmean(grad, axis_name='i')
        params_update, optimizer_world_state = optimizer_world.update(grad, optimizer_world_state)
        world_params = optax.apply_updates(world_params, params_update)
        world_end = time.time()
        world_training_time = world_end - world_start
        
        metrics = {
            'grad_norm': optax.global_norm(grad),
            'world_loss': world_loss,
            'world_training_time': world_training_time
        } 
        
        return (
            optimizer_world_state,
            normalizer_params,
            world_params,
            key,
        ), metrics
    
    def minibatch_policy_step(optimizer_policy_state, normalizer_params, policy_params, world_params, normalizer_world_params, key, policy_data):
        key, key_grad = jax.random.split(key)
        
        policy_start = time.time()
        grad, policy_loss = loss_grad_policy(
            policy_params, normalizer_params, world_params, normalizer_world_params, policy_data, key_grad
        )
        
        grad = clip_by_global_norm(grad)
        grad = jax.lax.pmean(grad, axis_name='i')        
        params_update, optimizer_policy_state = optimizer_policy.update(grad, optimizer_policy_state)
        policy_params = optax.apply_updates(policy_params, params_update)
        
        policy_end = time.time()
        policy_training_time = policy_end - policy_start
        
        metrics = {
            'grad_norm': optax.global_norm(grad),
            'policy_loss': policy_loss,
            'policy_training_time': policy_training_time
        }
        
        return (
            optimizer_policy_state,
            normalizer_params,
            policy_params,
            key
        ), metrics
        

    def training_epoch(
        training_world_state: TrainingState,
        training_policy_state: TrainingState,
        env_state: Union[envs.State, envs_v1.State],
        key: PRNGKey,
    ):
        """
        In each training epoch, we will do the following three steps:
        (1) collect data from the environment
        (2) update the world model network
        (3) update the policy network
        """
        # # 1. TODO: collect data from the environment
        # # named: env_world_data, env_policy_data
        # # we need to collect the (state, reference, action, reward) data
        # def collect_env_data(policy_params, normalizer_params, env_state, key):
        #     f = functools.partial(
        #         env_step, policy=make_policy((normalizer_params, policy_params))
        #     )
            
        #     (final_states, _), (transitions, obs) = jax.lax.scan(
        #         f, (env_state, key), (jnp.arange(horizon_length // action_repeat))
        #     )
            
        #     # update the normalizer params
        #     normalizer_params = running_statistics.update(
        #         normalizer_params, obs, pmap_axis_name=_PMAP_AXIS_NAME
        #     )
            
        #     return final_states, transitions, normalizer_params
        
        # state_h, transition_data, normalizer_policy_params = collect_env_data(
        #     training_policy_state.network_params,
        #     training_policy_state.normalizer_params,
        #     env_state,
        #     key,
        # )
        
        # # training_policy_state.normalizer_params = normalizer_params_policy
        
        # # shape of transition_data
        # # state: [T, B, 169]
        # # reference: [T, B, 169]
        # # action: [T, B, 12]
        # # next_state: [T, B, 169]
        # # reward[T, B]
        # # done: [T, B]
        
        
        # # 2. TODO(Optional): restore the transition data into buffer
        # # then sample data from the buffer for policy/world training
        # # or we can directly use the transition data for training
        # # filter out terminated trajectories
        # # breakpoint()
        # done_except_last = transition_data['done']
        # valid_mask = jnp.all(done_except_last == 0, axis=0)[:,0]
        # jax.debug.print("{}", valid_mask)
        # # breakpoint()
        # def mask_fn(x):
        #     # x: [T, B, ...] -> keep B where valid_mask is True
        #     indices = jnp.where(valid_mask)[0]
        #     return jnp.take(x, indices, axis=1)
        #     # return x[:, valid_mask]
        #     # return jax.vmap(lambda x_b, keep: jax.lax.select(keep, x_b, jnp.zeros_like(x_b)), in_axes=(1, 0))(x, valid_mask).T

        # filtered_transition_data = {
        #     k: mask_fn(v) if v.ndim >= 2 else v  # only mask T-B shaped arrays
        #     for k, v in transition_data.items()
        # }
        
        # filtered_transition_data['done'] = filtered_transition_data['done'].at[0, -1, :].set(1)
        # _, T,  B = filtered_transition_data['done'].shape
        # filtered_transition_data = { k: v.reshape( B * T, -1) for k, v in filtered_transition_data.items() }
        # jax.debug.print("added_transition_data:{}", B*T)
        # # breakpoint()
        # # breakpoint()
        # # valid_idx = jnp.where(valid_mask)[0]
        # # valid_idx = jax.device_get(valid_idx)
        # # def mask_fn(x):
        # #     return x[:, valid_idx]
        # # NOTE: hot fix for the filtering
        # # filtered_transition_data = transition_data
        # # filtered_transition_data = {k: mask_fn(v) for k, v in transition_data.items()}
        # buffer.add_traj(filtered_transition_data)
            
        # 3. sample (state, actions, next_state) from buffer to train the world model
        metrics = {
            'world_training_loss': 0.0,
            'policy_training_loss': 0.0,
            'world_training_time': 0.0,
            'policy_training_time': 0.0,
        }
        
        world_key = jax.random.PRNGKey(0)
        world_name_list  = ['state', 'action', 'next_state']
        batch = buffer.sample(
            name_list=world_name_list,
            rollout_length=world_training_length,
            batch_size=batch_size,
            minibatch_num=world_updates_per_epoch,
            key=world_key,
        )
        batched = batchify(batch, batch_size=batch_size)
        
        normalizer_world_params = training_world_state.normalizer_params
        optimizer_world_state = training_world_state.optimizer_state
        world_params = training_world_state.network_params
        # breakpoint()
        for i in range(batched['state'].shape[0]):
            inputs = {k: v[i] for k, v in batched.items()}
            # TODO: world_model_training_step
            (
                optimizer_world_state,
                normalizer_world_params,
                world_params,
                world_key,
            ), metrics_w = minibatch_world_step(
                                optimizer_world_state,
                                normalizer_world_params,
                                world_params,
                                world_key,
                                inputs,
                            )      
            metrics['world_training_time'] += metrics_w['world_training_time']
            metrics['world_training_loss'] += metrics_w['world_loss'] / world_updates_per_epoch
        print('world_training_loss', metrics['world_training_loss'])
        print('world_training_time', metrics['world_training_time'])
        # 4. sample (state, referece) from buffer the train the policy model
        policy_key = jax.random.PRNGKey(14)
        policy_name_list = ['state', 'reference', 'ref_qpos', 'next_state']
        batch = buffer.sample(
            name_list=policy_name_list,
            rollout_length=policy_training_length,
            batch_size=batch_size,
            minibatch_num=policy_updates_per_epoch,
            key=policy_key,
        )
        batched = batchify(batch, batch_size=batch_size)
        
        normalizer_policy_params = training_policy_state.normalizer_params
        optimizer_policy_state = training_policy_state.optimizer_state
        policy_params = training_policy_state.network_params
        # for i in range(batched['state'].shape[0]):
        #     inputs = {k: v[i] for k, v in batched.items()}
        #     # TODO: policy_training_step
        #     (
        #         optimizer_policy_state,
        #         normalizer_policy_params,
        #         policy_params,
        #         policy_key,
        #     ), metrics_p = minibatch_policy_step(
        #                         optimizer_policy_state,
        #                         normalizer_policy_params,
        #                         policy_params,
        #                         world_params,
        #                         normalizer_world_params,
        #                         policy_key,
        #                         inputs,
        #                     )
        #     metrics['policy_training_time'] += metrics_p['policy_training_time']
        #     metrics['policy_training_loss'] += metrics_p['policy_loss'] / policy_updates_per_epoch
        # print('policy_training_loss', metrics['policy_training_loss'])
        # print('policy_training_time', metrics['policy_training_time'])
        
        ### TODO: pass out the policy and world parameters
        # maybe we need a better name for the namespace
        training_world_state_next = TrainingState(
            optimizer_state = optimizer_world_state,
            normalizer_params = normalizer_world_params,
            network_params = world_params,
        )

        training_policy_state_next = TrainingState(
            optimizer_state = optimizer_policy_state,
            normalizer_params = normalizer_policy_params,
            network_params = policy_params,
        )
        
        
        return training_world_state_next, training_policy_state_next, metrics, key
    
    # training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)
    
    training_walltime = 0
    
    def training_epoch_with_timing(
        training_world_state: TrainingState,
        training_policy_state: TrainingState,
        env_state: Union[envs.State, envs_v1.State],
        key: PRNGKey,
    )-> Tuple[TrainingState, TrainingState, Union[envs.State, envs_v1.State], Metrics, PRNGKey]:
        
        #------------------------------------------
        # 1. TODO: collect data from the environment
        # named: env_world_data, env_policy_data
        # we need to collect the (state, reference, action, reward) data
        def collect_env_data(policy_params, normalizer_params, env_state, key):
            f = functools.partial(
                env_step, policy=make_policy((normalizer_params, policy_params))
            )
            
            (final_states, _), (transitions, obs) = jax.lax.scan(
                f, (env_state, key), (jnp.arange(horizon_length // action_repeat))
            )
            
            # update the normalizer params
            normalizer_params = running_statistics.update(
                normalizer_params, obs, pmap_axis_name=_PMAP_AXIS_NAME
            )
            
            return final_states, transitions, normalizer_params
        
        state_h, transition_data, normalizer_policy_params = collect_env_data(
            training_policy_state.network_params,
            training_policy_state.normalizer_params,
            env_state,
            key,
        )
        
        
        # shape of transition_data
        # state: [T, B, 169]
        # reference: [T, B, 169]
        # action: [T, B, 12]
        # next_state: [T, B, 169]
        # reward[T, B]
        # done: [T, B]
        
        
        # 2. TODO(Optional): restore the transition data into buffer
        # then sample data from the buffer for policy/world training
        # breakpoint()
        done_except_last = transition_data['done']
        valid_mask = jnp.all(done_except_last == 0, axis=0)[:,0]
        jax.debug.print("{}", valid_mask)
        # breakpoint()
        def mask_fn(x):
            # x: [T, B, ...] -> keep B where valid_mask is True
            indices = jnp.where(valid_mask)[0]
            return jnp.take(x, indices, axis=1)
            # return x[:, valid_mask]
            # return jax.vmap(lambda x_b, keep: jax.lax.select(keep, x_b, jnp.zeros_like(x_b)), in_axes=(1, 0))(x, valid_mask).T

        filtered_transition_data = {
            k: mask_fn(v) if v.ndim >= 2 else v  # only mask T-B shaped arrays
            for k, v in transition_data.items()
        }
        
        filtered_transition_data['done'] = filtered_transition_data['done'].at[0, -1, :].set(1)
        _, T,  B = filtered_transition_data['done'].shape
        filtered_transition_data = { k: v.reshape( B * T, -1) for k, v in filtered_transition_data.items() }
        jax.debug.print("added_transition_data:{}", B*T)
        # breakpoint()
        # breakpoint()
        # valid_idx = jnp.where(valid_mask)[0]
        # valid_idx = jax.device_get(valid_idx)
        # def mask_fn(x):
        #     return x[:, valid_idx]
        # NOTE: hot fix for the filtering
        # filtered_transition_data = transition_data
        # filtered_transition_data = {k: mask_fn(v) for k, v in transition_data.items()}
        buffer.add_traj(filtered_transition_data)
        env_state = state_h
        
        training_policy_state_next = TrainingState(
            optimizer_state = training_policy_state.optimizer_state,
            normalizer_params = normalizer_policy_params,
            network_params = training_policy_state.network_params,
        )
        
        training_policy_state = training_policy_state_next
        
        #---------------------------------------------------------
        
        
        
        nonlocal training_walltime
        t = time.time()
        (
            training_world_state,
            training_policy_state,
            metrics,
            key,
        ) = training_epoch(
            training_world_state, training_policy_state, env_state, key
        )
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t

        training_walltime += epoch_training_time
        sps = (updates_per_epoch_policy * num_envs * horizon_length) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        print('training_metrics', metrics)
        return training_world_state, training_policy_state, env_state, metrics, key  # pytype: disable=bad-return-type  # py311-upgrade

    ##### Initialize world / policy networks and training states #######
    # initialize the network as the same for different processes.
    policy_params = policy_network.policy_network.init(global_key)
    world_params = world_network.init(global_key)
    # breakpoint()
    del global_key
    
    dtype = 'float64' if use_float64 else 'float32'
    # TODO: specify the world model input size
    # breakpoint()
    training_world_state = TrainingState(
        optimizer_state=optimizer_world.init(world_params),
        network_params=world_params,
        normalizer_params=running_statistics.init_state(
            specs.Array((world_input_size,), jnp.dtype(dtype))
        ),
    )
    # training_world_state = jax.device_put_replicated(
    #     training_world_state, jax.local_devices()[:local_devices_to_use]
    # )
    
    training_policy_state = TrainingState(
        optimizer_state=optimizer_policy.init(policy_params),
        network_params=policy_params,
        normalizer_params=running_statistics.init_state(
            specs.Array((env.observation_size,), jnp.dtype(dtype))
        ),
    )
    
    # training_policy_state = jax.device_put_replicated(
    #     training_policy_state, jax.local_devices()[:local_devices_to_use]
    # )
    

    
    # NOTE: begin to evaluate the policy
    if not eval_env:
        eval_env = environment
    if wrap_env:
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(rng, num_eval_envs // process_count)
            )
        eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )
    
    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )
    
    # Run initial eval
    metrics = {}
    
    # if process_id == 0 and num_evals > 1:
    #     metrics = evaluator.run_evaluation(
    #         _unpmap(
    #             (training_policy_state.normalizer_params, training_policy_state.network_params)
    #         ),
    #         training_metrics={},
    #     )
    #     logging.info(metrics)
    #     progress_fn(0, metrics)
    
    init_key, scramble_key, local_key = jax.random.split(local_key, 3)
    init_key = jax.random.split(
        init_key, ( num_envs // process_count)
    )
    env_state = reset_fn(init_key)
    
    env_state = scramble_times(env_state, scramble_key)
    env_state, ref_state, done, ref_qpos = step_fn(
        env_state,
        jnp.zeros(
            (num_envs // process_count, env.action_size),
        ),
    )
    
    epoch_key, local_key = jax.random.split(local_key)
    epoch_key, _ = jax.random.split(epoch_key)
    # breakpoint()
    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)

        for sub_iter in range(int(jnp.round(updates_per_epoch_policy / policy_updates_per_epoch))):
            (training_world_state, training_policy_state, env_state, training_metrics, epoch_key) = (
                training_epoch_with_timing(training_world_state, training_policy_state, env_state, epoch_key)
            )
        
        if process_id == 0:
            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_policy_state.normalizer_params, training_policy_state.network_params)
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(it + 1, metrics)
    
    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_world_state)
    pmap.assert_is_replicated(training_policy_state)
    world_params = _unpmap(
        (training_world_state.normalizer_params, training_world_state.network_params)
    )
    policy_params = _unpmap(
        (training_policy_state.normalizer_params, training_policy_state.network_params)
    )
    
    pmap.synchronize_hosts()
    return (make_policy, world_params, policy_params, metrics)
    
    
    
    
    
    
    

