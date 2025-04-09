import jax.numpy as jnp
import jax

class IndexCounter:
    @staticmethod
    def sample_rollout(feasible_index, batch_size, rollout_length, key):
        indices = jax.random.permutation(key, feasible_index.shape[0])[:batch_size]
        begin_idx = feasible_index[indices][:, None]
        bias = jnp.arange(rollout_length)[None, :]
        return begin_idx + bias
    
    @staticmethod
    def calculate_feasible_index(done_flag: jnp.ndarray, rollout_length: int, total_count: int):
        res_flag = jnp.ones_like(done_flag, dtype=bool)
        done_idx = jnp.where(done_flag != 0, size=done_flag.shape[0])[0]
        bias = jnp.arange(rollout_length)[None, :]
        mask_idx = done_idx[:, None] - bias
        mask_idx = mask_idx.flatten()
        mask_idx = jnp.clip(mask_idx, 0, done_flag.shape[0] - 1)
        res_flag = res_flag.at[mask_idx].set(False)
        
        over_bound_mask = jnp.arange(done_flag.shape[0]) >= total_count
        res_flag = jnp.where(over_bound_mask, False, res_flag)

        # 返回所有为 True 的 index
        return jnp.where(res_flag, size=done_flag.shape[0])[0]
        # if total_count < len(done_flag):
        #     res_flag = res_flag.at[total_count:].set(False)
        
        # return jnp.where(res_flag)[0]
    # @staticmethod
    # def sample_rollout(feasible_index: jnp.ndarray, batch_size: int, rollout_length: int, key: jax.Array):
        
    #     # 随机打乱并选取前 batch_size 个 index
    #     key, subkey = jax.random.split(key)
    #     perm = jax.random.permutation(subkey, feasible_index.shape[0])
    #     chosen_idx = feasible_index[perm[:batch_size]]  # [B]

    #     # 构造偏移：[0, 1, ..., rollout_length-1]
    #     bias = jnp.arange(rollout_length)[None, :]  # [1, L]
    #     rollout_indices = chosen_idx[:, None] + bias  # [B, L]

    #     return rollout_indices  # [B, L], 每行是一个完整 rollout 的索引
    # @staticmethod
    # def calculate_feasible_index(done_flag: jnp.ndarray, rollout_length: int, total_count: int):
        
    #     N = done_flag.shape[0]

    #     # (N - rollout_length + 1, rollout_length)
    #     def make_mask(start):
    #         return jnp.all(done_flag[start:start + rollout_length] == 0)

    #     feasible = jax.vmap(make_mask)(jnp.arange(N - rollout_length + 1))

    #     # 去除超出 total_count 范围的 index
    #     feasible = jnp.where(jnp.arange(N - rollout_length + 1) + rollout_length <= total_count, feasible, False)

    #     return jnp.where(feasible)[0]

    
class ReplayBufferSuper:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self._head = 0
        self._total_count = 0
        self._data_buf = None
        
    def _init_data_buf(self, trajectory: dict):
        self._data_buf = {}
        for k, v in trajectory.items():
            v_shape = v.shape[1:]
            self._data_buf[k] = jnp.zeros((self.buffer_size, ) + v_shape, dtype=v.dtype)
    
    def add_traj(self, trajectory: dict):
        if self._data_buf is None:
            self._init_data_buf(trajectory)
            
        n = trajectory['done'].shape[0]
        store_n = min(n, self.buffer_size - self._head)
        # breakpoint()
        for key in self._data_buf:
            curr_n = trajectory[key].shape[0]
            assert (n == curr_n)
            
            self._data_buf[key] = self._data_buf[key].at[self._head:self._head + store_n].set(trajectory[key][:store_n])
            remainder = n - store_n
            
            if remainder > 0:
                self._data_buf[key] = self._data_buf[key].at[0:remainder].set(trajectory[key][store_n:])
        
        self._head = (self._head + n) % self.buffer_size
        self._total_count = min(self._total_count + n, self.buffer_size)
    
    def feasible_index(self, rollout_length: int):
        done_flag = self._data_buf['done']
        feasible_index = IndexCounter.calculate_feasible_index(done_flag, rollout_length, self._total_count)
        return feasible_index
    
    def sample(self, name_list, rollout_length, batch_size, minibatch_num, key):
        feasible_idx = self.feasible_index(rollout_length)
        full_batch = batch_size * minibatch_num
        sample_key, subkey = jax.random.split(key)
        
        sampled_indices = IndexCounter.sample_rollout(feasible_idx, full_batch, rollout_length, subkey)
        sampled_indices = sampled_indices % self.buffer_size
        
        result = {
            name: jnp.take(self._data_buf[name], sampled_indices, axis=0)
            for name in name_list
        }

        return result # dict of shape (name: (B*L, ...))
    
    
    
    
    
    