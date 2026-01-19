# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import re
import ray
import hydra
import torch
import numpy as np

from pprint import pprint
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.reward_score import qa_em_simple as qa_em
from verl.utils.fs import copy_local_path_from_hdfs
from verl.trainer.ppo.ray_trainer_self_ext import RayPPOTrainer, ResourcePoolManager, Role

def _select_rm_score_fn(data_source):
    impleted_list = [
        'nq', 
        'triviaqa', 
        'popqa', 
        'web_questions', 
        'hotpotqa', 
        '2wikimultihopqa', 
        'musique', 
        'bamboogle', 
        'strategyqa',
        '2wikimultihop', 
        'multihoprag',
        'strategyqa',
        'self_propose'
        ]
    if data_source in impleted_list:
        return qa_em.compute_score_em
    else:
        raise ValueError(f'Unsupported data source {data_source} for reward score!')

class RewardManager():
    """
        The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto, is_debug=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}
        if is_debug:
            print('#'*20)
            print(data[0:20].non_tensor_batch['index'])
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem            

            prompt_ids = data_item.batch['prompts']
            uid = data_item.non_tensor_batch['index']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
            #                          format_score=self.format_score, is_debug=(i<=10) and is_debug)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                     format_score=self.format_score, is_debug=(i<=10)) # and is_debug

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
           

        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor


# Hydra 框架的装饰器，用于管理应用程序的配置 
# config_path='config' 指定配置文件所在的目录（通常是 config/ 文件夹）
# config_name='proto' 指定主配置文件的名称（即 config/proto.yaml）
# version_base=None 禁用 Hydra 的版本检查


@hydra.main(config_path='config', config_name='proto', version_base=None) 
def main(config):
    # '''
    # Hydra 装饰的主函数
    # config 参数会自动从指定的 YAML 配置文件加载
    # '''
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
        # '''
        # 检查 Ray 是否已初始化，如果没有则初始化 ray.init() Ray 集群, 会检测所有 8 块 GPU，并初始化 Ray 运行

        # runtime_env 设置了环境变量：

        # TOKENIZERS_PARALLELISM='true'：允许 tokenizers 并行处理

        # NCCL_DEBUG='WARN'：设置 NCCL (NVIDIA Collective Communications Library) 的调试级别为 WARN

        # '''

    ray.get(main_task.remote(config))

    # '''
    # main_task.remote(config) 将 main_task 函数作为远程任务提交到 Ray 集群

    # ray.get() 等待任务完成并获取结果

    # .remote() 是 Ray 的语法，表示将函数调用分发到远程 worker 执行
    # '''

@ray.remote
def main_task(config):
    # '''
    # @ray.remote 将这个函数标记为 分布式可执行任务。
    # 当调用 main_task.remote(config) 时，Ray 不会立即执行它，而是：
    # 将函数和参数序列化。
    # 提交到 Ray 集群的任务队列中。
    # 返回一个 ObjectRef（未来对象的引用）。
    # '''

    # print initial config
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    # '''
    # OmegaConf.to_container(config, resolve=True)：
    # 将 Hydra/OmegaConf 的配置对象 config 转换为 Python 原生容器（如 dict）。
    # resolve=True 表示 解析所有变量插值（如 ${data.path} → 实际路径）。
    # 此处的解析只是为了打印,下面才是真正使用的config解析.
    # '''
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path) # bash 中的命令行参数会覆盖 config 文件中的默认值。这是 Hydra 框架的默认行为，也是其核心功能之一：通过命令行动态覆盖配置。
    print('*' * 10)
    print(local_path)
    print('*' * 10)
    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)
    # '''
    # 不是所有模型都有官方的 hf_tokenizer，但绝大多数主流模型都有
    # '''

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp': # FSDP 策略分支
        # '''
        #     导入的类：

        #     ActorRolloutRefWorker：负责生成 Actor 的轨迹（rollout）的 Worker。

        #     CriticWorker：负责计算价值函数的 Worker。

        #     RayWorkerGroup：管理 Ray 分布式任务的控制器类。

        #     适用场景：

        #     FSDP 是 PyTorch 的完全分片数据并行策略，适合 单机多卡或多机训练大模型。

        #     典型用例：训练参数量超过单卡显存容量的模型（如 7B+ LLM）。
        # '''

        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers_self import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron': # Megatron 策略分支

        # '''
        #     导入的类：
        #     ActorRolloutRefWorker 和 CriticWorker：Megatron 定制实现的 Worker。
        #     NVMegatronRayWorkerGroup：支持 Megatron 的 Ray 控制器（可能集成 NVIDIA 的优化）。
        #     适用场景：
        #     Megatron-LM 是 NVIDIA 的分布式训练框架，支持 张量并行、流水线并行。
        #     适合超大规模模型训练（如 100B+ 参数）。        
        # '''
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    
    # '''
    #     ResourcePoolManager：
    #     管理 Ray 集群的资源池（如 GPU 分组分配），确保不同角色的 Worker（Actor、Critic 等）按需获取资源。
    #     Role：
    #     枚举类，定义训练中的角色类型（如 ActorRollout 生成轨迹，Critic 计算价值函数）。
    # '''

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),

        Role.ExternalRollout: ray.remote(ActorRolloutRefWorker),
    }
    # '''
    #     ray.remote(...) 将普通类转换为 分布式可调用的 Ray Actor。
    #     ActorRolloutRefWorker 被复用于 ActorRollout 和 RefPolicy 角色（可能共享相同实现但不同配置）。
    # '''

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    # '''
    # 资源池结构：
    # global_pool_id：资源池名称（此处只有一个全局池）。
    # resource_pool_spec：定义资源池的实际资源，格式为 {pool_id: [gpu_per_node] * num_nodes}。
    # 例如，若 n_gpus_per_node=8 且 nnodes=2，则池子包含 [8, 8]，表示 2 个节点，每节点 8 GPU。
    # '''

    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,

        Role.ExternalRollout: global_pool_id,
    }
    # '''
    #     指定每个角色从哪个资源池获取资源。
    #     所有角色共享全局资源池（global_pool），由 Ray 动态调度。
    # '''

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    # '''
    #     resource_pool_spec：定义全局资源池（如 {'global_pool': [8, 8]} 表示 2 节点 × 8 GPU）。
    #     mapping：指定每个角色（Actor、Critic、RewardModel）使用的资源池。
    
    # '''
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()






