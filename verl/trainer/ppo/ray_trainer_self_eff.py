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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import re
import gc
import os
import math
import uuid
import copy
import json
import torch
import random
import string
import requests
import numpy as np
import pandas as pd
from enum import Enum
from tqdm import tqdm
from pprint import pprint
from codetiming import Timer
from typing import Type, Dict
from typing import List, Union
from omegaconf import ListConfig
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from omegaconf import OmegaConf, open_dict
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F


from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.reward_score import qa_em_simple
import verl.utils.torch_functional as verl_F
from verl.single_controller.base import Worker
from verl.utils.torch_functional import masked_mean
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray.base import create_colocated_worker_cls
from search_r1.llm_agent.generation_self import LLMGenerationManager, GenerationConfig
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance




from tools.seed_tool import SeedTool
from tools.prefix_tool import PrefixTool


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    ExternalRollout = 8


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]





def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 is_debug=False,):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id,
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.is_debug = is_debug

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)
        self.prefix_tool = PrefixTool(prefix_path=self.config.prompt_prefix_path)
        # self._create_dataloader()
        self._my_init()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
    def _my_init(self):       

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if self.is_debug:
            print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
    def _create_dataloader(self):
        raise ValueError("DataLoader creation is moved to SEED in self-play setting.")
        # TODO: we have to make sure the batch size is divisible by the dp size
        
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        
        if self.is_debug:
            print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        if self.is_debug:
            print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        if self.is_debug:
            print(f'Size of train dataloader: {len(self.train_dataloader)}')
            print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if self.is_debug:
            print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


    def _seed_loader(self,global_end,data_num = None):
        if data_num is None:
            data_num=self.config.data.train_batch_size
        self.seed_dataset = SeedData(tokenizer=self.tokenizer,
                                        data_num=data_num,
                                        seed_num = self.config.data.seed_size,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='error',
                                        prompt_prefix=self.prefix_tool.get_prefix(prefix_type='proposer'),
                                        nltk_data_path = self.config.nltk_data_path, 
                                        statrt_index = global_end)
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.seed_dataset.dataframe):
                print(f"[WARNING] seed dataset size is smaller than desired size. Using the dataset as the original size {len(self.seed_dataset.dataframe)}")
            else:
                self.seed_dataset.dataframe = self.seed_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        if self.is_debug:
            print(f"filtered seed dataset size: {len(self.seed_dataset.dataframe)}")

        self.seed_dataloader = DataLoader(dataset=self.seed_dataset,
                                           batch_size=int(self.config.data.train_batch_size),
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)
        return global_end + self.config.data.train_batch_size
    def train_generation(self,id_l= [],q_l=[],a_l=[],i_l=[]):
        '''
        tokenizer: PreTrainedTokenizer,
        data_num: int =1024,
        prompt_key='prompt',
        max_prompt_length=1024,
        filter_prompts=True,
        chat_template_func=None,
        return_raw_chat=False,
        truncation='error',
        prompt_prefix = '',
        positive_prompt_prefix='',
        negative_prompt_prefix='',
        q_l=[],
        a_l=[],
        i_l=[]
        '''
        self.train_dataset = TrainData(tokenizer=self.tokenizer,
                                    data_num=self.config.data.train_batch_size,
                                    prompt_key=self.config.data.prompt_key,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    filter_prompts=True,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation='error',
                                    prompt_prefix=self.prefix_tool.get_prefix(prefix_type='solver'),
                                    positive_prompt_prefix=self.prefix_tool.get_prefix(prefix_type='p_external'),
                                    negative_prompt_prefix=self.prefix_tool.get_prefix(prefix_type='n_external'),
                                    id_l = id_l,
                                    q_l=q_l,
                                    a_l=a_l,
                                    i_l=i_l)
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] train dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        if self.is_debug:
            print(f"filtered train dataset size: {len(self.train_dataset.dataframe)}")

        return DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)
    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        raise ValueError("Self-Validation not supported in self-play setting.")
        reward_tensor_lst = []
        data_source_lst = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ExternalRollout)
        eactor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ExternalRollout],
                                                    config=self.config.actor_rollout_ref,
                                                    role='ext')
        self.resource_pool_to_cls[resource_pool]['ext'] = eactor_rollout_cls
        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        self.external_model = all_wg['ext']
        self.external_model.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def extract_qa_from_batch(self, proposer_batch: DataProto):
        # 使用 with torch.no_grad() 避免计算图保留
        with torch.no_grad():
        
            q_l = []
            a_l = []
            i_l = []
            print_flag = 0
            for i in range(len(proposer_batch)):
                data_item = proposer_batch[i]
                
                
                
                # print('####[Debug] Processing batch item ###'*3)
                # print(data_item.batch.keys())
                prompt_ids = data_item.batch['prompts']
                response_ids = data_item.batch['responses']
                attention_mask = data_item.batch['attention_mask']
                
                # 后续处理使用CPU张量
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = attention_mask[:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                
                valid_response_length = attention_mask[prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                # 解码
                proposed_sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                proposed_str = self.tokenizer.decode(proposed_sequences.numpy())
                
                # information_sequences = torch.cat((prompt_ids, response_ids))
                # sequences_str = self.tokenizer.decode(information_sequences.numpy())
                
                q_str, a_str, info_str = self._extract_qa_internal(proposed_str, proposed_str)
                q_l.append(q_str)
                a_l.append(a_str)
                i_l.append(info_str)
                if print_flag == 0 and q_str != '' and a_str != '' and self.is_debug:
                    print(f'[Info] Extracted QA pair:\n###############Q: {q_str}\n###############A: {a_str}\n###############Info: {info_str}\n---')
                    print(f'[Info] Full proposed_str:\n{proposed_str}\n---###############')
                    # print(f'[Info] Full sequences_str:\n{sequences_str}\n---###############')
                    print_flag = 1
        return q_l, a_l, i_l
    def _extract_qa_internal(self, solution_str, all_str):
            
            """Extract the equation from the solution string."""

            def remove_non_printable_translate(text):
                """消除非英语内容, 使用translate方法, 性能最好"""
                # 创建转换表：非printable字符映射为None
                non_printable_chars = set(text) - set(string.printable)
                translation_table = str.maketrans('', '', ''.join(non_printable_chars))
                return text.translate(translation_table)
            

            qa_pattern = r'<answer>(.*?)</answer>'
            match = re.finditer(qa_pattern, solution_str, re.DOTALL)
            matches = list(match)
            
            # 这时候其实是我的prompt里的例子
            if len(matches) <= 1:
                return ('', '', '')
            # 最后一个是模型形成的最终答案
            final_str = matches[-1].group(1).strip()
            # If there are 2 or more matches, return the last one
            """根据#*#标记切分字符串"""
            
            parts = final_str.split('#*#', 1)
            if len(parts) > 1:
                q_str = parts[0]
                a_str = parts[1]
                if a_str.count('<answer>') >1 or a_str.count('John Doe') >=1:
                    return ('', '', '')
                info_pattern = r'<search>(.*?)</search>'
                info_match = re.finditer(info_pattern, all_str, re.DOTALL)
                info_matches = list(info_match)[1:]
                info_str =''
                for info in info_matches:
                    info_str = remove_non_printable_translate(info.group(1).strip()) + '##' + info_str # 越靠后的的搜索结果越接近出题时的重点
                q_str = remove_non_printable_translate(q_str)
                a_str = remove_non_printable_translate(a_str)
                return q_str, a_str, info_str
            else:
                return ('', '', '')
    
 
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0

        
        # self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            proposer_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            
        )

        q_ne_l = []
        a_ne_l = []
        i_ne_l = []
        idx_ne_l = []
        replace_ratio = 0.1
        negative_ratio = 0.25 # 1:4
        buffer_size =1024
        max_retry_num = 1
        negative_sample_num = int(buffer_size * replace_ratio * negative_ratio)

        

        global_end = 0
        proposed_pool = Proposer_pool(pad_token_id=self.pad_token_id,divisor=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes)
        # start training loop
        with tqdm(total=self.total_training_steps) as pbar:

            for epoch in range(self.config.trainer.total_epochs):
                self.global_steps += 1 # we start from step 1
                pbar.update(1)
                pbar.set_description(f"Processing epoch={epoch} step={self.global_steps} \
                                        of {self.config.trainer.total_epochs} epochs \
                                            {self.total_training_steps} steps")
                
                proposed_pool.clear_all()

                if len(q_ne_l) > 0:
                    remain_size = int(buffer_size*(1-replace_ratio))
                    # 将三个列表组合成一个元组列表
                    combined = list(zip(idx_ne_l, q_ne_l, a_ne_l, i_ne_l))
                    # 对组合后的列表进行 shuffle
                    random.shuffle(combined)
                    # 截断
                    combined = combined[:remain_size]
                    # 将 shuffle 后的数据解压回三个列表
                    idx_ne_l, q_ne_l, a_ne_l, i_ne_l = zip(*combined)
                    # 如果你需要结果是列表而不是元组，可以转换一下
                    idx_ne_l = list(idx_ne_l)
                    q_ne_l = list(q_ne_l)
                    a_ne_l = list(a_ne_l)
                    i_ne_l = list(i_ne_l)


                if not (len(q_ne_l) == len(a_ne_l) == len(i_ne_l) == len(idx_ne_l)):
                    raise ValueError(f"Length mismatch q_nel:{len(q_ne_l)} a_nel:{len(a_ne_l)}  i_nel:{len(i_ne_l)}  idx_nel:{len(idx_ne_l)}.")
                
                q_len_before = 0
                stall_times = 0

                idx_ut_l = []
                idx_ab_l = []
                while len(q_ne_l) < buffer_size:
                    q_len_before = len(q_ne_l)
                    new_end = self._seed_loader(global_end) #, data_num= int(buffer_size * replace_ratio * 2)
                    global_end = new_end
                    for seed_batch_dict in self.seed_dataloader:
                        metrics = {}
                        timing_raw = {}
                        seed_batch: DataProto = DataProto.from_single_dict(seed_batch_dict)

                        # # 预览第一项 ##################################################
                        if self.is_debug:
                            print_data(seed_batch, 0, info='original seed_batch')
                        # #/ 预览第一项 ##################################################

                        # pop those keys for generation
                        seed_gen_batch = seed_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                        # proposer rollout
                        first_input_ids = seed_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        
                        generation_manager.timing_raw = timing_raw
                        proposer_batch_output = generation_manager.run_llm_loop(
                            gen_batch=seed_gen_batch,
                            initial_input_ids=first_input_ids,
                            is_proposer=True,
                        )

                         # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in proposer_batch_output.batch.keys():
                            proposer_batch_output.batch[key] = proposer_batch_output.batch[key].long()

                        with torch.no_grad():
                            thisoutput = self.actor_rollout_wg.compute_log_prob(proposer_batch_output)
                            proposer_batch_output = proposer_batch_output.union(thisoutput)

                         # repeat to align with repeated responses in rollout


                        seed_batch = seed_batch.union(proposer_batch_output)

                        # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                        for key in seed_batch.batch.keys():
                            if key != 'old_log_probs':
                                seed_batch.batch[key] = seed_batch.batch[key].long()

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                seed_ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(seed_batch)
                                seed_batch = seed_batch.union(seed_ref_log_prob)

                        

                        # compute values
                        if self.use_critic:
                            with _timer('values', timing_raw):
                                seed_values = self.critic_wg.compute_values(seed_batch)
                                seed_batch = seed_batch.union(seed_values)


                        

                        q_l, a_l, i_l = self.extract_qa_from_batch(seed_batch)
                        
                        global_id = seed_batch.non_tensor_batch['id']
                        # print(f'##Generated global_ids##' * 3)
                        # print(global_id)
                        abded = 0
                        for idx, q,a,i in zip(global_id, q_l, a_l, i_l):
                            if q != '' and a!= '' and a.find('John Doe')==-1: # avoid repeating example answer
                                idx_ne_l.append(idx)
                                q_ne_l.append(q)
                                a_ne_l.append(a)
                                i_ne_l.append(i)
                                idx_ut_l.append(idx)
                            else:
                                idx_ab_l.append(idx)
                                abded += 1
                        proposed_pool.add(seed_batch)
                        
                        if self.is_debug:
                            print(f'###################### total_gen: {len(global_id)} ############################')
                            print(f'###################### Task_num: {len(q_ne_l)} ############################')
                            print(f'###################### Idxs_num: {len(idx_ne_l)} ############################')
                            print(f'###################### Nore_num: {len(set(idx_ne_l))} ############################')
                            print(f'###################### pool_num: {proposed_pool.get_len()} ############################')
                            print(f'###################### abandoned: {abded} ############################')
                        if len(idx_ne_l) != len(set(idx_ne_l)) or len(idx_ne_l) != len(q_ne_l):
                            raise ValueError("Duplicate IDs detected or length mismatch during question generation.")
                        
                    if q_len_before < len(q_ne_l):
                        q_len_before = len(q_ne_l)
                        stall_times = 0
                    else:
                        stall_times += 1
                    if stall_times >= max_retry_num:
                        raise ValueError(f"Too many stall times {stall_times} during question generation.")
                remain_size = buffer_size
                # 将三个列表组合成一个元组列表
                combined = list(zip(idx_ne_l, q_ne_l, a_ne_l, i_ne_l))
                # 对组合后的列表进行 shuffle 为了保证更新比例，不进行
                # 
                # 截断

                combined = combined[:buffer_size]
                # 将 shuffle 后的数据解压回三个列表
                random.shuffle(combined)
                idx_ne_l, q_ne_l, a_ne_l, i_ne_l = zip(*combined)
                idx_ne_l = list(idx_ne_l)
                q_ne_l = list(q_ne_l)
                a_ne_l = list(a_ne_l)
                i_ne_l = list(i_ne_l)



                
                idx_po_l = list(set(idx_ne_l) & set(idx_ut_l))
                random.shuffle(idx_ab_l)
                positive_sample_num = len(idx_po_l)
                negative_sample_num = negative_sample_num +1 if (negative_sample_num + positive_sample_num)%2 !=0 else negative_sample_num
                if len(idx_ab_l) >= negative_sample_num:                    
                    idx_ab_l = idx_ab_l[:negative_sample_num]  


                
                final_pool_idx_l = idx_po_l + idx_ab_l
                proposed_pool.update_pool(final_pool_idx_l)
                # assert len(final_pool_idx_l) == proposed_pool.get_len(), f'final_pool_idx_l size: {len(final_pool_idx_l)} and proposed_pool size: {proposed_pool.get_len()}'
                self.train_dataloader = self.train_generation(id_l = idx_ne_l, q_l = q_ne_l, a_l = a_ne_l, i_l = i_ne_l)

                if self.is_debug:
                    print(f'########## Final Task_num: {len(q_ne_l)} ###########')
                    print(f'########## question ###########')
                    print(q_ne_l[0])
                    print(f'########## answer ###########')
                    print(a_ne_l[0])
                    print(f'########## related_info ###########')
                    print(i_ne_l[0])
                

                

                solver_batch_list = []
                # solver 训练循环
                for batch_dict in self.train_dataloader:
                    # print(f'epoch {epoch}, step {self.global_steps}')
                    metrics = {}
                    timing_raw = {}

                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent * 4, interleave=True)

                    # pop those keys for generation
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    pos_batch = batch.pop(batch_keys=['p_input_ids', 'p_attention_mask', 'p_position_ids'])
                    p_meta_info = {'input_ids_key': 'p_input_ids', 
                                 'attention_mask_key': 'p_attention_mask',
                                 'position_ids_key': 'p_position_ids', 
                                 'responses_key': 'p_responses',
                                 'prompt_key': 'p_prompt'}
                    pos_batch.meta_info.update(p_meta_info)
                    nega_batch = batch.pop(batch_keys=['n_input_ids', 'n_attention_mask', 'n_position_ids'])
                    n_meta_info = {'input_ids_key': 'n_input_ids', 
                                 'attention_mask_key': 'n_attention_mask',
                                 'position_ids_key': 'n_position_ids', 
                                 'responses_key': 'n_responses',
                                 'prompt_key': 'n_prompt'}
                    nega_batch.meta_info.update(n_meta_info)
                    p_batch_output = self.external_model.generate_sequences_ext(pos_batch)
                    batch = batch.union(p_batch_output)
                    p_reward_tensor = self.external_reward_fn(data=batch, positive_mode=True, is_debug=False)
                    batch.batch['p_ext_scores'] = p_reward_tensor
                    n_batch_output = self.external_model.generate_sequences_ext(nega_batch)
                    batch = batch.union(n_batch_output)
                    n_reward_tensor = self.external_reward_fn(data=batch, positive_mode=False, is_debug=False)
                    batch.batch['n_ext_scores'] = n_reward_tensor
                    
                    
                    
                    
                    ####################
                    # original code here

                    with _timer('step', timing_raw):
                        if not self.config.do_search:
                            raise ValueError("Non-search PPO not supported in self-play setting.")
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                            batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                    dtype=object)
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)

                        ####################
                        # Below is aLL about agents - the "LLM + forloop"
                        ####################
                        # with _timer('step', timing_raw):
                        else:
                            first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                            with _timer('gen', timing_raw):
                                generation_manager.timing_raw = timing_raw
                                final_gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    initial_input_ids=first_input_ids,
                                )

                            # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                            for key in final_gen_batch_output.batch.keys():
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                            with torch.no_grad():
                                output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                final_gen_batch_output = final_gen_batch_output.union(output)

                            # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                            #                                         dtype=object)
                            batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                                
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(final_gen_batch_output)

                        ####################
                        ####################

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                        # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                        for key in batch.batch.keys():
                            if key != 'old_log_probs':
                                batch.batch[key] = batch.batch[key].long()

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with _timer('values', timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with _timer('adv', timing_raw):
                            # compute scores. Support both model and function-based.
                            # We first compute the scores using reward model. Then, we call reward_fn to combine
                            # the results from reward model and rule-based results.
                            if self.use_rm:
                                # we first compute reward model score
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # we combine with rule-based rm
                            reward_tensor = self.reward_fn(batch, self.is_debug)
                            batch.batch['token_level_scores'] = reward_tensor

                            

                            # compute rewards. apply_kl_penalty if available
                            if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                batch, kl_metrics = apply_kl_penalty(batch,
                                                                    kl_ctrl=self.kl_ctrl,
                                                                    kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                            





                            # compute advantages, executed on the driver process
                            batch = compute_advantage(batch,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    num_repeat=self.config.actor_rollout_ref.rollout.n)
                            solver_batch_list.append(batch)

                    gc.collect()
                    

                
                seed_batch: DataProto = proposed_pool.get_pool()
                
                # raise ValueError(f"Debug stop. {(proposed_pool.pool.batch['attention_mask']).shape, (seed_batch.batch['attention_mask']).shape}")
                # self._balance_batch(seed_batch, metrics=metrics)

                # compute global_valid tokens
                seed_batch.meta_info['global_token_num'] = torch.sum(seed_batch.batch['attention_mask'], dim=-1).tolist()

                seed_batch.batch['token_level_scores'] = self.proposer_reward_fn(seed_batch, solver_batch_list, is_debug=self.is_debug, alpha=0.1)
                # compute rewards. apply_kl_penalty if available
                if not self.config.actor_rollout_ref.actor.use_kl_loss:
                    seed_batch, kl_metrics = apply_kl_penalty(seed_batch,
                                                        kl_ctrl=self.kl_ctrl,
                                                        kl_penalty=self.config.algorithm.kl_penalty)
                    metrics.update(kl_metrics)
                else:
                    seed_batch.batch['token_level_rewards'] = seed_batch.batch['token_level_scores']

               

                # compute advantages, executed on the driver process
                seed_batch = compute_advantage(seed_batch,
                                        adv_estimator=self.config.algorithm.adv_estimator,
                                        gamma=self.config.algorithm.gamma,
                                        lam=self.config.algorithm.lam,
                                        num_repeat=self.config.actor_rollout_ref.rollout.n)


                
                for batch in solver_batch_list:
                    batch.pop(batch_keys=['p_attention_mask', 'p_position_ids', 'n_attention_mask', 'n_position_ids', 'n_input_ids', 'p_input_ids','p_responses', 'p_prompt', 'n_responses', 'p_ext_scores', 'n_prompt', 'n_ext_scores'])

                
                if self.global_steps >= 10:
                    seed_batch = DataProto.concat([seed_batch] + solver_batch_list, auto_pad=True, id_pad=self.pad_token_id[0])
                else:
                    pass # seed_batch = seed_batch

                gc.collect()
                # update critic
                if self.use_critic:
                    with _timer('update_critic', timing_raw):
                        critic_output = self.critic_wg.update_critic(seed_batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer('update_actor', timing_raw):
                        if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                            seed_batch, metrics = self._create_loss_mask(seed_batch, metrics)
                        actor_output = self.actor_rollout_wg.update_actor(seed_batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)
                

                if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                gc.collect()
                

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    # if self.val_reward_fn is not None:
                    #     val_metrics = self._validate()
                    #     pprint(f'Final validation metrics: {val_metrics}')
                    #     logger.log(data=val_metrics, step=self.global_steps)
                    return

    def external_reward_fn(self, data: DataProto, positive_mode=True, is_debug=False):

        if positive_mode:
            input_ids_key: str = 'p_input_ids', 
            attention_mask_key: str = 'p_attention_mask', 
            position_ids_key: str = 'p_position_ids', 
            responses_key: str = 'p_responses', 
            prompt_key: str = 'p_prompt'
        else:
            input_ids_key: str = 'n_input_ids', 
            attention_mask_key: str = 'n_attention_mask', 
            position_ids_key: str = 'n_position_ids', 
            responses_key: str = 'n_responses', 
            prompt_key: str = 'n_prompt'

        reward_tensor = torch.zeros(data.batch[responses_key].shape[0], 1, dtype=torch.float32)


        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem            

            prompt_ids = data_item.batch[prompt_key]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch[attention_mask_key][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch[responses_key]
            valid_response_length = data_item.batch[attention_mask_key][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']



            score = qa_em_simple.compute_score_bleu(solution_str=sequences_str, ground_truth=ground_truth, format_score=0.0, is_debug=(i<=10) and is_debug) # 

            reward_tensor[i, 0] = score


        return reward_tensor

    def proposer_reward_fn(self, seed_batch: DataProto, solver_batch_list: DataProto, is_debug=False, alpha=0.1, beta=0.1, format_score=0.1):
        # 使用 with torch.no_grad() 避免计算图保留
        with torch.no_grad():

            # global_solver_batch = DataProto.concat(solver_batch_list)
            group_score = {}
            p_ralative_score = {}
            n_ralative_score = {}
            for global_solver_batch in solver_batch_list:
                for i in range(len(global_solver_batch)):
                    data_item = global_solver_batch[i]  # DataProtoItem 
                    prompt_ids = data_item.batch['prompts']
                    uid = data_item.non_tensor_batch['index']
                    prompt_length = prompt_ids.shape[-1]
                    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

                    score = data_item.batch['token_level_scores']
                    score = score[valid_response_length - 1].item()
                    group_score[str(uid)] = group_score.get(str(uid), [])
                    group_score[str(uid)].append(score)

                    p_score = data_item.batch['p_ext_scores'][0].item()
                    p_ralative_score[str(uid)] = p_ralative_score.get(str(uid), [])
                    p_ralative_score[str(uid)].append(p_score)
                    n_score = data_item.batch['n_ext_scores'][0].item()
                    n_ralative_score[str(uid)] = n_ralative_score.get(str(uid), [])
                    n_ralative_score[str(uid)].append(n_score)

                    

            proposer_reward_tensor = torch.zeros_like(seed_batch.batch['responses'], dtype=torch.float32)
            
            
            sum_value = 0.0
            for i in range(len(seed_batch)):
                data_item = seed_batch[i]

                prompt_ids = data_item.batch['prompts']
                uid = data_item.non_tensor_batch['index']

                prompt_length = prompt_ids.shape[-1]

                # valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                # response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                # valid_response_ids = response_ids[:valid_response_length]

                score = (float(np.mean(group_score.get(str(uid), [0.0-format_score])) + format_score )) * (1/(1+format_score))
                # score = 1-score if (1>score>0) else 0.0 # 这里是越大越好
                score = (1-score) * score * alpha

                p_elenment = float(np.mean(p_ralative_score.get(str(uid), [0.0])))
                # p_elenment = math.exp(p_score_this)
                n_elenment = float(np.mean(n_ralative_score.get(str(uid), [0.0])))
                # n_elenment = math.exp(n_score_this)

                # rela_score = (math.log(1e-9+ p_elenment/(p_elenment + n_elenment + 1e-9))) * beta
                rela_score = F.logsigmoid(torch.tensor((p_elenment-n_elenment)*beta)).item()
                
                if is_debug:
                    sum_value += score
                proposer_reward_tensor[i, valid_response_length - 1] = score + rela_score

            if is_debug:
                print(f'##[Debug] Proposer Reward Calculation ##'*3)
                print(f'##[Debug] sum proposer reward: {sum_value}\n')
        
        return proposer_reward_tensor



    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics



    def balance_samples_for_divisibility(self, idx_po_l, idx_ab_l, divisor=4, strategy='smart_repeat'):
        """
        调整正负样本数量，使其总和能被divisor整除
        策略：优先通过重复样本来补齐，避免删除有效样本
        
        Args:
            idx_po_l: 正样本索引列表
            idx_ab_l: 负样本索引列表
            divisor: 目标除数（如GPU数量4）
            strategy: 补齐策略
                - 'repeat_positive': 优先重复正样本
                - 'repeat_negative': 优先重复负样本
                - 'random_repeat': 从所有样本中随机重复
                - 'smart_repeat': 智能选择（样本少的优先重复）
        
        Returns:
            调整后的正样本列表和负样本列表
        """
        positive_sample_num = len(idx_po_l)
        negative_sample_num = len(idx_ab_l)
        total = positive_sample_num + negative_sample_num
        
        # 如果已经可整除，直接返回
        if total % divisor == 0:
            return idx_po_l, idx_ab_l
        
        # 计算需要补充的数量
        deficit = divisor - (total % divisor)
        print(f"总数 {total} 不能被 {divisor} 整除，需要补充 {deficit} 个样本")
        
        # 策略1: 优先重复正样本（适用于正样本更重要或更少的场景）
        if strategy == 'repeat_positive':
            if positive_sample_num > 0:
                # 从正样本中随机重复
                repeat_samples = random.choices(idx_po_l, k=deficit)
                idx_po_l = idx_po_l + repeat_samples
            else:
                # 如果没有正样本，重复负样本
                repeat_samples = random.choices(idx_ab_l, k=deficit)
                idx_ab_l = idx_ab_l + repeat_samples
        
        # 策略2: 优先重复负样本
        elif strategy == 'repeat_negative':
            if negative_sample_num > 0:
                repeat_samples = random.choices(idx_ab_l, k=deficit)
                idx_ab_l = idx_ab_l + repeat_samples
            else:
                repeat_samples = random.choices(idx_po_l, k=deficit)
                idx_po_l = idx_po_l + repeat_samples
        
        # 策略3: 从所有样本中随机重复（最公平）
        elif strategy == 'random_repeat':
            all_samples = idx_po_l + idx_ab_l
            repeat_samples = random.choices(all_samples, k=deficit)
            # 将重复的样本按比例分配到正负列表
            for sample in repeat_samples:
                if sample in idx_po_l and random.random() < 0.5:
                    idx_po_l.append(sample)
                else:
                    idx_ab_l.append(sample)
        
        # 策略4: 智能重复（优先重复数量少的类别，保持类别平衡）
        elif strategy == 'smart_repeat':
            # 如果某一类样本明显少于另一类，优先重复它
            if positive_sample_num < negative_sample_num and positive_sample_num > 0:
                # 正样本少，优先重复正样本
                repeat_samples = random.choices(idx_po_l, k=deficit)
                idx_po_l = idx_po_l + repeat_samples
            elif negative_sample_num > 0:
                # 负样本少或相等，重复负样本
                repeat_samples = random.choices(idx_ab_l, k=deficit)
                idx_ab_l = idx_ab_l + repeat_samples
            else:
                #  fallback
                repeat_samples = random.choices(idx_po_l, k=deficit)
                idx_po_l = idx_po_l + repeat_samples
        
        # 验证结果
        new_total = len(idx_po_l) + len(idx_ab_l)
        assert new_total % divisor == 0, f"调整后仍不能被整除: {new_total} % {divisor} = {new_total % divisor}"
        
        return idx_po_l, idx_ab_l



class SeedData(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,                 
                 tokenizer: PreTrainedTokenizer,
                 data_num: int =1024,
                 seed_num: int = 3,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 prompt_prefix = '',
                 nltk_data_path = None,
                 statrt_index=0,
                 ):
        
        self.start_index = statrt_index

        self.data_num = data_num
        self.seed_num = seed_num
        self.tokenizer = tokenizer
        self.return_raw_chat = return_raw_chat

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts


        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.prompt_prefix = prompt_prefix
        self.nltk_data_path = nltk_data_path
        self.seed_tool= SeedTool(tokenizer=self.tokenizer, nltk_data_path=nltk_data_path)
        self._generate_data_and_tokenize()

    def generate_seed_pd(self):
            ''' 
            参考案例
            id train_0
            question total number of death row inmates in the us?
            golden_answers [2,718]
            data_source nq
            
            prompt [{'content': 'Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
            After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. 
            If the original query is complex or involves multiple parts, you are encouraged to decompose it into smaller sub-questions, separated by ##. For example: <search> sub-question 1 ## sub-question 2 </search>. 
            You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. 
            For example, <answer> Beijing </answer>. Question: total number of death row inmates in the us? ', 'role': 'user'}]

            ability fact-reasoning
            reward_model {'ground_truth': {'target': ['2,718']}, 'style': 'rule'}
            extra_info {'index': 0, 'split': 'train', 'type': 'na'}
            metadata  None    
            '''
            seed_len = self.data_num
            seed_list = self.seed_tool.get_random_english_words(batch_size=seed_len, K=self.seed_num)
            id_list = ['propose_'+ str(i) for i in range(self.start_index, self.start_index+seed_len)]
            question_list = ['' for _ in seed_list]
            seed_list = [seeds for seeds in seed_list]
            golden_answers_list = [[''] for _ in range(seed_len)]
            data_source_list = ['self_propose' for _ in range(seed_len)]
            prompt_list = [[{'content': self.prompt_prefix + seeds , 'role': 'user'}] for seeds in seed_list]
            ability_list = ['question proposing' for _ in range(seed_len)]
            reward_model_list = [{'ground_truth': {'target': ['']}, 'style': 'na'} for _ in range(seed_len)]
            extra_info_list = [{'index': i, 'split': 'train', 'type': 'na'} for i in range(self.start_index, self.start_index+seed_len)]
            metadata_list = [None for _ in range(seed_len)]
            seed_pd = pd.DataFrame({'id': id_list,
                                    'infos': ['' for _ in seed_list],
                                    'seeds': seed_list,
                                    'question': question_list,
                                    'golden_answers': golden_answers_list,
                                    'data_source': data_source_list,
                                    'prompt': prompt_list,
                                    # 'p_prompt': [[{'content': '', 'role': 'user'}] for _ in seed_list],
                                    # 'n_prompt': [[{'content': '', 'role': 'user'}] for _ in seed_list],
                                    'ability': ability_list,
                                    'reward_model': reward_model_list,
                                    'extra_info': extra_info_list,
                                    'metadata': metadata_list,
                                    })
            return seed_pd
    
            # id_list = ['train_'+ str(i) for i in self.id_l]
            # question_list = [str(question) for question in self.q_l]
            # info_list = [str(info) for info in self.i_l]
            # golden_answers_list = [[str(answer)] for answer in self.a_l]
            # data_source_list = ['self_propose' for _ in self.a_l]
            # prompt_list = [[{'content': self.train_prompt_prefix + str(question) , 'role': 'user'}] for question in self.q_l]
            # p_prompt_list = [[{'content': self.positive_prompt_prefix + str(question) + 'Relative information:' + str(info)  , 'role': 'user'}] for info, question in zip(self.i_l,self.q_l)]
            # n_prompt_list = [[{'content': self.negative_prompt_prefix + str(question) , 'role': 'user'}] for question in self.q_l]
            # ability_list = ['question proposing' for _ in self.a_l]
            # reward_model_list = [{'ground_truth': {'target': [str(answer)], 'p_question': [str(question)] }, 'style': 'na'} for answer,question in zip(self.a_l, self.q_l)]
            # extra_info_list = [{'index': i, 'split': 'train', 'type': 'na'} for i in self.id_l]
            # metadata_list = [None for _ in self.a_l]
            # train_pd = pd.DataFrame({'id': id_list,
            #                         'infos': info_list,
            #                         'seeds': [None for _ in self.a_l],
            #                         'question': question_list,
            #                         'golden_answers': golden_answers_list,
            #                         'data_source': data_source_list,
            #                         'prompt': prompt_list,
            #                         'p_prompt': p_prompt_list,
            #                         'n_prompt': n_prompt_list,
            #                         'ability': ability_list,
            #                         'reward_model': reward_model_list,
            #                         'extra_info': extra_info_list,
            #                         'metadata': metadata_list,
            #                         })

    def _generate_data_and_tokenize(self):
        self.dataframe = self.generate_seed_pd()
        # print(f'original dataset len: {len(self.dataframe)}')


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


class TrainData(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(self,                 
                 tokenizer: PreTrainedTokenizer,
                 data_num: int =1024,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 prompt_prefix = '',
                 positive_prompt_prefix='',
                 negative_prompt_prefix='',
                 id_l=[],
                 q_l=[],
                 a_l=[],
                 i_l=[]
                 ):

        self.data_num = data_num
        self.tokenizer = tokenizer
        self.return_raw_chat = return_raw_chat

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts


        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.train_prompt_prefix = prompt_prefix
        self.positive_prompt_prefix = positive_prompt_prefix
        self.negative_prompt_prefix = negative_prompt_prefix
        self.id_l = id_l
        self.q_l = q_l
        self.a_l = a_l
        self.i_l = i_l
        self.searcher = light_searcher()

        self._generate_data_and_tokenize()

    def generate_train_pd(self):
            ''' 
            参考案例
            id train_0
            question total number of death row inmates in the us?
            golden_answers [2,718]
            data_source nq
            
            prompt [{'content': 'Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
            After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. 
            If the original query is complex or involves multiple parts, you are encouraged to decompose it into smaller sub-questions, separated by ##. For example: <search> sub-question 1 ## sub-question 2 </search>. 
            You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. 
            For example, <answer> Beijing </answer>. Question: total number of death row inmates in the us? ', 'role': 'user'}]

            ability fact-reasoning
            reward_model {'ground_truth': {'target': ['2,718']}, 'style': 'rule'}
            extra_info {'index': 0, 'split': 'train', 'type': 'na'}
            metadata  None    
            '''

            id_list = ['train_'+ str(i) for i in self.id_l]
            question_list = [str(question) for question in self.q_l]
            info_list =  self.searcher.batch_search([str(info) for info in self.i_l])
            golden_answers_list =[[str(answer)] for answer in self.a_l]
            data_source_list = ['self_propose' for _ in self.a_l]
            prompt_list = [[{'content': self.train_prompt_prefix + str(question) , 'role': 'user'}] for question in self.q_l]
            p_prompt_list = [[{'content': self.positive_prompt_prefix + str(question) + 'Relative information:' + str(info)  , 'role': 'user'}] for info, question in zip(self.i_l,self.q_l)]
            n_prompt_list = [[{'content': self.negative_prompt_prefix + str(question) , 'role': 'user'}] for question in self.q_l]
            ability_list = ['question solving' for _ in self.a_l]
            reward_model_list = [{'ground_truth': {'target': [str(answer)], 'p_question': [str(question)] }, 'style': 'na'} for answer,question in zip(self.a_l, self.q_l)]
            extra_info_list = [{'index': i, 'split': 'train', 'type': 'na'} for i in self.id_l]
            metadata_list = [None for _ in self.a_l]
            train_pd = pd.DataFrame({'id': id_list,
                                    'infos': info_list,
                                    'seeds': ['' for _ in self.a_l],
                                    'question': question_list,
                                    'golden_answers': golden_answers_list,
                                    'data_source': data_source_list,
                                    'prompt': prompt_list,
                                    'p_prompt': p_prompt_list,
                                    'n_prompt': n_prompt_list,
                                    'ability': ability_list,
                                    'reward_model': reward_model_list,
                                    'extra_info': extra_info_list,
                                    'metadata': metadata_list,
                                    })
            return train_pd

    def _generate_data_and_tokenize(self):
        self.dataframe = self.generate_train_pd()
        # print(f'original dataset len: {len(self.dataframe)}')


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        p_chat = row_dict.pop('p_prompt')
        n_chat = row_dict.pop('n_prompt')

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            p_prompt_with_chat_template = self.tokenizer.apply_chat_template(p_chat, add_generation_prompt=True, tokenize=False)
            n_prompt_with_chat_template = self.tokenizer.apply_chat_template(n_chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = chat[0]['content']
            p_prompt_with_chat_template = p_chat[0]['content']
            n_prompt_with_chat_template = n_chat[0]['content']
        # prompt_with_chat_template = chat

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        p_input_ids, p_attention_mask = verl_F.tokenize_and_postprocess_data(prompt=p_prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        n_input_ids, n_attention_mask = verl_F.tokenize_and_postprocess_data(prompt=n_prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)
        p_position_ids = compute_position_id_with_mask(p_attention_mask)
        n_position_ids = compute_position_id_with_mask(n_attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['p_input_ids'] = p_input_ids[0]
        row_dict['p_attention_mask'] = p_attention_mask[0]
        row_dict['p_position_ids'] = p_position_ids[0]
        row_dict['n_input_ids'] = n_input_ids[0]
        row_dict['n_attention_mask'] = n_attention_mask[0]
        row_dict['n_position_ids'] = n_position_ids[0]
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

def print_data(data_proto_item, id=0, info='', max_length=20):
    print(f"=== {info} 预览 ===")
    first_item = data_proto_item[id]  # 这会返回一个DataProtoItem
    print(f"first_item类型: {type(first_item)}")

    # 辅助函数：处理长字符串的显示
    def format_value(value, max_len=max_length):
        if isinstance(value, torch.Tensor):
            return f"Tensor(shape: {value.shape})"
        else:
            str_value = str(value)
            if len(str_value) <= max_len:
                return str_value
            else:
                half_len = max_len // 2
                return f"{str_value[:half_len]}...{str_value[-half_len:]}"

    # 访问DataProtoItem的内容
    if first_item.batch is not None:
        print("Tensor数据:")
        for key, tensor in first_item.batch.items():
            print(f"  {key}: {format_value(tensor)}")

    if first_item.non_tensor_batch:
        print("Non-tensor数据:")
        for key, value in first_item.non_tensor_batch.items():
            print(f"  {key}: {format_value(value)} (type: {type(value).__name__})")

    # 处理meta_info
    if first_item.meta_info:
        print("Meta信息:")
        for key, value in first_item.meta_info.items():
            print(f"  {key}: {format_value(value)}")
    else:
        print("Meta信息: 无")


class Proposer_pool:
    def __init__(self,pad_token_id=None,divisor: int=2):
        self.pool=None
        self.id_l = []
        self.pad_token_id = pad_token_id[0]
        self.divisor = divisor

    def add(self, data_proto: DataProto):
        if self.pool is None:
            # print('!'*100 + 'in this')
            self.pool = data_proto
            self.id_l = data_proto.non_tensor_batch['id']

        else:
            
            self.pool = DataProto.concat([self.pool, data_proto], auto_pad=True, key_list=list(self.pool.batch.keys()), id_pad=self.pad_token_id)
            self.id_l=self.pool.non_tensor_batch['id']
        self.pad_to_div()
    def update_pool(self, ind_l):
        self.id_l = self.pool.non_tensor_batch['id']
        mask = torch.zeros(len(self.pool.batch), dtype=torch.bool)
        remain_num = 0
        remain_indices = []
        for i, id_val in enumerate(self.id_l):
            if id_val in ind_l:
                mask[i] = True
                remain_num += 1
                remain_indices.append(i)

        if remain_num>0:
            non_tensor_batch_keys = list(self.pool.non_tensor_batch.keys())
            # 应用掩码
            self.pool.batch = self.pool.batch[mask]
            for key in non_tensor_batch_keys:
                self.pool.non_tensor_batch[key] = copy.deepcopy(self.pool.non_tensor_batch[key][remain_indices])
            self.id_l = self.pool.non_tensor_batch['id']
        else:
            raise ValueError("Proposer pool is empty after removal.")
        
        self.pad_to_div()
        
    def pad_to_div(self):
        current_len = len(self.pool.non_tensor_batch['id'])
        deficit = (self.divisor - (current_len % self.divisor)) % self.divisor
        if deficit > 0:
            valid_indices = list(range(current_len))
            # 随机选择deficit个索引（可重复）
            repeat_indices = random.choices(valid_indices, k=deficit)
            
            # 安全地重复数据
            indices_to_add = torch.tensor(repeat_indices, dtype=torch.long)


            self.pool = DataProto.concat([self.pool, self.pool[indices_to_add]])
                

            self.id_l = self.pool.non_tensor_batch['id']

        assert len(self.pool.non_tensor_batch['id']) % self.divisor ==0, f"Padding to divisor failed, len {len(self.pool.non_tensor_batch['id'])}, diversor {self.divisor}."


    def get_pool(self):
        self.pad_to_div()
        return self.pool    
    def get_len(self):
        if self.pool is None:
            return 0
        return len(self.pool.non_tensor_batch['id'])
    def clear_keys_in_new_step(self,keys = ['token_level_scores', 'token_level_rewards', 'advantages', 'returns']):
        if self.pool is not None:
            for key in keys:
                if key in self.pool.batch:
                    self.pool.batch.pop(key)
    def clear_all(self):
        self.pool =None
        self.id_l = []


class light_searcher:
    def __init__(self, topk:int = 3, search_url:str = "http://127.0.0.1:8000/retrieve"):
        self.topk = topk
        self.search_url = search_url


    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        query_indices = []
        sub_queries = []
        for index, query in enumerate(queries):
            tmp_queries = query#.split("##")
            sub_queries.extend(tmp_queries)
            query_indices.extend([index] * len(tmp_queries))

        results = self._batch_search(sub_queries)['result']
        results = [self._passages2string(result) for result in results]

        final_results = [""] * len(queries)

        for index, result in enumerate(results):
            final_index = query_indices[index]
            final_results[final_index] += result

        return final_results
        

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True
        }
        
        return requests.post(self.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    

if __name__ == "__main__":
    my_searcher = light_searcher()

    myqueries = [
        'antonov##ai##NLP##Nobel',
        'car##EFV(Marine)##CVN##APFSDS',
        'kamenrider##superman##ultraman',
    ]
    print(len(my_searcher.batch_search(myqueries)))