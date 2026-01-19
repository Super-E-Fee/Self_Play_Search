import re
import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_step_timing_flexible(input_file, output_file):
    """
    灵活版本：使用正则表达式匹配
    """
    try:
        # 正则表达式解释：
        # ^step: - 以'step:'开头
        # .* - 任意字符
        # - timing_per_token_ms/ref: - 包含目标字符串
        # .{8} - 后面正好8个字符
        pattern = r'^step:.*- timing_per_token_ms/ref:.{8}$'
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            found_count = 0
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                
                if re.search(pattern, line):
                    outfile.write(line + '\n')
                    found_count += 1
                    print(f"第 {line_num} 行找到匹配: {line}")
            
            print(f"\n处理完成！共找到 {found_count} 个匹配项")
            print(f"结果已保存到: {output_file}")
            
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file}'")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# 更简洁的版本，使用正则表达式处理多行
def extract_step_timing_multiline_regex(input_file, output_file):
    """
    使用正则表达式处理多行匹配的版本
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            content = infile.read()
        
        # 正则表达式模式：
        # step:.*? - 匹配以step:开头的任意字符（非贪婪模式）
        # - timing_per_token_ms/ref: - 目标字符串
        # .{8} - 后面8个字符
        pattern = r'step:\s*(?:[0-9]|[1-9][0-9]|[1-9][0-9][0-9]|1000)\b.*?- timing_per_token_ms/ref:.{8}'
        
        matches = re.findall(pattern, content, re.DOTALL)
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for match in matches:
                # 清理多余的空白字符
                cleaned_match = ' '.join(match.split())
                outfile.write(cleaned_match + '#$##\n')
                # print(f"找到匹配: {cleaned_match}")
        
        print(f"\n处理完成！共找到 {len(matches)} 个匹配项")
        print(f"结果已保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file}'")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

def assert_metrics_length_consistency(verl_metrics):
    """
    断言VERL指标字典内各项值的list长度相同
    
    Args:
        verl_metrics: VERL训练指标字典
    """
    # 获取所有值的长度
    lengths = [len(value) for value in verl_metrics.values() if isinstance(value, list)]
    
    # 检查是否所有长度都相同
    if len(set(lengths)) > 1:
        # 找出长度不一致的项
        length_groups = {}
        for key, value in verl_metrics.items():
            if isinstance(value, list):
                length = len(value)
                if length not in length_groups:
                    length_groups[length] = []
                length_groups[length].append(key)
        
        # 构建错误信息
        error_msg = "VERL指标字典长度不一致:\n"
        for length, keys in length_groups.items():
            error_msg += f"长度 {length}: {len(keys)} 个指标\n"
            if len(keys) <= 5:  # 如果数量不多，显示具体指标名
                for key in keys:
                    error_msg += f"  - {key}\n"
            else:
                error_msg += f"  示例: {keys[:3]}...\n"
        
        raise AssertionError(error_msg)
    
    # 如果所有长度都相同，返回True
    return True
def read_metrics_from_log(input_log_file, csv_file):
    # VERL训练指标中文解释
    # 序列长度相关指标
    # global_seqlen/min: 最小全局序列长度
    # global_seqlen/max: 最大全局序列长度
    # global_seqlen/minmax_diff: 序列长度极差（最大值-最小值）
    # global_seqlen/balanced_min: 平衡后的最小序列长度
    # global_seqlen/balanced_max: 平衡后的最大序列长度
    # global_seqlen/mean: 平均序列长度

    # Critic（价值函数）指标
    # critic/kl: KL散度（衡量策略变化程度，负值表示新策略更确定）
    # critic/kl_coeff: KL惩罚系数
    # critic/vf_loss: 价值函数损失
    # critic/vf_clipfrac: 价值函数梯度裁剪比例（0表示没有裁剪）
    # critic/vpred_mean: 价值预测均值
    # critic/grad_norm: 梯度范数
    # mfu/critic: Critic模型利用率
    # critic/lr: Critic学习率

    # Actor（策略函数）指标
    # actor/entropy_loss: 熵损失（鼓励探索，值适中表示探索充分）
    # actor/pg_loss: 策略梯度损失
    # actor/pg_clipfrac: 策略梯度裁剪比例
    # actor/ppo_kl: PPO的KL约束（为0表示严格满足约束）
    # actor/grad_norm: Actor梯度范数
    # mfu/actor: Actor模型利用率
    # actor/lr: Actor学习率

    # 奖励和价值统计指标
    # critic/score/mean: 平均得分
    # critic/score/max: 最大得分
    # critic/score/min: 最小得分
    # critic/rewards/mean: 平均奖励
    # critic/rewards/max: 最大奖励
    # critic/rewards/min: 最小奖励
    # critic/advantages/mean: 优势函数均值（接近0是正常的）
    # critic/advantages/max: 最大优势值
    # critic/advantages/min: 最小优势值
    # critic/returns/mean: 回报均值
    # critic/returns/max: 最大回报
    # critic/returns/min: 最小回报
    # critic/values/mean: 价值函数预测均值
    # critic/values/max: 价值函数预测最大值
    # critic/values/min: 价值函数预测最小值
    # critic/vf_explained_var: 价值函数解释方差

    # 生成长度统计指标
    # response_length/mean: 平均回复长度
    # response_length/max: 最大回复长度
    # response_length/min: 最小回复长度
    # response_length/clip_ratio: 回复长度裁剪比例
    # prompt_length/mean: 平均提示长度
    # prompt_length/max: 最大提示长度
    # prompt_length/min: 最小提示长度
    # prompt_length/clip_ratio: 提示长度裁剪比例

    # 环境交互指标
    # env/number_of_actions/mean: 平均动作数量
    # env/number_of_actions/max: 最大动作数量
    # env/number_of_actions/min: 最小动作数量
    # env/finish_ratio: 任务完成率
    # env/number_of_valid_action: 有效动作数量
    # env/ratio_of_valid_action: 有效动作比例
    # env/number_of_valid_search: 有效搜索数量

    # 状态令牌指标
    # state_tokens/total: 总状态令牌数
    # state_tokens/coverage: 状态令牌覆盖率

    # 时间性能指标（绝对时间，单位：秒）
    # timing_s/gen: 生成时间
    # timing_s/ref: 参考模型时间
    # timing_s/values: 价值计算时间
    # timing_s/adv: 优势计算时间
    # timing_s/update_critic: Critic更新时间
    # timing_s/update_actor: Actor更新时间
    # timing_s/step: 总步长时间

    # 时间性能指标（每token时间，单位：毫秒）
    # timing_per_token_ms/update_actor: 每token Actor更新时间
    # timing_per_token_ms/values: 每token价值计算时间
    # timing_per_token_ms/adv: 每token优势计算时间
    # timing_per_token_ms/update_critic: 每token Critic更新时间
    # timing_per_token_ms/gen: 每token生成时间
    # timing_per_token_ms/ref: 每token参考模型时间
    # VERL训练指标字典（值为空列表）
    verl_metrics = {
        "step": [],
        # 序列长度相关指标
        "global_seqlen/min": [],
        "global_seqlen/max": [],
        "global_seqlen/minmax_diff": [],
        "global_seqlen/balanced_min": [],
        "global_seqlen/balanced_max": [],
        "global_seqlen/mean": [],
        
        # Critic（价值函数）指标
        "critic/kl": [],
        "critic/kl_coeff": [],
        "critic/vf_loss": [],
        "critic/vf_clipfrac": [],
        "critic/vpred_mean": [],
        "critic/grad_norm": [],
        "mfu/critic": [],
        "critic/lr": [],
        
        # Actor（策略函数）指标
        "actor/entropy_loss": [],
        "actor/pg_loss": [],
        "actor/pg_clipfrac": [],
        "actor/ppo_kl": [],
        "actor/grad_norm": [],
        "mfu/actor": [],
        "actor/lr": [],
        
        # 奖励和价值统计指标
        "critic/score/mean": [],
        "critic/score/max": [],
        "critic/score/min": [],
        "critic/rewards/mean": [],
        "critic/rewards/max": [],
        "critic/rewards/min": [],
        "critic/advantages/mean": [],
        "critic/advantages/max": [],
        "critic/advantages/min": [],
        "critic/returns/mean": [],
        "critic/returns/max": [],
        "critic/returns/min": [],
        "critic/values/mean": [],
        "critic/values/max": [],
        "critic/values/min": [],
        "critic/vf_explained_var": [],
        
        # 生成长度统计指标
        "response_length/mean": [],
        "response_length/max": [],
        "response_length/min": [],
        "response_length/clip_ratio": [],
        "prompt_length/mean": [],
        "prompt_length/max": [],
        "prompt_length/min": [],
        "prompt_length/clip_ratio": [],
        
        # 环境交互指标
        "env/number_of_actions/mean": [],
        "env/number_of_actions/max": [],
        "env/number_of_actions/min": [],
        "env/finish_ratio": [],
        "env/number_of_valid_action": [],
        "env/ratio_of_valid_action": [],
        "env/number_of_valid_search": [],
        
        # 状态令牌指标
        "state_tokens/total": [],
        "state_tokens/coverage": [],
        
        # 时间性能指标（绝对时间，单位：秒）
        "timing_s/gen": [],
        "timing_s/ref": [],
        "timing_s/values": [],
        "timing_s/adv": [],
        "timing_s/update_critic": [],
        "timing_s/update_actor": [],
        "timing_s/step": [],
        
        # 时间性能指标（每token时间，单位：毫秒）
        "timing_per_token_ms/update_actor": [],
        "timing_per_token_ms/values": [],
        "timing_per_token_ms/adv": [],
        "timing_per_token_ms/update_critic": [],
        "timing_per_token_ms/gen": [],
        "timing_per_token_ms/ref": []
    }
    with open(input_log_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
                line = line.strip()
                elements = line.split(' - ')
                for element in elements:
                    k_v = element.split(':')
                    k, v  = k_v[0].strip(), k_v[1].strip()
                    if ' ' in v:
                        v = v.split(' ')[0].strip()
                    if k == "step":
                        verl_metrics[k].append(int(v))
                    elif k in verl_metrics.keys():
                        verl_metrics[k].append(float(v))
    try:
        assert_metrics_length_consistency(verl_metrics)
        print("所有指标长度一致")
    except AssertionError as e:
        print(f"断言失败: {e}")
    # 1. 将字典转换为DataFrame
    df_metrics = pd.DataFrame(verl_metrics)

    # 2. 将DataFrame保存为CSV文件
    #    主要参数说明：
    #    'verl_metrics.csv' - 保存的文件名
    #    index=False - 不将行索引写入文件:cite[2]:cite[3]:cite[7]
    #    encoding='utf-8-sig' - 可选，使Excel打开中文不乱码:cite[6]
    df_metrics.to_csv(csv_file, index=False, encoding='utf-8-sig')

    print(f"数据已成功保存至 {csv_file }")

def plot_metric_from_csv(csv_file, y_metric, save_pdf=None, color='blue', dpi=300, step_col="step", figsize=(12, 6)):
    """
    从CSV文件读取数据并绘制指定指标的折线图，可保存为PDF
    
    Args:
        csv_file: CSV文件路径
        y_metric: 要绘制的纵坐标指标名称
        step_col: 横坐标列名（默认为"step"）
        figsize: 图表大小
        save_pdf: PDF保存路径，如果为None则不保存
        dpi: 图片分辨率
    """
    try:
        # 1. 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"✓ 成功读取文件: {csv_file}")
        print(f"数据形状: {df.shape}")
        
        # 2. 检查所需的列是否存在
        if step_col not in df.columns:
            raise KeyError(f"横坐标列 '{step_col}' 不存在于数据中")
        if y_metric not in df.columns:
            raise KeyError(f"纵坐标指标 '{y_metric}' 不存在于数据中")
        
        # 3. 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制折线图
        ax.plot(df[step_col], df[y_metric], 
                linewidth=2, marker='o', markersize=4, alpha=0.7,
                color=color, label=y_metric)
        
        # 设置图表样式
        ax.set_title(f'{y_metric} - {step_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(step_col, fontsize=12)
        ax.set_ylabel(y_metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 自动调整x轴刻度
        if len(df[step_col]) > 10:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 4. 保存为PDF
        if save_pdf:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_pdf) if os.path.dirname(save_pdf) else '.', exist_ok=True)
            plt.savefig(save_pdf, dpi=dpi, bbox_inches='tight')
            print(f"✓ 图表已保存至: {save_pdf}")
        else:
            plt.show()
        
        # 5. 打印统计信息
        print(f"\n统计信息 - {y_metric}:")
        print(f"  最小值: {df[y_metric].min():.6f}")
        print(f"  最大值: {df[y_metric].max():.6f}")
        print(f"  平均值: {df[y_metric].mean():.6f}")
        print(f"  最后值: {df[y_metric].iloc[-1]:.6f}")
        
        return fig
        
    except Exception as e:
        print(f"错误: {e}")
        return None
if __name__ == "__main__":
    input_log_file = "outputs/Prototype-ppo-Qwen2.5-3B-Instruct.log"
    output_result_file = "outputs/read_output.txt"
    csv_file = "outputs/verl_metrics.csv"
    
    extract_step_timing_multiline_regex(input_log_file, output_result_file)
    read_metrics_from_log(output_result_file, csv_file)

    paint_dict = {
        "env/ratio_of_valid_action": "green",
        "critic/score/mean": "blue",
        "critic/rewards/mean": "orange",
        "mfu/actor": "red"
    }
    for y_metric, color in paint_dict.items():
        save_pdf = f"outputs/plots/{y_metric.replace('/','_')}.pdf"
        plot_metric_from_csv(csv_file, y_metric, save_pdf=save_pdf, color=color)