import transformers
import random
import numpy as np
import nltk
from nltk.corpus import words

def get_random_english_words(tokenizer, K=10):
    """
    从tokenizer词汇表中获取随机英语单词
    """
    # 获取词汇表
    vocab = list(tokenizer.vocab.keys())
    
    # 过滤特殊token和子词
    special_tokens = tokenizer.special_tokens_map.values()
    valid_words = []
    
    for word in vocab:
        # 排除特殊token
        if word in special_tokens:
            continue
        # 排除子词片段（通常以##开头）
        if word.startswith('##'):
            continue
        # 排除非字母字符
        if not word.isalpha():
            continue
        # 排除过短或过长的单词
        if len(word) < 2 or len(word) > 20:
            continue
            
        valid_words.append(word)
    
    # 随机采样
    if len(valid_words) >= K:
        return random.sample(valid_words, K)
    else:
        return valid_words
    



class SeedTool:
    def __init__(self, tokenizer, nltk_data_path='/mnt/shared-storage-user/wangyifei3/nltk_files'):
        self.tokenizer = tokenizer
        
        # 设置nltk数据路径
        self.nltk_data_path = nltk_data_path
        self._setup_nltk()
        
        # 获取词汇表
        vocab_list = list(tokenizer.vocab.keys())
    
        # 获取特殊token
        special_tokens = set()
        for value in tokenizer.special_tokens_map.values():
            if isinstance(value, list):
                special_tokens.update(value)
            else:
                special_tokens.add(value)
        
        # 使用nltk英语词典
        try:
            english_words = set(words.words())
            # print(f"成功加载 {len(english_words)} 个英语单词")
        except LookupError as e:
            print(f"无法加载nltk words语料库: {e}")
            raise e
        
        # 使用列表推导式进行过滤（更高效）
        self.valid_words = [
            word for word in vocab_list
            if (word not in special_tokens and
                not word.startswith('##') and
                word in english_words and
                3 <= len(word) <= 12 and
                len(word) > 0 and word[0].isalpha())
        ]
        
        # print(f"过滤后得到 {len(self.valid_words)} 个有效英语单词")
        # if len(self.valid_words) > 0:
        #     print(f"示例: {random.sample(self.valid_words, min(5, len(self.valid_words)))}")

    def _setup_nltk(self):
        """设置nltk数据路径"""
        import os
        os.makedirs(self.nltk_data_path, exist_ok=True)
        
        if self.nltk_data_path not in nltk.data.path:
            nltk.data.path.append(self.nltk_data_path)
            # print(f"已添加nltk数据路径: {self.nltk_data_path}")
        
        words_path = os.path.join(self.nltk_data_path, 'corpora', 'words')
        if os.path.exists(words_path):
            # print(f"找到words语料库: {words_path}")
            pass
        else:
            print(f"警告: words语料库不存在于 {words_path}")

    # def _get_fallback_wordlist(self):
    #     """备用英语词表"""
    #     fallback_words = {
    #         'apple', 'book', 'computer', 'data', 'example', 'function',
    #         'knowledge', 'language', 'model', 'number', 'problem', 'question',
    #         'solution', 'system', 'theory', 'value', 'word', 'algorithm',
    #         'analysis', 'concept', 'definition', 'equation', 'formula', 'method',
    #         'principle', 'process', 'result', 'structure', 'technique', 'variable',
    #         'math', 'science', 'physics', 'chemistry', 'biology', 'history',
    #         'geography', 'literature', 'art', 'music', 'programming', 'logic'
    #     }
    #     return fallback_words

    def get_random_english_words(self, batch_size=1024, K=3):
        if len(self.valid_words) < K * batch_size:
            needed_size = K * batch_size
            current_size = len(self.valid_words)
            
            # 计算需要重复的次数
            repeat_times = (needed_size + current_size - 1) // current_size  # 向上取整
            
            # 重复并截取
            use_valid_words = (self.valid_words * repeat_times)[:needed_size]
        else:
            use_valid_words = self.valid_words

        all_seeds = random.sample(use_valid_words, K * batch_size)
        # 在创建seed_batches时直接转换
        seed_batches = [",".join(all_seeds[i:i+K]) for i in range(0, len(all_seeds), K)]
        return seed_batches
    


if __name__ == '__main__':
    # 使用示例
    model_id = "/mnt/shared-storage-user/wangyifei3/saved/Models/Qwen/Qwen2.5-3B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    seed_producer = SeedTool(tokenizer)
    random_words = seed_producer.get_random_english_words(batch_size=1024, K=10)
    print(f"随机生成的10个单词:\n {random_words}")