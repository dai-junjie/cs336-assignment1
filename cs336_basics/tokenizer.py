from functools import partial
from multiprocessing import Pool, cpu_count
import regex as re
import os
from typing import BinaryIO, Counter, Iterator, List
from tqdm import tqdm
from dataclasses import dataclass
from collections import OrderedDict
from queue import PriorityQueue
from collections import defaultdict
from cs336_basics.util import gpt2_bytes_to_unicode

class PreTokenizer:
    def __init__(self,special_tokens:list[str]):
        self.special_tokens = special_tokens
        # if special_tokens:
        #     self.special_tokens_pat = '|'.join(re.escape(token) for token in self.special_tokens)
        # else:
        #     self.special_tokens_pat = ''
        # 分词正则表达式
        self.word_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.split_token = '<|endoftext|>'
        
    def find_chunk_boundaries(
        self,
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    # @staticmethod
    def process_chunk(self,bound:tuple[int,int],corpus_path:str):
        
        with open(corpus_path,'rb') as f:
            counter = Counter()
            start, end = bound
            f.seek(start)
            chunk_text = f.read(end - start).decode('utf-8',errors='ignore')
            # 去掉split_token
            chunk_text = chunk_text.replace(self.split_token,'')
            # 去掉特殊token
            for token in self.special_tokens:
                chunk_text = chunk_text.replace(token,'')
            # 使用作业给出的分词正则表达式
            for match in re.finditer(self.word_pattern,chunk_text):
                token:str = match.group()
                if not token:
                    continue
                # 是合法的词就加入counter
                counter[token] += 1
            return counter

    def pretokenize(self,corpus_path:str)->Counter:
        counter = Counter()
        with open(corpus_path,'rb') as f:
            boundaries = self.find_chunk_boundaries(f,100,self.split_token.encode('utf-8'))
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i+1]
                f.seek(start)
                chunk_text = f.read(end - start).decode('utf-8',errors='ignore')
                # 去掉split_token
                chunk_text = chunk_text.replace(self.split_token,'')
                # 去掉特殊token
                for token in self.special_tokens:
                    chunk_text = chunk_text.replace(token,'')
                # 使用作业给出的分词正则表达式
                for match in re.finditer(self.word_pattern,chunk_text):
                    token:str = match.group()
                    if not token:
                        continue
                    # 是合法的词就加入counter
                    counter[token] += 1
                
        return counter
    
    def parallel_pretokenize(self,corpus_path:str):
        counter = Counter()
        bounds = []
        with open(corpus_path,'rb') as f:
            boundaries = self.find_chunk_boundaries(f,100,self.split_token.encode('utf-8'))
            for i in range(len(boundaries) - 1):
                bounds.append((boundaries[i],boundaries[i+1]))
        # 多线程解决
        min_proc = min(cpu_count(),len(bounds))
        print(f'process num: {min_proc}')
        with Pool(processes=min_proc) as pool :
            process_chunk_with_args = partial(
                self.process_chunk,corpus_path=corpus_path
            )
            results = list(tqdm(
                pool.imap(process_chunk_with_args,bounds),
                total=len(bounds),
                desc='pretokenizing'
                ))
    
        for cnt in results:
            counter.update(cnt)
        return counter

class TokenNode:
    def __init__(self, token_id,count, prev=None, next=None) -> None:
        self.token_id = token_id
        self.count = count
        self.prev: TokenNode = prev
        self.next: TokenNode = next
        # 针对b,b,,b,b这种序列 合并为bb,b,b后，第2个b标记失效，因此会到第3个b 合并为[bb,bb]
        self.valid = True

class TokenSequence:
    def __init__(self,count) -> None:
        self.head: TokenNode = None
        self.tail: TokenNode = None
        self.count = count
        
@dataclass
class PairInfo:
    def __init__(self):
        self.count = 0
        # 使用 OrderedDict 替代 list，保持插入顺序且支持高效删除
        self.node_positions = OrderedDict()

    def __post_init__(self):
        if self.node_positions is None:
            self.node_positions = OrderedDict()

@dataclass
class MergeEntry:
    def __init__(self, freq, merge, valid: bool):
        self.freq = freq
        self.merge = merge
        self.valid = valid

    def __lt__(self, obj):
        if self.freq == obj.freq:
            return self.merge > obj.merge
        else:
            return self.freq > obj.freq

class BpeTokenizer:
    def __init__(self,vocab_size:int,special_tokens:list[str]):
        """
        初始化BPETrainer
            :param vocab_size: 词汇表大小
            :param special_tokens: 特殊token列表
        先利用gpt2的映射0-255,然后合并更新vocab 最后加入special tokens
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pretokenizer = PreTokenizer(special_tokens)
        self.counter_dict = defaultdict(PairInfo)
        self.entry_map = {} # 用来追踪合并对在优先队列中的位置，以便修改valid属性
        self.vocab = {}
        self.merges = []
        for i in range(256):
            self.vocab[i] = bytes([i])
        
        for i,token in enumerate(special_tokens):
            self.vocab[256 + i] = token.encode('utf-8')
    
    
    def serial_train(self,corpus_path:str):
        counter = self.pretokenizer.pretokenize(corpus_path)
        # print(counter)
        step = 10
        for item in counter.most_common():
            print(item)
            if step == 0:
                break
            step -= 1
        print(f'pretokenize token num: {len(counter)}')
    
    def parallel_train(self,corpus_path:str):
        counter = self.pretokenizer.parallel_pretokenize(corpus_path)
        # print(counter)
        step = 10
        for item in counter.most_common():
            print(item)
            if step == 0:
                break
            step -= 1
        print(f'parallel pretokenize token num: {len(counter)}')
        
    def train(self,corpus_path:str):
        """ 
        这些预分词的token是经过计数合并的，避免重复计算
        使用双链表来构建token序列
        _ t h e : 97次
        """
        word_freq = self.pretokenizer.parallel_pretokenize(corpus_path)
        """
        初始化合并对
        """
        for word in word_freq.keys():
            u8_bytes = word.encode('utf-8')
            linked_list = TokenSequence(word_freq[word])
            for i,b in enumerate(u8_bytes):
                if linked_list.head == None:
                    node = TokenNode(b,word_freq[word])
                    linked_list.head = node
                    linked_list.tail = node
                else:
                    node = TokenNode(b,word_freq[word],linked_list.tail,None)
                    linked_list.tail.next = node
                    linked_list.tail = node
                # 可以形成一个合并对
                if i < len(u8_bytes) - 1:
                    pair_key = (bytes([u8_bytes[i]]),
                                bytes([u8_bytes[i+1]]))
                    if pair_key in self.counter_dict:
                        self.counter_dict[pair_key].count += word_freq[word]
                        self.counter_dict[pair_key].node_positions[node] = None
                    else:
                        pair_info = PairInfo()
                        pair_info.count += word_freq[word]
                        self.counter_dict[pair_key] = pair_info
                        pair_info.node_positions[node] = None
        """
        初始化优先队列，以便后续高效合并
        """            
        pq = PriorityQueue()
        for pair_key,pair_info in self.counter_dict.items():
            entry = MergeEntry(pair_info.count,pair_key,True)
            self.entry_map[pair_key] = entry
            pq.put(entry)
        def get_best_pair():
            while not pq.empty():
                entry:MergeEntry = pq.get()
                if entry.valid:
                    return entry.merge
        """
        合并过程
        """
        num_merges = self.vocab_size - len(self.special_tokens) - 256
        token_offset = 256 + len(self.special_tokens)
        for i in tqdm(range(num_merges),desc='merge'):
            # ======================= 每次只更新受影响的pair 有新增有删除 ==============================
            affected_pairs = set()
            new_token_id = token_offset + i
            # 获取本轮应该合并的最高频的pair
            pair = get_best_pair()
            self.merges.append((pair[0], pair[1]))
            self.vocab[new_token_id] = pair[0] + pair[1]
            # 合并
            pair_info: PairInfo = self.counter_dict[pair]
            merge = pair[0] + pair[1]
            # (b1,b2,b3,b4)
            # 删除 (b1,b2)
            # 新增 (b1,b2b3)
            # 删除 (b2,b3)  这个不用搞 直接从字典中删除这个key
            # 新增 (b2b3,b4)
            for node in pair_info.node_positions:
                # 先判断node是否有效
                if not node.valid:
                    continue
                node:TokenNode = node
                
                if node.prev:
                    # 删除 (b1,b2)
                    # print(self.vocab[node.prev.token_id],node.prev)
                    old_pair = (
                        self.vocab[node.prev.token_id], self.vocab[node.token_id])
                    if old_pair == pair:
                        print(f'(b1,b2) error! {pair}')
                        return
                    self.counter_dict[old_pair].count -= node.count
                        
                    # 从node_positions中删除node.prev
                    if node.prev in self.counter_dict[old_pair].node_positions:
                        del self.counter_dict[old_pair].node_positions[node.prev]
                    # 加入affected 中
                    affected_pairs.add(old_pair)

                    # 新增 (b1,b2b3)
                    new_pair = (self.vocab[node.prev.token_id], merge)
                    if new_pair in self.counter_dict:
                        self.counter_dict[new_pair].count += node.count
                        self.counter_dict[new_pair].node_positions[node.prev] = None
                    else:
                        pair_info = PairInfo()
                        pair_info.count = node.count
                        pair_info.node_positions[node.prev] = None
                        self.counter_dict[new_pair] = pair_info
                    # 加入affected 中
                    affected_pairs.add(new_pair)
                    if new_pair == pair:
                        print(f'(b1,b2b3) error! {pair}')
                        return

                # 合并 b2,b3 如果有next->next 就要合并 如果没有 不合并也没事 因为next无效了

                # 毋庸置疑 这个next没用了
                node.next.valid = False
                if node.next.next:
                    # 删除 (b3,b4)
                    old_pair = (self.vocab[node.next.token_id], self.vocab[node.next.next.token_id])
                    if old_pair == pair:
                        # print(f'(b3,b4) error! {pair}') # 这个重叠没事 后面会让当前node无效
                        pass
                    else:
                        # return
                        self.counter_dict[old_pair].count -= node.count
                        
                        # 从node_positions中删除node.prev
                        if node.next in self.counter_dict[old_pair].node_positions:
                            del self.counter_dict[old_pair].node_positions[node.next]
                        # 加入affected 中
                        affected_pairs.add(old_pair)

                    # 新增 (b2b3,b4)
                    new_pair = (merge, self.vocab[node.next.next.token_id])
                    if new_pair == pair:
                        print(f'(b2b3,b4) error! {pair}')
                        return
                    self.counter_dict[new_pair].count += node.count
                    self.counter_dict[new_pair].node_positions[node] = None
                    # 加入affected 中
                    affected_pairs.add(new_pair)

                    # (b2b3<-b4)   (b2b3->b4) 这个地方要重新构建链表节点关系
                    neighbor = node.next.next
                    node.next = neighbor
                    neighbor.prev = node
                else:
                    # 如果没有next->next 说明当前节点是最后一个节点
                    node.next.prev = None
                    node.next = None
                # 合并得到新的token_id
                node.token_id = new_token_id

            # merge完成后 删除pair(b2,b3) 已经从大根堆出去了 就不加入affected 中
            self.counter_dict.pop(pair)
            self.entry_map.pop(pair)

            if pair in affected_pairs:
                print('error!')
                return

            # ================== 从pq中添加/逻辑删除 =======================
            for pair in affected_pairs:
                # 修改操作
                if pair in self.entry_map:
                    entry: MergeEntry = self.entry_map[pair]
                    entry.valid = False
                    # 把最新的状态加入到优先队列中
                    if self.counter_dict[pair].count != 0:
                        new_entry = MergeEntry(
                            self.counter_dict[pair].count, entry.merge, True)
                        pq.put(new_entry)
                        self.entry_map[pair] = new_entry
                    else:
                        self.counter_dict.pop(pair)
                        self.entry_map.pop(pair)
                else:
                    # 新增操作
                    new_entry = MergeEntry(
                        self.counter_dict[pair].count, pair, True)
                    pq.put(new_entry)
                    self.entry_map[pair] = new_entry

            
            
            pass
        """
        返回结果
        """
        return self.vocab,self.merges
    
    def save(self,base_dir:str):
        import json
        import os
        gpt2_encoder = gpt2_bytes_to_unicode()
        token_to_id = {} # str - > int
        for k,v in self.vocab.items():
            if k < 256 : 
                token_to_id[gpt2_encoder[k]] = k
            else:
                token_to_id[ ''.join( gpt2_encoder[b] for b in v)] = k
        # print(token_to_id)
        with open(os.path.join(base_dir,'vocab.json'),'w') as f:
            json.dump(token_to_id,f,ensure_ascii=False,indent=4)
            
        # 保存merges
        merge_list = []
        for merge in self.merges:
            bs0,bs1 = merge
            bs0 = ''.join( gpt2_encoder[b] for b in bs0) # 引用切换 不修改merge了
            bs1 = ''.join( gpt2_encoder[b] for b in bs1)
            merge_list.append((bs0,bs1))

        with open(os.path.join(base_dir,'merges.txt'),'w') as f:
            for pair in merge_list:
                f.write(' '.join(pair) + '\n')
            
            
        # 找到频率最高的对,如果频率一致,那么选择字典序最大的一对
    

if __name__ == '__main__':
    # 主程序入口
    corpus_path = 'tests/fixtures/tinystories_sample_5M.txt'
    # corpus_path = "data/TinyStoriesV2-GPT4-train.txt"
    tokenizer = BpeTokenizer(10_000,['<|endoftext|>'])
    # tokenizer.parallel_train(corpus_path)
    # print('serial train')
    # tokenizer.serial_train(corpus_path)
    vocab,merges = tokenizer.train(corpus_path)
    # print(f'vocab : {(vocab)}')
    # print('-'*100)
    # print(f'merges : {(merges)}')
    
    tokenizer.save('fast_bpe')