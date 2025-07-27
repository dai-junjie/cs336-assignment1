from multiprocessing import Pool
from sympy.logic.inference import valid
from pretokenizer import PreTokenizer
import regex as re
from typing import List
from collections import defaultdict
from dataclasses import dataclass
from queue import PriorityQueue
from tqdm import tqdm
from collections import OrderedDict
from util import gpt2_bytes_to_unicode
import time
from functools import partial
from os import cpu_count

class TokenNode:
    def __init__(self,token_id,prev = None,next = None) -> None:
        self.token_id = token_id
        self.prev:TokenNode = prev
        self.next:TokenNode = next
        # 针对b,b,,b,b这种序列 合并为bb,b,b后，第2个b标记失效，因此会到第3个b 合并为[bb,bb]
        self.valid = True

class TokenSequence:
    def __init__(self) -> None:
        self.head:TokenNode = None
        self.tail:TokenNode = None

@dataclass
class PairInfo:
    def __init__(self):
        self.count = 0
        # 使用 OrderedDict 替代 list，保持插入顺序且支持高效删除
        self.node_positions = OrderedDict()
    
    def __post_init__(self):
        if self.node_positions is None:
            self.node_positions = []

def apply_merges(segments,merges,vocab):
    # vocab是 id -> bytes 要改成bytes - > id
    bytes_to_id = {v:k for k,v in vocab.items()}
    token_ids = []
    
    for id,seg in enumerate(segments):
        linked_list = TokenSequence()
        # 为当前链表构建merge map来进行合并操作 这个必须在循环中 否则会导致跨segment合并 巨大的bug!!!
        merge_map = defaultdict(OrderedDict)
        merge_set = set(merges)
        # for merge in merges:
        #     merge_map[merge] = OrderedDict()
            
        # 构建链表 和pair-> list[node] 索引
        for i,token_id in enumerate(seg):
            if linked_list.head == None:
                node = TokenNode(bytes([token_id]))
                linked_list.head = node
                linked_list.tail = linked_list.head
            else:
                # (tail<-node) (tail->node) 更新tail
                node = TokenNode( bytes([token_id]) ,linked_list.tail)
                linked_list.tail.next = node
                linked_list.tail = node
            if i<len(seg) - 1:
                pair = (bytes([seg[i]]) , bytes([seg[i+1]]))
                if pair in merge_set:
                    # print(f'merge in merge_set')
                    if pair not in merge_map:
                        # print(f'merge no in merge_map')
                        merge_map[pair] = OrderedDict()
                    merge_map[pair][node] = None
                    
        # 按照merge顺序合并
        for merge in merges:
            # 假设b1,b2,b3,b4 这里node是b2位置
            if merge in merge_map:
                for node in merge_map[merge]: 
                    if not node.valid: continue
                    
                    new_bytes = merge[0] + merge[1]
                    # 删除 (b1,b2)
                    node:TokenNode = node
                    if node.prev :
                        old_pair = (node.prev.token_id,node.token_id)
                        if old_pair == merge: # checkpoint
                            print('error (b1,b2,b3) = (b,b,b)')
                            print(f' prev:{node.prev.token_id} , cur:{node.token_id},next:{node.next.token_id}')
                        
                        if old_pair in merge_map:
                            # print('合并导致了后面会出现的pair被删除了')
                            merge_map[old_pair].pop(node.prev)
                        # 新增 (b1,b2b3)
                        new_pair = (node.prev.token_id, new_bytes)
                        
                        if new_pair == merge: # checkpoint
                            print('error (b1,b2b3) ')

                            
                        if new_pair  in merge_set:
                            merge_map[new_pair][node.prev] = None
                        
                    # 删除 (b2,b3) 防止 b,b,b,b情况  我合并bb后有bb,b,b 然后bb中的第二个b想和后面的b合并
                    
                    node.next.valid = False
                    
                    # 新增 (b2b3,b4)
                    if node.next.next and node.next.next.valid:
                        neighbor = node.next.next
                        # 丢掉b3 那么就得重新建立联系
                        node.next = neighbor
                        neighbor.prev = node
                        
                        new_pair = (new_bytes,neighbor.token_id)
                        
                        if new_pair == merge: # checkpoint
                            print('error (b1b2,b3) ')
                            
                        if  new_pair in merge_set:
                            merge_map[new_pair][node] = None
                     
                    
                    node.token_id = new_bytes
                    
        # 合并后 收集每个seg的token_ids

        p = linked_list.head
        while p:
            if p.valid:
                token_bytes = p.token_id
                try:
                    # print(f'token id:{token_id}')
                    token_id = bytes_to_id[token_bytes]
                except:
                    print(f'erorr :{type(token_bytes)}')
                token_ids.append(token_id)
                
            p = p.next
                
    return token_ids            

class BPETrainer:
    def __init__(self,vocab_size:int, special_tokens:list[str]):
        """
        初始化BPETrainer
            :param vocab_size: 词汇表大小
            :param special_tokens: 特殊token列表
        先利用gpt2的映射0-255,然后合并更新vocab 最后加入special tokens
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pretokenizer = PreTokenizer(special_tokens)
        self.vocab = {}
        self.merges = []
        self.counter_dict = defaultdict(PairInfo)
        self.word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_token_pat = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        self.entry_map = {}
        self.pq = PriorityQueue()
        
        for i in range(256): self.vocab[i] = bytes([i])
        
    def add_special_tokens(self):
        if len(self.vocab) + len(self.special_tokens) > self.vocab_size:
            raise ValueError("Special tokens exceed vocab size")
        
        # 默认无视vocab size
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token.encode('utf-8')

    def get_best_pair(self):
        while not self.pq.empty():
            max_count, reversed_bytes, pair, valid = self.pq.get()
            # print(f'valid:{valid}  ,freq:{-max_count},pair:{pair}, pq size:{self.pq.qsize()}')
            if valid:
                # print(max_count)
                return pair
        
        raise ValueError("Priority queue is empty")

    def init_corpus_and_counter(self,corpus_path):
        for text in self.pretokenizer.corpus_iter(corpus_path):
            for match in re.finditer(self.word_pattern,text):
                seq:str = match.group()
                if not seq: continue
                # 这个bytes序列可以构建一个双向链表
                u8_bytes:bytes = seq.encode('utf-8')
                
                linked_list = TokenSequence()
                for i,b in enumerate(u8_bytes):
                    if linked_list.head == None:
                        node = TokenNode(b)
                        linked_list.head = node
                        linked_list.tail = node
                    else:
                        node = TokenNode(b,linked_list.tail,None)
                        linked_list.tail.next = node
                        linked_list.tail = node
                        
                    if i < len(u8_bytes) - 1:
                        # 可以形成一对 counter除了要把[bytes,bytes] 的key记住 还要记住node指针
                        pair_key = (bytes([u8_bytes[i]]),bytes([u8_bytes[i+1]]))
                        if pair_key in self.counter_dict:
                            self.counter_dict[pair_key].count += 1
                            self.counter_dict[pair_key].node_positions[node] = None
                        else:
                            pair_info = PairInfo()
                            pair_info.count += 1
                            pair_info.node_positions[node] = None
                            self.counter_dict[pair_key] = pair_info
        
    
    def init_priority_queue(self):
        for pair, pair_info in self.counter_dict.items():
            pair_info:PairInfo = pair_info
            # 创建反向排序的bytes
            reversed_bytes = bytes(255 - b for b in (pair[0] + pair[1]))
            entry = [-pair_info.count, reversed_bytes, pair, True]
            self.entry_map[pair] = entry
            self.pq.put(entry)
    
    def merge(self):
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        token_offset = 256
        for i in tqdm(range(num_merges)):
            # ======================= 每次只更新受影响的pair 有新增有删除 ==============================
            affected_pairs = set()
            
            new_token_id = token_offset + i
            pair = self.get_best_pair()

            self.merges.append((pair[0],pair[1]))
            self.vocab[new_token_id] = pair[0] + pair[1]
            # 合并 
            pair_info:PairInfo = self.counter_dict[pair]
            merge = pair[0] + pair[1]
            # (b1,b2,b3,b4)
            # 删除 (b1,b2) 
            # 新增 (b1,b2b3)
            # 删除 (b2,b3)  这个不用搞 直接从字典中删除这个key
            # 新增 (b2b3,b4) 
            for node in pair_info.node_positions:
                if not node.valid: continue
                if node.prev:
                    # 删除 (b1,b2)
                    # print(self.vocab[node.prev.token_id],node.prev)
                    old_pair = (self.vocab[node.prev.token_id] , self.vocab[node.token_id])
                    if old_pair == pair:
                        print(f'(b1,b2) error! {pair}')
                        return 
                    self.counter_dict[old_pair].count -= 1
                    # 从node_positions中删除node.prev
                    if node.prev in self.counter_dict[old_pair].node_positions:
                        del self.counter_dict[old_pair].node_positions[node.prev]
                    # 加入affected 中
                    affected_pairs.add(old_pair)
                    
                    # 新增 (b1,b2b3)
                    new_pair = (self.vocab[node.prev.token_id], merge)
                    if new_pair in self.counter_dict:
                        self.counter_dict[new_pair].count += 1
                        self.counter_dict[new_pair].node_positions[node.prev] = None
                    else:
                        pair_info = PairInfo()
                        pair_info.count = 1
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
                    # 新增 (b2b3,b4)
                    new_pair = (merge , self.vocab[node.next.next.token_id])
                    if new_pair == pair:
                        print(f'(b2b3,b4) error! {pair}')
                        return 
                    self.counter_dict[new_pair].count += 1
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
                    entry = self.entry_map[pair]
                    entry[-1] = False
                    # 把最新的状态加入到优先队列中
                    if self.counter_dict[pair].count != 0:
                        new_entry = [-self.counter_dict[pair].count, entry[1], entry[2], True]
                        self.pq.put(new_entry)
                        self.entry_map[pair] = new_entry
                    else:
                        self.counter_dict.pop(pair)
                        self.entry_map.pop(pair)
                else:
                    # 新增操作
                    new_entry = [-self.counter_dict[pair].count,
                                 bytes([255 - b for b in (pair[0] + pair[1])]),
                                 pair, True]
                    self.pq.put(new_entry)
                    self.entry_map[pair] = new_entry
        

    def train(self,corpus_path:str):
        """
        训练BPE模型
            :param corpus_path: 语料库路径
        """
        start_time = time.time()
        # 初始化counter
        self.init_corpus_and_counter(corpus_path)
        print(f'链表构建和counter初始化耗时:{time.time() - start_time}')
        start_time = time.time()
        
        # 利用counter来初始化最大堆优先队列
        self.init_priority_queue()
        print(f'优先队列初始化耗时:{time.time() - start_time}')

        # while not self.pq.empty():
        #     max_count, reversed_bytes, pair, valid = self.pq.get()
        #     print(f'valid:{valid}  ,freq:{-max_count},pair:{pair}')
        # return 
        
        self.merge()
        
        self.add_special_tokens()
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
    
    
        
    def encode(self,text:str):
        """
        编码文本
            :param text: 输入文本
            :return: 编码后的token列表
        """
        matches = self.special_token_pat.finditer(text)
        splits = self.special_token_pat.splititer(text)
        
        
        special_map = {tok: self.vocab_size - len(self.special_tokens) + i 
                       for i,tok in enumerate(self.special_tokens)}
        
        
        # 把merges
        token_ids = []
        segments_chunks = []
        split_chunks = []
        for split_text in splits:
            # print(f'encode:{split_text}')
            segments = []
            for t in re.finditer(self.word_pattern,split_text):
                pretoken = t.group()
                segments.append(pretoken.encode('utf-8'))
            # 按照merges合并
            
            # ids = apply_merges(segments,self.merges,self.vocab)
            # token_ids.extend(ids)
            segments_chunks.append(segments)
            # 通过special token分隔的 这里单独加入一个special token
            try:
                match = next(matches)
                split_chunks.append(special_map[match.group()])
            except:
                pass
        min_proc = min(cpu_count(),len(segments_chunks))
        print(f'process num:{min_proc}')
        with Pool(processes=min_proc) as pool:
            # 使用partial来固定merges和vocab参数
            apply_merges_with_params = partial(apply_merges, merges=self.merges, vocab=self.vocab)
            chunks = pool.map(apply_merges_with_params, segments_chunks)
            
            for i,chunk in enumerate(chunks):
                token_ids.extend(chunk)
                if i < len(split_chunks):
                    token_ids.append(split_chunks[i])
        
        return token_ids
        
    def decode(self,tokens_ids:List[int]):
        """
        解码token列表
            :param tokens: 输入token列表
            :return: 解码后的文本
        """
        result = bytes([])
        for token_id in tokens_ids:
            result += self.vocab[token_id]
        return result.decode('utf-8')

if __name__ == "__main__":
    corpus_path = './tests/fixtures/tinystories_sample_5M.txt'
    # corpus_path = './tests/fixtures/corpus.en'
    # corpus_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    
    trainer = BPETrainer(vocab_size,special_tokens)
    vocab,_ =trainer.train(corpus_path)
    
    trainer.save('cs336_basics')
    
    exit(0)
    
    print(len( set(vocab.values()) ))
 
    with open(corpus_path,'r') as f:
        text = f.read()
    
    start = time.time()
    token_ids = trainer.encode(text)
    end = time.time()
    print(f'encode time:{end - start}')
    
    # compression rate
    print(f'compression rate:{len(text) / len(token_ids)}')
    
    start = time.time()
    decoded_text = trainer.decode(token_ids)
    end = time.time()
    
    
    print(f'decode time:{end - start}')
    # print(decoded_text)
    print(f'encoded text stat: max id:{max(token_ids)},min id:{min(token_ids)}')
    
    
    with open('cs336_basics/decoded.txt','w') as f:
        f.write(decoded_text)
    # print(f'检查解码的文本是否与原文本相等: {decoded_text == text}')
    assert decoded_text == text
