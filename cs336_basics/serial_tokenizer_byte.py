from collections import Counter
import regex as re
from tqdm import tqdm
# from pretokenization_example import *

def get_segments(text):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    segments = []
    # pre-tokenize 防止词越界
    for t in re.finditer(PAT, text):
        pre_token = t.group()
        # print("group:",pre_token)
        u8_bytes = (pre_token.encode("utf-8"))
        # 修复：应该创建bytes对象列表，而不是生成器
        segments.append([bytes([b]) for b in u8_bytes])

    return segments

# 找到频率最高的对,如果频率一致,那么选择字典序最大的一对
# 每次合并过程只有一个合并,也就是一个新的token_id


def apply_merge(segments, merge_pair):
    results = []
    
    for tokens in segments:
        result = []  # 合并后的新token序列
        i = 0
        while i < len(tokens):
            # 检查当前位置是否是要合并的pair
            if (i < len(tokens) - 1) and tokens[i] == merge_pair[0] and tokens[i+1] == merge_pair[1]:
                result.append(merge_pair[0] + merge_pair[1])
                i += 2
            else:
                result.append(tokens[i])
                # print(type(tokens[i]))
                i += 1
        results.append(result)
    return results


def count_pairs(segments):
    byte_pair_count = Counter()
    for tokens in segments:
        for i in range(len(tokens) - 1):
            byte_pair_count[(tokens[i], tokens[i+1])] += 1
    return byte_pair_count

# 找到频率最高的字节对，如果频率相同则选择字典序最大的


def get_best_pair(counter):
    if not counter:
        return None

    # 找到最高频率
    max_freq = max(counter.values())

    # 找到所有具有最高频率的token id对
    candidates = [pair for pair, freq in counter.items() if freq == max_freq]

    # 按字典序排序，选择最da的
    best_pair = max(candidates)

    return best_pair, max_freq


def train_bpe(text, vocab_size, special_tokens: list = None):
    # 移除文本中的special token 内容
    special_pat = re.compile("|".join(re.escape(tok)
                             for tok in special_tokens))
    text = special_pat.sub("", text)

    if special_tokens:
        for token in special_tokens:
            text = text.replace(token, "")
    # 初始化bytes
    segments = get_segments(text)
    # 构建初始词汇表
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    # 映射特殊字符
    if special_tokens:
        for i, token in enumerate(special_tokens):
            vocab[256 + i] = token.encode("utf-8")

    num_merges = vocab_size - 256 - len(special_tokens)

    merges = []
    base_token_id = 256 + (len(special_tokens) if special_tokens else 0)

    bias_token_count = 0
    for i in tqdm(range(num_merges), desc="训练BPE", unit="merge"):
        # 计算字节对频率并返回最大的字节对
        try:
            pair, _ = get_best_pair(count_pairs(segments))
        except Exception:
            return vocab, merges

        # print("pair:",pair)
        new_token_id = base_token_id + bias_token_count
        # 保存合并规则
        merges.append(pair)
        # 映射新token到bytes
        new_bytes = pair[0] + pair[1]
        vocab[new_token_id] = new_bytes
        # 更新合并后的tokens
        segments = apply_merge(segments, pair)
        bias_token_count += 1

    # 修改返回值，应该返回vocab和merges，而不是token_ids
    return vocab, merges  # 添加token_to_bytes用于解码


def decode(token_ids, vocab):
    result = bytes([])
    for token_id in token_ids:
        result += vocab[token_id]
    return result.decode('utf-8')


def encode(text: str, merges, vocab, special_tokens: list = None):
    # 倒转vocab
    vocab_bytes_to_id = {bytes_val: token_id for token_id, bytes_val in vocab.items()}
    # 先处理特殊字符
    special_pat = re.compile("|".join(re.escape(tok)
                             for tok in special_tokens))
    
    special_map = {special_token : vocab_bytes_to_id[special_token.encode('utf-8')] for special_token in special_tokens}

    # 正则匹配 按照special token分隔,同时找到匹配的special token
    matches = special_pat.finditer(text)
    splits = special_pat.splititer(text)

    # 普通token进行分隔
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    token_ids = []
    for split_text in splits:
        segments = get_segments(split_text)
        
        # 把当前的segments按照顺序合并
        for merge_pair in merges:
            segments = apply_merge(segments, merge_pair)
        # 合并后的tokens添加到token_ids
        token_ids.extend([vocab_bytes_to_id[t] for seg in segments for t in seg]) # bytes->id

        # 当前普通文本处理完毕 检查是否是特殊token分隔开
        try:
            match = next(matches)
            token_ids.append(special_map[match.group()])
        except StopIteration:
            # 迭代器已经遍历完成，不需要处理
            pass

    return token_ids


def main():
    # text = "some text that i'll pre-tokenize"

    corpus_path = './tests/fixtures/tinystories_sample.txt'
    # corpus_path = './tests/fixtures/corpus.en'
    with open(corpus_path, 'r') as file:
        text = file.read()

    vocab_size = 500
    special_tokens = [
        "<|endoftext|>",
    ]
    vocab, merges = train_bpe(text, vocab_size, special_tokens)
    # print(vocab)
    real_merges = []
    # print(sorted(vocab.values()))
    # for pair,id in merges:
    #     print(vocab[pair[0]],' ',vocab[pair[1]])
    encoded = encode(text, merges,vocab, special_tokens)
    # print('encoded :',encoded)
    # print(f'max id:{max(encoded)}')
    # print(f'min id:{min(encoded)}')
    # print(f'avg token len:{sum(t for t in encoded) / len(encoded)}')
    # print(encoded)
    decoded_text = decode(encoded, vocab)
    assert decoded_text == text
    # print('decoded_text :',decoded_text)
    print(f'compression rate:{len(text) / len(encoded)}')


if __name__ == "__main__":

    main()

# print(vocab)
