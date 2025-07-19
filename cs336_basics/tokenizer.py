import regex as re
from functools import lru_cache
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
from pretokenizer import PreTokenizer
from os import cpu_count
from queue import PriorityQueue
from collections import OrderedDict

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"),
                                                          ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

class TokenNode:
    def __init__(self, token_id, prev=None, next=None) -> None:
        self.token_id = token_id
        self.prev: TokenNode = prev
        self.next: TokenNode = next
        # 针对b,b,,b,b这种序列 合并为bb,b,b后，第2个b标记失效，因此会到第3个b 合并为[bb,bb]
        self.valid = True


class TokenSequence:
    def __init__(self) -> None:
        self.head: TokenNode = None
        self.tail: TokenNode = None


class Entry:
    def __init__(self, priority, merge) -> None:
        self.priority = priority
        self.merge = merge
        self.node_list = OrderedDict()

    def __lt__(self, obj):
        return self.priority < obj.priority



def apply_merges(segments, merges, vocab):
    # vocab是 id -> bytes 要改成bytes - > id
    bytes_to_id = {v: k for k, v in vocab.items()}
    token_ids = []

    merge_to_id = {merge: i for i, merge in enumerate(merges)}

    for id, seg in enumerate(segments):
        collectors = {}  # merge - > list[node]
        pq = PriorityQueue()
        linked_list = TokenSequence()
        # 为当前链表构建merge map来进行合并操作 这个必须在循环中 否则会导致跨segment合并 巨大的bug!!!

        # 构建链表 和pair-> list[node] 索引
        for i, token_id in enumerate(seg):
            if linked_list.head == None:
                node = TokenNode(bytes([token_id]))
                linked_list.head = node
                linked_list.tail = linked_list.head
            else:
                # (tail<-node) (tail->node) 更新tail
                node = TokenNode(bytes([token_id]), linked_list.tail)
                linked_list.tail.next = node
                linked_list.tail = node
            if i < len(seg) - 1:
                pair = (bytes([seg[i]]), bytes([seg[i+1]]))
                if pair in merge_to_id:
                    if pair in collectors:
                        collectors[pair][node] = None
                    else:
                        collectors[pair] = OrderedDict([(node, None)])

        # 把有的pair初始化进入pq
        for pair in collectors:
            entry = Entry(merge_to_id[pair], pair)
            entry.node_list = collectors[pair]
            pq.put(entry)

        # 按照merge顺序合并
        while not pq.empty():
            entry: Entry = pq.get()
            merge = entry.merge
            node_list: OrderedDict = entry.node_list

            added_merge = set()

            for node in node_list:
                if not node.valid:
                    continue

                new_bytes = merge[0] + merge[1]
                # 删除 (b1,b2)
                node: TokenNode = node
                if node.prev:
                    old_pair = (node.prev.token_id, node.token_id)
                    if old_pair == merge:  # checkpoint
                        print('error (b1,b2,b3) = (b,b,b)')
                        print(
                            f' prev:{node.prev.token_id} , cur:{node.token_id},next:{node.next.token_id}')

                    if old_pair in merge_to_id:
                        collectors[old_pair].pop(node.prev)
                        # merge_map[old_pair].pop(node.prev)
                    # 新增 (b1,b2b3)
                    new_pair = (node.prev.token_id, new_bytes)

                    if new_pair == merge:  # checkpoint
                        print('error (b1,b2b3) ')

                    if new_pair in merge_to_id:
                        if new_pair in collectors:
                            collectors[new_pair][node.prev] = None
                        else:  # 出现了优先队列中之前没出现的情况
                            collectors[new_pair] = OrderedDict(
                                [(node.prev, None)])
                            added_merge.add(new_pair)
                    # todo
                # 删除 (b2,b3) 防止 b,b,b,b情况  我合并bb后有bb,b,b 然后bb中的第二个b想和后面的b合并

                node.next.valid = False

                # 新增 (b2b3,b4)
                if node.next.next:
                    # 丢掉b3 那么就得重新建立联系
                    neighbor = node.next.next
                    node.next = neighbor
                    neighbor.prev = node

                    new_pair = (new_bytes, neighbor.token_id)

                    if new_pair == merge:  # checkpoint
                        print('error (b1b2,b3) ')

                    if new_pair in merge_to_id:
                        if new_pair in collectors:
                            collectors[new_pair][node] = None
                        else:  # 出现了优先队列中之前没出现的情况
                            collectors[new_pair] = OrderedDict([(node, None)])
                            added_merge.add(new_pair)
                # 删掉node.next 否则 _ t h e 会先是_t -> h -> e 然后是_t -> (he) -> e 然后是_the->(he)->e 然后得到_thee
                # 删掉后 可以保证 _t h e变成 _t h e然后是 _t he 然后合并为_the
                else:
                    node.next = None

                node.token_id = new_bytes
            # 当前merge完成后 把新出现的merge加入到优先队列中
            for add_merge in added_merge:
                entry = Entry(merge_to_id[add_merge], add_merge)
                entry.node_list = collectors[add_merge]
                pq.put(entry)

        # 合并后 收集每个seg的token_ids

        p = linked_list.head
        while p:
            if p.valid:
                token_bytes = p.token_id
                try:
                    token_id = bytes_to_id[token_bytes]
                except:
                    print(f'erorr :{type(token_bytes)}')
                token_ids.append(token_id)

            p = p.next

    return token_ids



class BPETokenizer:
    def __init__(self, vocab: dict, merges: list, vocab_size: int, special_tokens: list = []):
        """初始化BPE分词器
        Args:
            vocab_size: 目标词汇表大小
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = vocab  # 词汇表
        self.merges = merges  # BPE合并规则
        self.word_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_token_pat = re.compile(
            "|".join(re.escape(tok) for tok in special_tokens))
        
    def encode(self, text: str):
        """
        编码文本
            :param text: 输入文本
            :return: 编码后的token列表
        """
        matches = self.special_token_pat.finditer(text)
        splits = self.special_token_pat.splititer(text)

        special_map = {tok: self.vocab_size - len(self.special_tokens) + i
                       for i, tok in enumerate(self.special_tokens)}

        # 把merges
        token_ids = []
        segments_chunks = []
        split_chunks = []
        for split_text in splits:
            segments = []
            for t in re.finditer(self.word_pattern, split_text):
                pretoken = t.group()
                segments.append(pretoken.encode('utf-8'))
            segments_chunks.append(segments)
            # 通过special token分隔的 这里单独加入一个special token
            try:
                match = next(matches)
                split_chunks.append(special_map[match.group()])
            except:
                pass
        min_proc = min(cpu_count(), len(segments_chunks))
        print(f'process num:{min_proc}')
        with Pool(processes=min_proc) as pool:
            # 使用partial来固定merges和vocab参数
            apply_merges_with_params = partial(
                apply_merges, merges=self.merges, vocab=self.vocab)
            chunks = pool.map(apply_merges_with_params, segments_chunks)

            for i, chunk in enumerate(chunks):
                token_ids.extend(chunk)
                if i < len(split_chunks):
                    token_ids.append(split_chunks[i])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """将token id序列解码为文本
        Args:
            ids: token id列表
        Returns:
            解码后的文本
        """

        pass




if __name__ == "__main__":
    
    
    pass