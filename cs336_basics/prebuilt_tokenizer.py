from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 初始化 tokenizer（用BPE模型）
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 使用空格作为预分词器（你也可以用BertPreTokenizer等）
tokenizer.pre_tokenizer = Whitespace()

# 设置训练参数：你可以指定 vocab size
trainer = BpeTrainer(
    vocab_size=300,
    show_progress=True,
    special_tokens=["[UNK]", "<|endoftext|>"]
)

# 你的语料文件路径（可以是多个文本文件）
# files = ["data/TinyStoriesV2-GPT4-train.txt"]
files = ["tests/fixtures/tinystories_sample_5M.txt"]

# 开始训练
tokenizer.train(files, trainer)

# 保存到本地
tokenizer.save("my-bpe-tokenizer.json")


