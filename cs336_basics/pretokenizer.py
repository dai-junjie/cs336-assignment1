import regex as re
import os
from typing import BinaryIO, Iterator, List

class PreTokenizer:
    def __init__(self,special_tokens:list[str]):
        if special_tokens:
            self.special_tokens = sorted(special_tokens,reverse=True,key=len)
            self.special_tokens_pat = '|'.join(re.escape(token) for token in self.special_tokens)
        else:
            self.special_tokens = []
    
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

    def corpus_iter(self,input_path:str) -> Iterator[List[str]]:
         """读取语料库 并进行分隔"""
         with open(input_path,'rb') as f:
            # 调用函数获取分块边界
            boundaries = self.find_chunk_boundaries(f,100,"<|endoftext|>".encode("utf-8"))
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i+1]
                f.seek(start)
                chunk_text = f.read(end - start).decode('utf-8',errors='ignore')
               
                for token in self.special_tokens:
                
                    chunk_text = chunk_text.replace(token,'')
                
                # 应用正则表达式进行分词
                yield chunk_text
            
    