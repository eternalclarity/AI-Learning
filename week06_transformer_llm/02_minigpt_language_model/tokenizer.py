"""字符级 tokenizer, 把每个字符映射成整数 ID，再把整数 ID 还原成字符"""

from __future__ import annotations

# 用于保存和读取 tokenizer 配置
import json
from pathlib import Path

# 未知字符对应的特殊 token
UNK = "<unk>"


class CharTokenizer:
    """ 字符级 Tokenizer """

    def __init__(self, chars: list[str]) -> None:
        # 建立 ID → token 的列表，并把 <unk> 放在第 0 位
        self.idx_to_token = [UNK, *chars]   # 解包 *列表/元组 **字典

        # 建立 token → ID 的字典
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}

        # <unk> 的 token ID 固定为 0
        self.unk_id = 0

    # 装饰器语法
    # 普通方法变成“类方法”,用于 可直接通过类调用 CharTokenizer.from_training_text("hello")
    # 用于 根据训练文本创建一个新的 tokenizer， 而不是 在操作一个已经存在的 tokenizer，即: 另一种创建对象的方法，工厂方法
    @classmethod
    def from_training_text(cls, text: str) -> "CharTokenizer":
        """ 从训练文本中提取所有不重复字符，并按顺序排序 """
        return cls(sorted(set(text)))   # self: 当前对象本身。 cls：当前类本身, 相当于 return CharTokenizer(sorted(set(text)))

    # 让一个方法可以像普通属性一样访问 tokenizer.vocab_size
    @property
    def vocab_size(self) -> int:
        """ 返回词表大小 """
        return len(self.idx_to_token)

    def encode(self, text: str) -> list[int]:
        """
        将字符串中的每个字符转换成对应的 token ID
        未见过的字符统一转换成 <unk> 的 ID
        """
        return [self.token_to_idx.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 转回字符，并重新拼成字符串
        非法 ID 使用 <unk> 代替"""
        return "".join(self.idx_to_token[int(i)] if 0 <= int(i) < self.vocab_size else UNK for i in ids)

    def save(self, path: Path) -> None:
        """
        创建保存 tokenizer 的目录
        将字符词表保存为 JSON 文件
        不保存第 0 位的 <unk>，加载时会自动重新添加
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chars": self.idx_to_token[1:]}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        """ 从 JSON 文件读取字符词表，并重新创建 tokenizer """
        return cls(json.loads(path.read_text(encoding="utf-8"))["chars"])
