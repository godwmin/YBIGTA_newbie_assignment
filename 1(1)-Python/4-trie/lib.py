from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """주어진 원소열을 트라이에 삽입하고 마지막 노드를 끝으로 표시한다."""
        current = 0

        for item in seq:
            next_node: Optional[int] = None

            for child in self[current].children:
                if self[child].body == item:
                    next_node = child
                    break

            if next_node is None:
                next_node = len(self)
                self.append(TrieNode(body=item))
                self[current].children.append(next_node)

            current = next_node

        self[current].is_end = True

    # 구현하세요!
