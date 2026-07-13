from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    def __init__(
        self,
        values: list[T],
        default: U,
        f_conv: Callable[[T], U],
        f_merge: Callable[[U, U], U],
    ) -> None:
        """원본 배열과 결합 규칙을 이용해 세그먼트 트리를 생성한다."""
        self.n = len(values)
        self.default = default
        self.f_conv = f_conv
        self.f_merge = f_merge

        self.size = 1
        while self.size < self.n:
            self.size *= 2

        self.tree: list[U] = [default] * (self.size * 2)

        for index, value in enumerate(values):
            self.tree[self.size + index] = self.f_conv(value)

        for node in range(self.size - 1, 0, -1):
            self.tree[node] = self.f_merge(
                self.tree[node * 2],
                self.tree[node * 2 + 1],
            )

    def _rebuild_ancestors(self, node: int) -> None:
        """변경된 노드의 부모들을 루트까지 다시 계산한다."""
        while node > 1:
            node //= 2
            self.tree[node] = self.f_merge(
                self.tree[node * 2],
                self.tree[node * 2 + 1],
            )

    def set(self, index: int, value: T) -> None:
        """원본 배열의 index 위치를 value로 교체한다."""
        if not 0 <= index < self.n:
            raise IndexError("segment tree index out of range")

        node = self.size + index
        self.tree[node] = self.f_conv(value)
        self._rebuild_ancestors(node)

    def modify(self, index: int, f: Callable[[U], U]) -> None:
        """index 위치의 트리 값에 변경 함수 f를 적용한다."""
        if not 0 <= index < self.n:
            raise IndexError("segment tree index out of range")

        node = self.size + index
        self.tree[node] = f(self.tree[node])
        self._rebuild_ancestors(node)

    def query(self, left: int, right: int) -> U:
        """left부터 right까지 양 끝을 포함한 구간 결과를 반환한다."""
        if left > right:
            return self.default
        if left < 0 or right >= self.n:
            raise IndexError("segment tree query range out of bounds")

        left += self.size
        right += self.size
        left_result = self.default
        right_result = self.default

        while left <= right:
            if left % 2 == 1:
                left_result = self.f_merge(left_result, self.tree[left])
                left += 1

            if right % 2 == 0:
                right_result = self.f_merge(self.tree[right], right_result)
                right -= 1

            left //= 2
            right //= 2

        return self.f_merge(left_result, right_result)

    def find_kth(self, k: int, measure: Callable[[U], int]) -> int:
        """누적 크기를 기준으로 1부터 센 k번째 원소의 인덱스를 반환한다."""
        if k < 1 or k > measure(self.tree[1]):
            raise IndexError("k is out of range")

        node = 1

        while node < self.size:
            left_child = node * 2
            left_count = measure(self.tree[left_child])

            if k <= left_count:
                node = left_child
            else:
                k -= left_count
                node = left_child + 1

        return node - self.size
