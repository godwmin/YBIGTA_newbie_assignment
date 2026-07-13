from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: 'Pair', b: 'Pair') -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    """수열 갱신과 구간 내 가장 큰 두 수의 합 쿼리를 처리한다."""
    tokens = iter(map(int, sys.stdin.buffer.read().split()))
    size = next(tokens)
    values = [next(tokens) for _ in range(size)]

    tree = SegmentTree[int, Pair](
        values=values,
        default=Pair.default(),
        f_conv=Pair.f_conv,
        f_merge=Pair.f_merge,
    )
    query_count = next(tokens)
    outputs: list[str] = []

    for _ in range(query_count):
        command = next(tokens)
        first = next(tokens)
        second = next(tokens)

        if command == 1:
            tree.set(first - 1, second)
        else:
            # Pair에 가장 큰 값 두 개
            result = tree.query(first - 1, second - 1)
            outputs.append(str(result.sum()))

    sys.stdout.write("\n".join(outputs) + ("\n" if outputs else ""))


if __name__ == "__main__":
    main()
