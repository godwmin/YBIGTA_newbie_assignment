from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    """사탕 개수 쿼리를 처리하고 꺼낸 사탕의 맛 번호를 출력한다."""
    tokens = iter(map(int, sys.stdin.buffer.read().split()))
    query_count = next(tokens)

    tree = SegmentTree[int, int](
        values=[0] * 1_000_001,
        default=0,
        f_conv=lambda value: value,
        f_merge=lambda left, right: left + right,
    )
    outputs: list[str] = []

    for _ in range(query_count):
        command = next(tokens)
        value = next(tokens)

        if command == 1:
            flavor = tree.find_kth(value, lambda count: count)
            outputs.append(str(flavor))
            tree.modify(flavor, lambda count: count - 1)
        else:
            amount = next(tokens)
            tree.modify(
                value,
                lambda count: count + amount,
            )

    sys.stdout.write("\n".join(outputs) + ("\n" if outputs else ""))


if __name__ == "__main__":
    main()
