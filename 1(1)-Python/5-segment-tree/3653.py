from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    """각 영화를 꺼낼 때 위에 쌓여 있던 영화의 수를 출력한다."""
    tokens = iter(map(int, sys.stdin.buffer.read().split()))
    test_count = next(tokens)
    outputs: list[str] = []

    for _ in range(test_count):
        movie_count = next(tokens)
        request_count = next(tokens)

        # 앞쪽 request_count칸은 영화를 맨 위로 옮길 자리로 비워 둔다
        tree = SegmentTree[int, int](
            values=[0] * request_count + [1] * movie_count,
            default=0,
            f_conv=lambda value: value,
            f_merge=lambda left, right: left + right,
        )
        positions = [0] * (movie_count + 1)

        for movie in range(1, movie_count + 1):
            positions[movie] = request_count + movie - 1

        next_top = request_count - 1
        answers: list[str] = []

        for _ in range(request_count):
            movie = next(tokens)
            current = positions[movie]

            # 현재 위치보다 앞에 있는 1의 개수 -> 위에 쌓인 영화 수
            above = tree.query(0, current - 1)
            answers.append(str(above))

            tree.set(current, 0)
            tree.set(next_top, 1)
            positions[movie] = next_top
            next_top -= 1

        outputs.append(" ".join(answers))

    sys.stdout.write("\n".join(outputs) + ("\n" if outputs else ""))


if __name__ == "__main__":
    main()
