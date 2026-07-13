
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
if command -v conda >/dev/null 2>&1; then
    conda_base=$(conda info --base)
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    conda_base="$HOME/miniconda3"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    conda_base="$HOME/anaconda3"
else
    os_name=$(uname -s)
    architecture=$(uname -m)

    if [[ "$os_name" == "Darwin" ]]; then
        platform="MacOSX"
    elif [[ "$os_name" == "Linux" ]]; then
        platform="Linux"
    else
        echo "[INFO] 지원하지 않는 운영체제입니다: $os_name"
        exit 1
    fi

    if [[ "$architecture" == "aarch64" ]]; then
        architecture="aarch64"
    elif [[ "$architecture" != "arm64" && "$architecture" != "x86_64" ]]; then
        echo "[INFO] 지원하지 않는 CPU 구조입니다: $architecture"
        exit 1
    fi

    installer="/tmp/miniconda_installer.sh"
    installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-${platform}-${architecture}.sh"

    echo "[INFO] Miniconda를 설치합니다."
    curl -fsSL "$installer_url" -o "$installer" || exit 1
    /bin/bash "$installer" -b -p "$HOME/miniconda3" || exit 1
    conda_base="$HOME/miniconda3"
fi

source "$conda_base/etc/profile.d/conda.sh" || exit 1


# Conda 환셩 생성 및 활성화
## TODO
if ! conda env list | grep -qE '^myenv[[:space:]]'; then
    echo "[INFO] myenv 가상환경을 생성합니다."
    conda create -y -n myenv python=3.9 || exit 1
fi
conda activate myenv || exit 1

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
python -m pip install mypy==1.19.1 || exit 1
mkdir -p output
shopt -s nullglob

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    problem_number="${file#*_}"
    problem_number="${problem_number%.py}"
    input_file="../input/${problem_number}_input"
    output_file="../output/${problem_number}_output"

    if [[ ! -f "$input_file" ]]; then
        echo "[INFO] 입력 파일을 찾을 수 없습니다: $input_file"
        exit 1
    fi

    echo "[INFO] 실행 중: $file"
    python "$file" < "$input_file" > "$output_file" || exit 1

done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
python -m mypy . > ../mypy_log.txt 2>&1
mypy_status=$?
if [[ $mypy_status -eq 0 ]]; then
    echo "[INFO] mypy 테스트: 성공"
else
    echo "[INFO] mypy 테스트: 실패 (mypy_log.txt 확인)"
fi

# conda.yml 파일 생성
## TODO
conda env export --no-builds > ../conda.yml || exit 1

# 가상환경 비활성화
## TODO
conda deactivate

