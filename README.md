# wuji_ros

Wuji Glove + 4뷰 RealSense RGB-D + OptiTrack(물체) 녹화에서 오른손의 MANO pose를 world 좌표로 추정한다.
손가락 관절은 glove가, 손 전체의 world 6D는 4뷰 RGB-D 정합이 맡는다.
계획과 결정 사항은 [plan.md](plan.md), 구조와 좌표 규약은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)에 있다.

현재 상태: M1(glove 브리지) 코드 작성. 녹화 PC에서의 실행 확인 전.

## 구성

| 경로 | 내용 |
|---|---|
| `plan.md` | 목표, 마일스톤, 완료 조건, 결정 사항 |
| `docs/` | 구조(`ARCHITECTURE.md`), 미해결 항목(`TODO.md`), 실험 기록(`experiments/`) |
| `calibration/<name>/` | 리그 캘리브 파일 (커밋). 현재 `20260824_v2` |
| `wuji_ros/` | 파이썬 패키지. 현재 `bridge/`(glove → ROS2) |
| `scripts/` | entrypoint. 현재 `glove_bridge.py`, 녹화 PC용 `shells/{setup_bridge,glove_bridge,record_glove}.sh` |
| `tests/` | 데이터·계약 테스트 |
| `environment.yml` | 분석 env `wuji_ros` 정의 |
| `requirements-bridge.txt` | 녹화 PC 브리지 의존성 (`python3.10 --user`) |
| `third_party/wuji-sdk` | Wuji SDK 예제·문서 (submodule, 수정 금지) |
| `third_party/hamer` | HaMeR upstream (submodule, 수정 금지) |
| `assets/` | 외부 모델 자산 심볼릭 링크 (git-ignored) |
| `data/`, `output/`, `scratch/` | 추출 데이터, 결과, 임시 확인용 (git-ignored) |

## 환경

실행 위치별로 env가 다르다.

| 단계 | 어디서 | env |
|---|---|---|
| glove 브리지, `ros2 bag record` | 녹화 PC (장갑·카메라 같은 PC) | ROS2 Humble system python3.10 + `wuji-sdk==2026.8.31` (user site) |
| bag 추출, HaMeR, MANO fit, 평가 | 분석 서버 | conda `wuji_ros` |

분석 env 생성 (한 번):

```bash
cd /home/user/extra_workdir/wuji_ros
git submodule update --init third_party/wuji-sdk third_party/hamer
conda env create -f environment.yml
conda run --no-capture-output -n wuji_ros python -m pip install --no-build-isolation \
    "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
conda run --no-capture-output -n wuji_ros python -c \
    "import torch, smplx, chumpy, pytorch_lightning, rosbags, cv2; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
```

`third_party/hamer`는 재귀 초기화하지 않는다. upstream의 `third-party/ViTPose`는 데모 검출기용이라 쓰지 않는다.

변환 계약 테스트 (ROS·SDK 없이 돈다):

```bash
cd /home/user/extra_workdir/wuji_ros
conda run --no-capture-output -n wuji_ros python -m unittest discover -s tests -v
```

## 녹화 PC: glove 브리지

wuji-sdk 휠은 glibc ≥ 2.34를 요구한다. 그래서 브리지는 녹화 PC(Ubuntu 22.04 + ROS2 Humble)에서만 돈다.
mocap_ros_py와 같은 방식으로 venv 없이 `python3.10`으로 실행한다. rclpy가 python3.10용으로 빌드돼 있어서, 그냥 `python3`로 돌리면 `No module named 'rclpy._rclpy_pybind11'`이 난다(mocap_ros_py `e94e059`).
wuji-sdk는 런타임 의존성이 없으므로 `--user`로 깔아도 ROS의 numpy 등은 바뀌지 않는다.

처음 한 번 (repo를 받고 `python3.10`에 wuji-sdk를 깐다):

```bash
git clone https://github.com/jinmiles/wuji_ros.git
cd wuji_ros
git submodule update --init third_party/wuji-sdk   # 캘리브레이션 예제용. 브리지 실행에는 필요 없다
bash scripts/shells/setup_bridge.sh                # 마지막 줄에 "bridge env ok"
```

녹화 전 캘리브레이션 (SDK 예제, `python3.10`으로 실행):

- 손 모델: named user를 만든 뒤 `third_party/wuji-sdk/examples/python/wuji_glove/5.calibration.py`.
- 촉각: `third_party/wuji-sdk/examples/python/wuji_glove/7.tactile_calibration.py`. 이 캘리브가 없으면 `tactile_binary`, `tactile_residual`이 나오지 않는다.

녹화 (터미널 두 개, repo 루트에서):

```bash
bash scripts/shells/glove_bridge.sh                # 터미널 1. 장갑이 여러 개면 --sn <SN>
bash scripts/shells/record_glove.sh <name>         # 터미널 2. data/bags/<name>에 기록
```

- 장갑 NIC가 기본 경로 NIC와 다르면 SDK scan(멀티캐스트)이 장갑을 못 찾는다(`No devices found`). 이때는 주소로 바로 연결한다: `bash scripts/shells/glove_bridge.sh --address 192.168.1.101:50001`.
- 연결이 안 되면 `bash scripts/shells/glove_netcheck.sh`로 NIC, ARP 응답, 장갑 패킷을 확인한다. PC IP는 설명서대로 `192.168.1.50/24`(장갑 `.100`/`.101`과 겹치지 않게)로 둔다.
- `record_glove.sh`는 `/wuji_glove/right/*` 전부와 `/tf_static`을 기록한다. 카메라·mocap 토픽은 뒤에 인자로 붙인다.
- 브리지를 먼저 띄운다. `info`, `/tf_static`은 latched라서 녹화를 나중에 시작해도 bag에 들어간다.

- 발행 토픽: SDK의 Wuji Glove 스트림 전부(`plan.md` M1 표).
  - 손 자세: `/wuji_glove/right/{emf_poses, tip_poses, hand_skeleton, hand_joint_angles}`
  - IMU: `/wuji_glove/right/imu_raw/{palm,thumb,index,middle,ring,pinky}`, `/wuji_glove/right/imu_data/{...}`
  - 촉각: `/wuji_glove/right/{tactile, tactile_binary, tactile_residual}`(24×31 `32FC1` 이미지), `/wuji_glove/right/tactile_zones/{palm,...,pinky}`, `/wuji_glove/right/tactile_point_cloud`
  - 변환: `/wuji_glove/right/tf`(IMU 기반 `waist → r_wrist`, 전역 `/tf`에는 넣지 않음), `/tf_static`
  - 메타: `/wuji_glove/right/info`
- 시작 5 s 안에 frame이 없는 스트림은 경고로 남는다(녹화는 계속).
- 모든 `header.stamp`는 장치의 `timestamp_us`(EMF 샘플링 시각, UTC)다.
- 멈추는 조건:
  - device stamp가 호스트 시계와 1 s 넘게 다를 때(time sync 전).
  - `emf_poses`가 2 s 동안 오지 않을 때.
  - 관절·손가락 개수, 촉각 taxel 수(24×31), 촉각 점 수(526)가 계약과 다를 때.
- 확인: `ros2 topic hz /wuji_glove/right/hand_skeleton`(≈ 120 Hz), `ros2 topic echo --once /wuji_glove/right/info`(`sdk_topics`, `tactile_point_cloud_layout`).

## 자산

MANO와 HaMeR checkpoint는 크고 별도 라이선스라 repo에 넣지 않는다. 서버에 있는 파일을 심볼릭 링크로 가리킨다.
`assets/hamer`는 HaMeR의 `_DATA` 레이아웃(`hamer_ckpts/`, `data/mano/`, `data/mano_mean_params.npz`)을 따른다.

```bash
cd /home/user/extra_workdir/wuji_ros
mkdir -p assets/hamer/data
ln -sfn /mnt/EvalAI/assets/mano assets/mano
ln -sfn /mnt/EvalAI/assets/mano assets/hamer/data/mano
ln -sfn /mnt/MV-SAM3D/submodules/Dyn-HaMR/_DATA/hamer_ckpts assets/hamer/hamer_ckpts
ln -sfn /mnt/MV-SAM3D/submodules/Dyn-HaMR/_DATA/data/mano_mean_params.npz assets/hamer/data/mano_mean_params.npz
```

`hamer.ckpt`는 위치만 Dyn-HaMR 폴더일 뿐 HaMeR 공식 checkpoint다. Dyn-HaMR 코드는 쓰지 않는다.
`MANO_RIGHT.pkl`은 두 위치의 파일이 같다(md5 `fd5a9d35…`).
