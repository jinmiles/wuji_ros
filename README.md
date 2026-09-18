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
| `scripts/` | entrypoint. 현재 `glove_bridge.py` |
| `tests/` | 데이터·계약 테스트 |
| `environment.yml` | 분석 env `wuji_ros` 정의 |
| `requirements-bridge.txt` | 녹화 PC 브리지 venv 정의 |
| `third_party/wuji-sdk` | Wuji SDK 예제·문서 (submodule, 수정 금지) |
| `third_party/hamer` | HaMeR upstream (submodule, 수정 금지) |
| `assets/` | 외부 모델 자산 심볼릭 링크 (git-ignored) |
| `data/`, `output/`, `scratch/` | 추출 데이터, 결과, 임시 확인용 (git-ignored) |

## 환경

실행 위치별로 env가 다르다.

| 단계 | 어디서 | env |
|---|---|---|
| glove 브리지, `ros2 bag record` | 녹화 PC (장갑·카메라 같은 PC) | ROS2 Humble system python3.10 + `wuji-sdk==2026.8.31` |
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
rclpy, `sensor_msgs_py`, `tf2_ros`, numpy는 ROS의 system 패키지를 쓰므로, venv를 `--system-site-packages`로 만든다.

설치 (wuji_ros repo 루트에서 한 번):

```bash
source /opt/ros/humble/setup.bash
python3.10 -m venv --system-site-packages .venv-bridge
.venv-bridge/bin/python -m pip install -r requirements-bridge.txt
```

실행 (wuji_ros repo 루트에서):

```bash
source /opt/ros/humble/setup.bash
.venv-bridge/bin/python scripts/glove_bridge.py            # 장갑이 여러 개면 --sn <SN>
```

- 발행 토픽: `/wuji_glove/right/{emf_poses, hand_skeleton, hand_joint_angles, imu_raw/palm, imu_data/palm, info}`, `/tf_static`.
- 모든 `header.stamp`는 장치의 `timestamp_us`(EMF 샘플링 시각, UTC)다.
- 멈추는 조건:
  - device stamp가 호스트 시계와 1 s 넘게 다를 때(time sync 전).
  - `emf_poses`가 2 s 동안 오지 않을 때.
  - 관절·손가락 개수가 계약과 다를 때.
- 확인: `ros2 topic hz /wuji_glove/right/hand_skeleton`(≈ 120 Hz), `ros2 topic echo --once /wuji_glove/right/info`.

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
