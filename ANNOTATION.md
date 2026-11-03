# 녹화 폴더 → 물체에 닿게 보정한 손 + 오버레이

수집기가 고정 주기로 뽑아 둔 녹화 폴더 하나에서 world 좌표의 오른손 MANO를 만든다. 물체를 들고 있는 프레임에서는 손 위치를 물체에 닿게 보정한다. 오버레이 영상도 모두 만든다.
각 단계의 내용은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), 결정 사항은 [plan.md](plan.md) §6 18, 실험 결과는 [docs/experiments/001_glove_hand_pose/experiments.md](docs/experiments/001_glove_hand_pose/experiments.md)(E014–E016)에 있다.

## 실행

```bash
cd /home/user/wj/wuji_ros   # 분석 서버. 다른 PC는 clone한 폴더 (아래 "설치")
./annotate.sh data/brl_mocap_cam_hand/20261022_163051_mocap_mk4_wuji_fixed_rate
```

- 인자는 녹화 폴더 경로 하나다. 세션 이름은 폴더 이름이다.
- 여러 GPU 중 하나를 고르려면 앞에 `CUDA_VISIBLE_DEVICES=0`을 붙인다.
- 로그를 남기려면 뒤에 `> annotate.log 2>&1`을 붙인다.
- 단계별 출력이 이미 있으면 그 단계는 건너뛴다. 처음부터 다시 만들려면 `--force`를 붙인다.
- `20261022_163051`(0.5 s 간격 50프레임)은 RTX 3090 한 장에서 약 8–9분 걸린다(손 단계 약 5분, 오버레이 3종과 촉각 2종 약 3분. 손 메시 위 촉각은 41 s).
- 옵션은 기본값이 맞지 않을 때만 쓴다: `--camera-config YAML`, `--object-mesh STL`, `--calibration NAME`, `--every 초`.

실행 순서: `import → segment → glovefit → keypoints → fit4 → contact → overlay(손) → overlay(손 + 물체) → overlay(보정한 손 + 물체) → tactile → tactile(손 메시 위)`. 단계를 하나씩 돌리는 명령은 README "분석 파이프라인"에 있다.

## 설치 (로봇룸 PC 등 새 PC)

분석 서버(`/home/user/wj/wuji_ros`)에는 이미 설치되어 있다. 새 PC에서는 아래를 한 번만 한다.
확인한 환경은 분석 서버(NVIDIA 드라이버 570, RTX 3090 24 GB)뿐이다. 더 작은 GPU에서는 아직 돌려 보지 않았다.

필요한 것:
- NVIDIA GPU와 드라이버. `keypoints`(HaMeR)와 `fit4`는 CUDA로만 돈다. env의 torch가 CUDA 11.8 빌드이므로 `nvidia-smi`의 "CUDA Version"이 11.8 이상이어야 한다.
- conda(Miniconda 또는 Anaconda), git, ffmpeg(`sudo apt install ffmpeg`). ffmpeg가 없으면 mp4/gif는 건너뛰고 jpg만 남는다.
- 디스크: HaMeR 압축 파일 6.0 GB(풀고 나면 지워도 된다), 풀린 checkpoint 2.6 GB, conda env.
- 녹화 PC와 같은 PC라면 녹화 중에는 돌리지 않는다. CPU·GPU·디스크를 많이 쓴다.

```bash
git clone https://github.com/jinmiles/wuji_ros.git
cd wuji_ros
bash scripts/shells/setup_annotation.sh --from user@<분석 서버 주소>:/home/user/wj/wuji_ros
```

마지막 줄에 `annotation env ok`가 나오면 된다. 중간에 멈추면 원인을 고치고 같은 명령을 다시 실행한다. 이미 끝난 단계는 건너뛴다.

`setup_annotation.sh`가 하는 일:
1. HaMeR 코드: `third_party/hamer` submodule만 받는다. `third_party/wuji-sdk`는 녹화 브리지용이라 분석에는 필요 없다.
2. conda env: `environment.yml`로 `wuji_ros` env를 만들고 chumpy를 따로 설치한다.
3. HaMeR checkpoint: 공식 demo 데이터(6.0 GB)를 `scratch/`에 받는다. 필요한 것만 `assets/hamer/`로 옮기고 압축 파일은 지운다.
4. MANO: `MANO_RIGHT.pkl`은 라이선스 때문에 저장소에 없다.
   - `--from`의 `assets/mano/MANO_RIGHT.pkl`을 scp로 복사하고 md5(`fd5a9d35f914987cf1cc04ffe338caa1`)를 확인한다. 다르면 멈춘다.
   - HaMeR도 같은 파일을 쓴다(`assets/hamer/data/mano` 링크).
   - 분석 서버에 접속할 수 없으면 MANO 사이트(https://mano.is.tue.mpg.de, 가입 후 라이선스 동의)에서 받거나 USB로 옮긴다. 파일을 `assets/mano/MANO_RIGHT.pkl`에 두고 `--from` 없이 실행한다.
5. 확인: env의 패키지를 import해 `2.0.1+cu118 True 1.23.5`(torch, CUDA 사용 가능, numpy)인지 보고, checkpoint와 MANO 파일이 있는지 본다.

- 스크립트는 `annotate.sh`와 같이 `PYTHONNOUSERSITE=1`, `unset PYTHONPATH`로 실행한다.
  녹화 PC의 브리지용 user site 패키지(`~/.local/lib/python3.10`)와 ROS 2를 source한 터미널의 PYTHONPATH가 env보다 먼저 읽히지 않게 한다. 다른 PC에서도 해가 없다.

설치 후 `assets/`는 다음과 같다(git이 무시하는 폴더).

```
assets/mano/MANO_RIGHT.pkl
assets/hamer/data/mano -> ../../mano
assets/hamer/data/mano_mean_params.npz
assets/hamer/hamer_ckpts/model_config.yaml, dataset_config.yaml, checkpoints/hamer.ckpt
```

첫 실행:
- 녹화 데이터를 아래 "입력 데이터 위치"대로 두고 `./annotate.sh`를 돌린다.
- 분석 서버의 `20261022_163051`을 복사해 돌렸다면 분석 서버 결과와 비교한다.
  - 복사: `rsync -a --exclude '*.zip' --exclude t2r_ext.npz user@<분석 서버 주소>:/home/user/wj/wuji_ros/data/brl_mocap_cam_hand/ data/brl_mocap_cam_hand/`
  - `output/<세션>/contact/summary.json`의 `held_frames`가 25개인지 본다.
  - 로그의 물체 "CAD fit RMSE"가 0.30 mm인지 본다.
- GPU가 달라 숫자가 조금 다를 수 있다.

## 입력 데이터 위치

```
wuji_ros/
├── annotate.sh
├── data/brl_mocap_cam_hand/
│   ├── <세션>/                                ← 수집기 출력 그대로
│   │   ├── image_metadata.csv
│   │   ├── capture_config.json
│   │   ├── wuji_info.json
│   │   ├── camera1/color/000001.png …, camera1/depth/000001.png …   (camera1–camera4)
│   │   ├── wuji/000001.npz …                  (장갑)
│   │   └── mocap/000001.npz …                 (물체 마커)
│   ├── config_rerun_object_mk4_wuji.yaml      ← 카메라 내부 파라미터 + 물체 마커 오프셋
│   ├── 01_Hexagonal_Prism_with_tracker.stl    ← 물체 CAD (mm, OBJ 좌표)
│   └── w2c_ext.npz                            ← 녹화와 같이 온 외부 파라미터 (캘리브 확인용)
└── calibration/20261015/                      ← w2c_ext.npz, t2r_ext.npz, mocap2cam.npy (저장소에 포함, 읽기 전용)
```

- yaml과 STL은 녹화 폴더 옆(부모 폴더)에서 찾는다. 하나뿐이어야 한다. 여러 개면 옵션으로 지정한다.
  녹화 폴더를 심볼릭 링크로 두면 링크가 가리키는 실제 폴더 옆에서 찾는다.
- 캘리브는 추정하지도 고치지도 않는다. 아래 순서에서 먼저 해당하는 것을 쓴다.
  1. `--calibration NAME`
  2. `wuji_ros/rig.py`의 `SESSION_CALIBRATION`에 적힌 값
  3. 녹화와 같이 온 `w2c_ext.npz`(녹화 폴더나 그 부모)와 바이트가 같은(sha256) `calibration/<이름>/w2c_ext.npz`
- 셋 다 없으면 멈춘다. 그때는 연구실의 캘리브 파일을 받는다. 정한 캘리브는 `data/<세션>/manifest.json`의 `calibration`, `calibration_source`에 남는다.
- `import`는 원본을 건드리지 않고 `data/<세션>/`에 같은 레이아웃으로 정리한다.

## 출력 (`output/<세션>/`)

| 경로 | 내용 |
|---|---|
| `contact/NNNNNN.npz` | **최종 손.** 아래 표 |
| `contact/summary.json` | 들고 있는 프레임 목록, 이동량, 보정 전후 지표 |
| `overlay_contact_object/`, `overlay_contact_object.{mp4,gif}` | 보정한 손 + 물체 |
| `overlay_object/`, `overlay_object.{mp4,gif}` | 보정 전(`fit4`) 손 + 물체 |
| `overlay/`, `overlay.{mp4,gif}` | 보정 전 손만 |
| `tactile/`, `tactile.{mp4,gif}` | 장갑 촉각 맵 (24×31) |
| `tactile_hand/`, `tactile_hand.{mp4,gif}` | 촉각 taxel을 `fit4` 손 메시 위에 그린 것. 펼친 손(손바닥), 자세 그대로(손바닥 쪽, 엄지 쪽). 압력은 SDK 값 그대로 |
| `fit4/`, `glovefit/`, `keypoints/`, `segment/`, `scene/` | 중간 단계 출력 |

오버레이에 참가자 얼굴이 보이므로 공유할 때 주의한다.

### `contact/NNNNNN.npz`

`fit4`에 결과가 있는 프레임마다 하나씩 있다. 좌표는 캘리브 world, 단위는 m다.

| 키 | 내용 |
|---|---|
| `verts_world` (778,3), `joints_world` (21,3) | MANO 정점과 관절(OpenPose-21 순서) |
| `R_world_from_mano` (3,3), `wrist_world` (3,), `betas` (10,) | 손 회전, 손목 위치, 손 크기 β |
| `T_world_from_wrist` (4,4) | 장갑 `r_wrist` 좌표계 → world (촉각 taxel, 장갑 골격용) |
| `held` | 물체를 들고 있는 프레임으로 판정했는지 |
| `t_world` (3,) | 보정한 손 이동량. 들고 있지 않은 프레임은 0 |
| `T_world_from_obj` (4,4) | 물체 자세(mocap 마커 + `mocap2cam`, camera1 경유) |
| `obj_lift_mm`, `obj_move_mm`, `obj_rot_deg` | 첫 프레임 대비 물체 높이, 이동, 회전 |
| `min_sd_mm_before/after`, `inside_before/after` | 보정 전후 손-물체 최소 부호 거리(음수는 관통 깊이), 0.1 mm 넘게 물체 안에 든 정점 수 |
| `reproj_px_before`, `reproj_px`, `surface_mm_before`, `surface_mm_median` | 보정 전후 영상 일치(HaMeR 2D 재투영, depth 표면) |
| `views`, `labels_mirrored`, `gravity_err_deg`, `imu_rot_err_deg` 등 | `fit4`의 진단 값 그대로 |

- 손가락 자세 θ는 `glovefit/NNNNNN.npz`의 `hand_pose`(15,3,3 회전행렬)다. `contact`는 손가락을 바꾸지 않는다.
- MANO 파라미터로 다시 만들 때의 식:
  `verts_world = R_world_from_mano · (MANO(β, θ, 회전 I, 이동 0) − 그 손목 관절) + wrist_world`

## 보정 규칙

- **들고 있는 프레임:** 아래 두 조건을 모두 만족하는 프레임이다.
  - 물체의 가장 낮은 점이 첫 프레임(책상 위)보다 5 mm 넘게 들려 있다.
  - `fit4` 손이 물체에서 20 mm 안에 있다.
  - 옆에 내려놓은 물체, 책상 위에서 돌리는 물체는 해당하지 않는다.
- **보정:** 손가락, β, 손 회전은 `fit4` 그대로 둔다. 손이 물체에 닿고 관통하지 않게 하는 가장 작은 위치 이동만 한다.
- **주의:**
  - 물체 위치(mocap)를 입력으로 쓰므로 손-물체 접촉은 검증이 아니다. 검증은 영상 지표로 한다.
  - 손 회전이 틀린 프레임은 위치만 옮기면 영상과 더 어긋난다. `20261022_163051`에서는 f144, f184, f224다.

## 새 세션을 돌릴 때

1. 녹화 폴더를 `data/brl_mocap_cam_hand/<새 세션>/`에 넣고 `./annotate.sh data/brl_mocap_cam_hand/<새 세션>`을 실행한다.
2. 캘리브가 정해지지 않아 멈추면(위 순서 1–3) 연구실 파일을 받아 `calibration/<이름>/`에 두고 `--calibration <이름>`으로 다시 돌린다.
3. yaml의 `object.marker_offsets`가 그 녹화의 마커 이름 대응과 맞는지 확인한다. 녹화마다 바뀐 적이 있다.
   로그의 "CAD fit RMSE"가 수 mm 이하이면 맞다(`20261022_163051`은 0.30 mm).
