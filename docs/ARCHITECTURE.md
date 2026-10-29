# 구조와 규약

[plan.md](../plan.md)의 설계를 코드 구조와 계약으로 옮긴 문서다. 코드가 생기면 실제 모듈에 맞춰 갱신한다.

## 기본 경로

녹화 PC에서 브리지로 bag을 기록하고, 분석 서버에서 다음 순서로 처리한다.

```
extract → segment(table, depth+RGB box) → G1 → keypoints(HaMeR, IMU 손 방향으로 재박스) → V1(fit4, 4뷰 world) → contact(물체를 쥔 프레임의 손 위치) → calib check → overlay / tactile
```

계획의 순서(`extract → calib check → keypoints → segment → G1 → shape → V1 → evaluate`)와 다른 점: 손 박스가 HaMeR 전에 필요해 `segment`가 앞에 오고, 2차 박스가 glove IMU의 world 자세(G1 필요)를 써서 G1이 `keypoints` 앞에 온다. β는 기본으로 G1 값을 쓴다. 영상 β는 `fit4 --shape image` 옵션이다(plan.md §6 16). evaluate는 아직 없다.

```
```

- G1: glove 관절을 MANO 관절각으로 변환한다(손목 좌표계, world 없음).
- V1: 손가락 관절을 G1 값으로 고정하고, 손 전체의 world 6D를 4뷰 RGB-D로 맞춘다.
- 영상으로 손가락 관절을 다시 맞추는 단계(V2)는 없다(사용자 결정 2026-10-27, plan.md §6 15). `fit4`의 손가락은 G1 값 그대로다.
- 대체 경로(예: 관절각을 HaMeR로 추정)는 비교 실험용이다. 기본 경로로 바꾸려면 명시적 요청이 있어야 한다.

## 모듈

| 위치 | 역할 |
|---|---|
| `wuji_ros/paths.py` | repo 경로, `data/`·`output/` 레이아웃, `assets/`·`calibration/` 경로의 단일 출처, logger (구현됨) |
| `wuji_ros/bridge/convert.py` | SDK frame → 메시지 필드 값. ROS·SDK import 없는 순수 함수 (구현됨, `tests/test_bridge_convert.py`) |
| `wuji_ros/bridge/node.py` | 녹화 PC용 ROS2 노드 `wuji_glove_bridge` (구현됨) |
| `wuji_ros/bagio.py` | bag → `data/<session>/` (마스터 클럭 동기), `Session` reader (구현됨). bag을 SQLite로 직접 읽어 `metadata.yaml`이 없어도 된다. `import_dir`: 다른 수집기가 미리 샘플링한 폴더 → 같은 레이아웃 (구현됨) |
| `wuji_ros/rig.py` | 캘리브 로드, camera ↔ world, 세션 → 캘리브 표 `SESSION_CALIBRATION`, mocap → world(`T_world_from_mocap`: 전달된 `mocap2cam.npy`를 camera1로, 없으면 None) (구현됨) |
| `wuji_ros/calibcheck.py` | 캘리브 확인: 카메라별 테이블 평면, 카메라별 V1 손목 일치 (구현됨) |
| `wuji_ros/keypoints.py` | HaMeR 2회(depth 박스 → 삼각측량 투영 박스) + DLT 삼각측량. `third_party/hamer` wrapper (sys.path 주입, `hamer.configs.CACHE_DIR_HAMER`를 `assets/hamer`로 지정, upstream `load_hamer`와 같은 설정에 `HAMER(init_renderer=False)`. 학습 시각화용 pyrender 렌더러는 EGL이 없는 서버에서 실패한다). crop은 upstream ViTDetDataset 수식을 재구현(env에 scikit-image 없음). 오른손만: 두 pass 모두 HaMeR 손바닥 관절 6개 중 2개 이상이 어두운(검은 장갑) 픽셀 위에 있는 뷰만 쓴다 (구현됨) |
| `wuji_ros/segment.py` | 테이블 평면·영역, depth + RGB 손 후보와 1차 박스, 삼각측량 골격 주변 손 점군 `hand_points` (구현됨) |
| `wuji_ros/mano.py` | MANO forward, OpenPose-21 관절 순서 (구현됨) |
| `wuji_ros/glovefit.py` | G1, glove IMU 자세의 world 변환(`waist_in_world`, `keypoints`·`fit4` 공용) (구현됨) |
| `wuji_ros/handfit.py` | V1 카메라별(`fit`)과 4뷰 world(`fit4`) (구현됨). `fit4`는 glove 융합 IMU 자세 항(기울기 + 세션 yaw, 기본 가중치 100)과 IMU 시작점, 2D 라벨 거울 가설(손바닥/손등 뒤집힘 대응)을 쓴다. β(손 크기)는 기본으로 G1 β다(`--shape glove`, 2026-10-27). `--shape image`는 세션의 모든 프레임으로 β 하나를 영상으로 푼다. 카메라별 `fit`은 늘 G1 β다. V2(손가락 미세조정)는 하지 않는다 |
| `wuji_ros/contactfit.py` | stage `contact`(plan.md §6 18): 물체를 쥔 프레임에서 `fit4` 손을 위치만 옮겨, 물체에 닿고 관통하지 않게 하는 가장 작은 이동을 구한다. 손가락·β·회전은 그대로다. 물체 mesh의 정확한 부호 거리(점-삼각형 거리, 일반화 winding number 부호)를 몸체마다 구해 최솟값을 쓴다. 안쪽으로 감긴 몸체(속이 빈 기둥의 내벽)는 뺀다. 물체 자세가 solver 입력이므로 독립 검증은 재투영과 depth 표면이다 (구현됨, 2026-10-29) |
| `wuji_ros/objectpose.py` | mocap 물체 자세: 수집기 yaml의 `marker_offsets`(OBJ 좌표, 마커 이름으로 대응)를 기록된 마커에 Kabsch로 맞춘 `T_mocap_from_obj`, STL mesh (구현됨, 2026-10-29). M5 물체 점 제거와 M9 관통·접촉에서 다시 쓴다 |
| `wuji_ros/viz.py` | 카메라 overlay(손 mesh + MANO 골격), `--object-config/--object-mesh`이면 mocap 물체 mesh도 같은 깊이 순서로 그려 `overlay_object/`에 둔다. 촉각 맵 별도 시퀀스. 각각 H.264(yuv444p) mp4 + gif (구현됨) |
| `wuji_ros/evaluate.py` | 일치도 지표, 프레임별 라벨 |
| `wuji_ros/cli.py`, `scripts/run.py` | 분석 entrypoint, 단계별 subcommand, `--force` (구현됨) |
| `scripts/glove_bridge.py` | 브리지 entrypoint (구현됨) |

## 데이터 레이아웃

```
data/<session>/
  camera{1..4}/color/NNNNNN.png      # rgb8 → png
  camera{1..4}/depth/NNNNNN.png      # aligned_depth_to_color, uint16 mm
  camera{1..4}/intrinsics.json       # bag의 color/camera_info
  object/NNNNNN.npz                  # 물체 rigid body pose + 마커 (마커 토픽에 메시지가 없으면 NaN)
  glove/NNNNNN.npz                   # skeleton_wrist, skeleton_confidence, emf_poses, tip_poses, joint_angles, accel_wrist, gravity_wrist,
                                     # tactile / tactile_binary / tactile_residual (24,31), tactile_zone_<zone> (n,),
                                     # tactile_points (526,3) r_wrist + tactile_pressure (526,),
                                     # R_waist_from_wrist (3,3) + gravity_wrist_fused (3,) (SDK 융합 IMU 자세). 키 설명은 bagio.py
  metadata.csv                       # frame_id별 스트림 timestamp와 dt
  manifest.json                      # bag, 캘리브 이름, SDK 정보, 장갑 두께, 추출 설정
output/<session>/<stage>/...         # 단계별 결과, 재실행 시 재사용, --force로 재생성
```

`<session>`은 bag 이름에서 정한다. 출력 레이아웃은 입력 레이아웃을 따른다.

### 미리 샘플링된 폴더 (`import`)

2026-10-22 녹화부터 연구실 수집기가 bag 대신 고정 주기 폴더를 준다. 예: `data/brl_mocap_cam_hand/20261022_163051_mocap_mk4_wuji_fixed_rate/`. 15 Hz 호스트 클럭 격자에 맞춰 스트림마다 틱당 한 샘플만 남긴다.

| 폴더 | `data/<session>/` |
|---|---|
| `image_metadata.csv` (`anchor_timestamp`, `<cam>_{color,depth}_timestamp`, `mocap_timestamp`) | `metadata.csv`. `master_stamp` = camera1 color stamp, `source_anchor_stamp` = 수집기의 격자 시각 |
| `camera{1..4}/{color,depth}/NNNNNN.png` (표준 RGB png, uint16 mm) | 파일을 그대로 복사 |
| 없음 | `camera{1..4}/intrinsics.json`: `--camera-config` yaml의 `cameras[].intrinsics`. 왜곡 계수는 없다 |
| `wuji/NNNNNN.npz` (`hand_skeleton`, `emf_poses`, `tip_poses`, `hand_joint_angles`, `tactile*`, `tactile_zones_<zone>`, `tactile_point_cloud`, `imu_raw_palm_*`, `tf_*`, 스트림별 `_timestamp`) | `glove/NNNNNN.npz`. 키는 `extract`와 같다(`bagio.DIR_GLOVE_PREFIX`) |
| `wuji_info.json` (`tf_static` 포함) | `manifest.json`의 `glove_info`, `tf_static` |
| `mocap/NNNNNN.npz` (`position`, `orientation`, `marker_names`, `marker_positions`) | `object/NNNNNN.npz`. `orientation`은 xyzw다. 이 순서일 때 마커가 rigid body 좌표계에서 고정된다(중앙값 0.14 mm, wxyz면 4.7 mm). 좌표는 mocap 프레임 그대로다 |

- 확인한 규약(`20261022_163051`): `tf_frames`는 `waist → r_wrist`다. 정지 61프레임에서 융합 자세 중력과 가속도계 중력의 차이는 중앙값 0.5°, 최대 3.1°다. skeleton 단위는 m이다.
- 이 폴더들은 카메라끼리 하드웨어 동기가 없다. 카메라마다 위상이 고정되어 있어 4뷰 간 시간차가 최소 18 ms다.
- 스트림 대응은 수집기의 행(index)이다. `--align index`(기본, E012)는 행을 그대로 쓰고, `--align stamp`는 그중 camera1 color와 허용오차를 넘는 스트림을 결측으로 둔다. 어느 쪽도 다른 행과 다시 매칭하지 않는다.

## 손 박스 (plan.md M5)

캘리브된 depth, RGB, HaMeR로 잡는다.
1. `segment`: 4뷰 depth를 world로 합친다. 테이블 영역 위 1.2–35 cm에 있고 RGB가 어두운(검은 장갑·소매) 점 중 가장 큰 덩어리를 고른다. 테이블 가장자리에서 팔이 들어오는 지점에서 가장 먼 13 cm를 손으로 보고, 각 뷰에 투영해 1차 박스로 쓴다. 맨손인 왼손과 밝은 옷은 RGB 조건에서 빠진다.
2. `keypoints`: 1차 박스로 HaMeR를 돌리고 4뷰 2D를 삼각측량한다(DLT, 잔차 40 px 넘는 뷰는 3뷰가 남을 때까지 뺀다). 손바닥 관절이 검은 장갑 위에 2개 미만인 뷰는 버린다(오른손만).
   그다음 2차 박스를 잡는다. 1차 손(삼각측량, 없으면 depth 후보) 주변 20 cm의 어두운 덩어리 중, glove IMU가 주는 world 손 방향(r_wrist −Z)으로 끝에서 13 cm다. IMU가 없으면 삼각측량 투영, 그것도 없으면 1차 박스를 쓴다. 그 박스로 HaMeR를 한 번 더 돌린다. 모든 뷰가 같은 물리적 손을 본다.
   물체를 쥔 동안 장갑, 검은 손목 유닛, 팔뚝은 하나의 어두운 덩어리가 된다. 그래서 1차 박스와 삼각측량 둘레로 잡은 박스에서는 HaMeR가 손을 손목 유닛 쪽으로 밀었다(E008).
3. 손 점군(`segment.hand_points`): 삼각측량 골격의 뼈에서 3 cm 안이고 테이블 위인 depth 점이다. 팔뚝은 손목 관절 밖이라 빠진다.

처음 버전(2026-10-16 새벽)은 캘리브 없이 SAM3 텍스트 프롬프트로 박스를 잡았다. 캘리브가 도착한 뒤 사용자 요청으로 이 방식으로 바꾸고 SAM3 의존을 없앴다.

## 좌표 규약

- 변환 이름은 `T_A_from_B`다. B 좌표를 A 좌표로 옮기고, 합성은 오른쪽에서 왼쪽으로 읽는다.
- 카메라 광학 프레임: z 앞, x 오른쪽, y 아래.

### 캘리브 `calibration/20261015/` (2026-10-29 다시 저장)

| 파일 | 내용 | sha256 |
|---|---|---|
| `w2c_ext.npz` | 키 `"0".."3"` = camera1..camera4, 각 3×4 `T_cam_from_world`, m | `5e33cc19…bb20` |
| `mocap2cam.npy` | 4×4 `T_cam1_from_mocap`, m | `2665c92b…c169` |
| `t2r_ext.npz` | camera3 기준 카메라 간 변환. 사용하지 않음 | `0052ebf0…0ecf` |

- 2026-10-29 사용자 결정: 기존 캘리브 정보(`20260824_v2` 폴더 전체, `20261015`의 파일, 두 `provenance.json`)를 모두 지우고, 사용자가 올린 위 3개 파일만 캘리브로 저장했다.
  - 세 파일은 지우기 전 `20261015` 파일과 sha256이 같다.
  - 지운 파일은 git 기록(`03f93b2`)에 남아 있다. DexTouch에서 복사해 depth ICP로 유도한 `20260824_v2`의 `mocap2cam`도 함께 지웠다.
- **캘리브는 GT라서 읽기만 한다**(사용자 결정 2026-10-27, 2026-10-29). 대상은 `calibration/`의 파일과 세션 `intrinsics.json`이다. 어긋나면 수치로 보고만 하고 고치지 않는다.
  보고된 어긋남은 `docs/TODO.md`에 있다. `20261022_163051` 물체 투영에서 camera1은 맞고 camera2–4는 어긋난다.
- intrinsics는 캘리브 폴더에 없다. 세션의 `intrinsics.json`(bag의 camera_info, 또는 `import --camera-config` yaml)에서 온다.
- **mocap → world**: `T_world_from_mocap = T_world_from_cam1 · T_cam1_from_mocap`. 그래서 camera1을 거쳐서만 들어간다.
  `mocap2cam.npy`를 다른 카메라에 직접 적용하지 않는다.
- **world**: 카메라 중심 z가 −0.57 ~ −0.88 m라서 z-DOWN이다.
  **테이블 면은 world z에서 12.3° 기울어 있다**(4뷰 RANSAC). 보드가 테이블에 평평하게 놓이지 않은 상태로 캘리브된 것으로 보인다. 그래서 "위쪽"과 "중력"은 world z가 아니라 측정한 테이블 법선(`segment.Table.up`)으로 정한다. IMU 검증은 테이블이 수평이라고 가정한다.
- 물체 rigid body quaternion 순서는 xyzw로 가정한다(DexTouch 기록).
- 확인 결과는 `docs/TODO_completed.md`와 `docs/experiments/001_glove_hand_pose/experiments.md`(E008, E014)에 있다.
- 세션 → 캘리브 대응은 `wuji_ros/rig.py: SESSION_CALIBRATION`이다.
  - `1015-5` → `20261015`.
  - `20261022_163051_mocap_mk4_wuji_fixed_rate` → `20261015`. 녹화와 같이 온 `w2c_ext.npz`가 sha256까지 같다.
    테이블은 카메라별로 1.22–1.53°, −9.3…+4.5 mm 안에서 일치하고, HaMeR 삼각측량 잔차는 11.4 px다(E008).
    camera2만 체계적 투영 편차가 크다(`docs/TODO.md`).

### glove (Wuji SDK 문서)

- `hand_skeleton`: MediaPipe 21점, `r_wrist` 프레임, m. MediaPipe-21과 OpenPose-21은 인덱스 순서가 같다.
- `r_wrist` 축: X 요측, Z 근위(손목→팔꿈치), Y = Z × X (오른손은 손바닥 쪽).
- `emf_poses`: `r_hand_emf_tx` 프레임. `T_wrist_from_emftx`는 `tf_static`이다(회전 포함, 2026-10-15 실측 약 13.6°). bag의 `/tf_static`에서 읽는다.
- 촉각 taxel(`tactile_point_cloud`)은 `r_wrist` 프레임이다. `fit4`의 `T_world_from_wrist`로 world에 옮긴다. `1015-5`에서 taxel과 G1 MANO 표면의 거리 중앙값은 5.5 mm다.
- glove 중력은 SDK 융합 자세 `/wuji_glove/right/tf`(`waist → r_wrist`, waist z-up)에서 `R_waist_from_wrist^T (0,0,−1)`로 구한다(roll·pitch만 의미가 있다). `imu_raw` 가속도계는 정지할 때만 중력이다.
- palm IMU 축은 손목 축과 다르다: `T_wrist_from_palmimu`의 회전이 x축 90°다(2026-10-15 실측). yaw는 드리프트한다.
- `fit4`의 IMU 회전(2026-10-26부터, plan.md §6 14): `R_world_from_wrist = R_world_from_waist · R_waist_from_wrist`.
  waist z는 world 위쪽(테이블 법선)에 맞춘다. waist의 world yaw는 세션 상수 하나로, HaMeR 뷰마다 `R_world_from_mano · R_wrist_from_mano^T · R_waist_from_wrist^T`의 yaw를 구한 최빈값이다(`glovefit.waist_in_world`; `keypoints`는 1차 뷰로, `fit4`는 2차 뷰로 구한다).
  `20261022_163051`에서 waist z와 테이블 법선은 4° 차이였다(`fit4` 결과로 3축을 다 풀었을 때). HaMeR 뷰 133개 중 75%가 최빈값 ±30° 안에 있다.

## 시간 규약

- 모든 스트림은 `header.stamp`로 맞춘다. 수신 시각은 쓰지 않는다.
- 브리지는 glove `timestamp_us`(장치 EMF 샘플링 시각, UTC)를 `header.stamp`로 싣는다. 장갑과 카메라는 같은 PC(같은 호스트 클럭)에 연결한다.
- 추출의 마스터 클럭은 `camera1` color다. 나머지 스트림은 nearest로 매칭하고 `dt`를 기록한다. 허용오차를 넘으면 결측으로 두고 버리지 않는다.

## 외부 의존성

| 대상 | 방식 | 버전 |
|---|---|---|
| Wuji SDK | 녹화 PC의 python3.10 user site에 PyPI 설치(`requirements-bridge.txt`), 예제는 `third_party/wuji-sdk`. 휠이 glibc ≥ 2.34를 요구해 분석 서버(glibc 2.31)에서는 import 불가 | `2026.8.31` (submodule `b0e4865`) |
| HaMeR | `third_party/hamer` submodule, 수정 없이 sys.path | upstream `3a01849` |
| MANO, HaMeR checkpoint | `assets/` 심볼릭 링크 | README "자산" |

참고: 이전 DexTouch 결과는 Dyn-HaMR이 수정한 HaMeR 사본(`models/hamer.py`, `mano_head.py`, `mano_wrapper.py` 등이 upstream과 다름)으로 만든 것이다. 그래서 수치를 이 repo 결과와 그대로 비교할 수 없다.
