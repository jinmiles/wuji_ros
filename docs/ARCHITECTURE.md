# 구조와 규약

[plan.md](../plan.md)의 설계를 코드 구조와 계약으로 옮긴 문서다. 코드가 생기면 실제 모듈에 맞춰 갱신한다.

## 기본 경로

녹화 PC에서 브리지로 bag을 기록하고, 분석 서버에서 다음 순서로 처리한다.

```
extract → segment(table, depth+RGB box) → keypoints(HaMeR, 삼각측량으로 재박스) → G1 → V1(fit4, 4뷰 world) → calib check → overlay / tactile
```

계획의 순서(`extract → calib check → keypoints → segment → G1 → shape → V1 → V2 → evaluate`)와 다른 점: 손 박스가 HaMeR 전에 필요해 `segment`가 앞에 오고, β는 아직 G1에서 푼다. V2와 evaluate는 아직 없다.

```
```

- G1: glove 관절을 MANO 관절각으로 변환한다(손목 좌표계, world 없음).
- V1: 손가락 관절을 G1 값으로 고정하고, 손 전체의 world 6D를 4뷰 RGB-D로 맞춘다.
- V2: 손가락 관절을 G1 값에서 너무 벗어나지 않게 미세조정한다.
- 대체 경로(예: 관절각을 HaMeR로 추정)는 비교 실험용이다. 기본 경로로 바꾸려면 명시적 요청이 있어야 한다.

## 모듈

| 위치 | 역할 |
|---|---|
| `wuji_ros/paths.py` | repo 경로, `data/`·`output/` 레이아웃, `assets/`·`calibration/` 경로의 단일 출처, logger (구현됨) |
| `wuji_ros/bridge/convert.py` | SDK frame → 메시지 필드 값. ROS·SDK import 없는 순수 함수 (구현됨, `tests/test_bridge_convert.py`) |
| `wuji_ros/bridge/node.py` | 녹화 PC용 ROS2 노드 `wuji_glove_bridge` (구현됨) |
| `wuji_ros/bagio.py` | bag → `data/<session>/` (마스터 클럭 동기), `Session` reader (구현됨). bag을 SQLite로 직접 읽어 `metadata.yaml`이 없어도 된다 |
| `wuji_ros/rig.py` | 캘리브 로드, camera ↔ world, 세션 → 캘리브 표 `SESSION_CALIBRATION` (구현됨. mocap → world는 아직 없음) |
| `wuji_ros/calibcheck.py` | 캘리브 확인: 카메라별 테이블 평면, 카메라별 V1 손목 일치 (구현됨) |
| `wuji_ros/keypoints.py` | HaMeR 2회(depth 박스 → 삼각측량 투영 박스) + DLT 삼각측량. `third_party/hamer` wrapper (sys.path 주입, `hamer.configs.CACHE_DIR_HAMER`를 `assets/hamer`로 지정). crop은 upstream ViTDetDataset 수식을 재구현(env에 scikit-image 없음) (구현됨) |
| `wuji_ros/segment.py` | 테이블 평면·영역, depth + RGB 손 후보와 1차 박스, 삼각측량 골격 주변 손 점군 `hand_points` (구현됨) |
| `wuji_ros/mano.py` | MANO forward, OpenPose-21 관절 순서 (구현됨) |
| `wuji_ros/glovefit.py` | G1 (구현됨) |
| `wuji_ros/handfit.py` | V1 카메라별(`fit`)과 4뷰 world(`fit4`) (구현됨). `fit4`는 glove 융합 IMU 기울기 항(기본 가중치 100)과 2D 라벨 거울 가설(손바닥/손등 뒤집힘 대응)을 쓴다. β는 현재 G1에서 푼다. V2는 아직 없음 |
| `wuji_ros/viz.py` | 카메라 overlay(손 mesh + MANO 골격만), 촉각 맵 별도 시퀀스, 각각 H.264(yuv444p) mp4 + gif (구현됨) |
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

## 손 박스 (plan.md M5)

캘리브된 depth, RGB, HaMeR로 잡는다.
1. `segment`: 4뷰 depth를 world로 합친다. 테이블 영역 위 1.2–35 cm에 있고 RGB가 어두운(검은 장갑·소매) 점 중 가장 큰 덩어리를 고른다. 테이블 가장자리에서 팔이 들어오는 지점에서 가장 먼 13 cm를 손으로 보고, 각 뷰에 투영해 1차 박스로 쓴다. 맨손인 왼손과 밝은 옷은 RGB 조건에서 빠진다.
2. `keypoints`: 1차 박스로 HaMeR를 돌리고 4뷰 2D를 삼각측량한다(DLT, 잔차 40 px 넘는 뷰는 3뷰가 남을 때까지 뺀다). 그 3D 관절을 모든 뷰에 투영해 박스를 다시 잡고 HaMeR를 한 번 더 돌린다. 모든 뷰가 같은 물리적 손을 본다.
3. 손 점군(`segment.hand_points`): 삼각측량 골격의 뼈에서 3 cm 안이고 테이블 위인 depth 점이다. 팔뚝은 손목 관절 밖이라 빠진다.

처음 버전(2026-10-16 새벽)은 캘리브 없이 SAM3 텍스트 프롬프트로 박스를 잡았다. 캘리브가 도착한 뒤 사용자 요청으로 이 방식으로 바꾸고 SAM3 의존을 없앴다.

## 좌표 규약

- 변환 이름은 `T_A_from_B`다. B 좌표를 A 좌표로 옮기고, 합성은 오른쪽에서 왼쪽으로 읽는다.
- 카메라 광학 프레임: z 앞, x 오른쪽, y 아래.

### 캘리브 `calibration/20260824_v2/` (2026-09-18 DexTouch에서 복사, sha256 동일)

| 파일 | 내용 | 출처 |
|---|---|---|
| `w2c_ext.npz` | 키 `"0".."3"` = camera1..camera4, 각 3×4 `T_cam_from_world` | 연구실 전달 |
| `mocap2cam.npy` | 4×4 `T_cam1_from_mocap`. **camera1에만 유효** | 연구실이 전달하지 않아 DexTouch가 depth ICP로 유도한 임시값 |
| `t2r_ext.npz` | camera3 기준 카메라 간 변환. 사용하지 않음 | 연구실 전달 |
| `provenance.json` | `mocap2cam.npy` 유도 방법과 불확실성 | DexTouch |

- **world**: DexTouch 문서 기준 aruco 보드 좌표계, **z-DOWN**, 보드(테이블) 평면 z ≈ 0.
  이 해석은 새 녹화의 샘플로 M4에서 다시 확인하기 전까지 가정이다.
- **mocap → world**: `T_world_from_mocap = T_world_from_cam1 · T_cam1_from_mocap`.
  `mocap2cam.npy`를 다른 카메라에 직접 적용하지 않는다.
- **불확실성**:
  - `mocap2cam.npy`는 세션 간 0.99° / 2.6 mm 차이가 났다(DexTouch `docs/calibration.md`).
  - `w2c_ext`와 `t2r_ext`는 같은 카메라 쌍에서 0.22–1.96° / 7–35 mm 어긋난다.
- **새 녹화에서의 유효성**: 이 캘리브는 2026-08-24 리그 상태다. 카메라가 움직였으면 틀린다.
  DexTouch 기록에 카메라가 녹화 사이에 움직인 사례가 있다. 그래서 M4에서 물체 마커 투영으로 확인한다.
- 물체 rigid body quaternion 순서는 xyzw로 가정한다(DexTouch 기록). 새 bag의 메시지 타입으로 확인한다.

### 캘리브 `calibration/20261015/` (2026-10-16 전달)

- 형식은 `20260824_v2`와 같다(`w2c_ext.npz` 3×4 `T_cam_from_world`). `mocap2cam`은 없다.
- **world**: 카메라 중심 z가 −0.57 ~ −0.88 m라서 z-DOWN이다.
  **테이블 면은 world z에서 12.3° 기울어 있다**(4뷰 RANSAC). 보드가 테이블에 평평하게 놓이지 않은 상태로 캘리브된 것으로 보인다. 그래서 "위쪽"과 "중력"은 world z가 아니라 측정한 테이블 법선(`segment.Table.up`)으로 정한다. IMU 검증은 테이블이 수평이라고 가정한다.
- 확인 결과는 `calibration/20261015/provenance.json`과 `docs/TODO_completed.md`에 있다.
- 세션 → 캘리브 대응은 `wuji_ros/rig.py: SESSION_CALIBRATION`이다(`1015-5` → `20261015`).

### glove (Wuji SDK 문서)

- `hand_skeleton`: MediaPipe 21점, `r_wrist` 프레임, m. MediaPipe-21과 OpenPose-21은 인덱스 순서가 같다.
- `r_wrist` 축: X 요측, Z 근위(손목→팔꿈치), Y = Z × X (오른손은 손바닥 쪽).
- `emf_poses`: `r_hand_emf_tx` 프레임. `T_wrist_from_emftx`는 `tf_static`이다(회전 포함, 2026-10-15 실측 약 13.6°). bag의 `/tf_static`에서 읽는다.
- 촉각 taxel(`tactile_point_cloud`)은 `r_wrist` 프레임이다. `fit4`의 `T_world_from_wrist`로 world에 옮긴다. `1015-5`에서 taxel과 G1 MANO 표면의 거리 중앙값은 5.5 mm다.
- glove 중력은 SDK 융합 자세 `/wuji_glove/right/tf`(`waist → r_wrist`, waist z-up)에서 `R_waist_from_wrist^T (0,0,−1)`로 구한다(roll·pitch만 의미가 있다). `imu_raw` 가속도계는 정지할 때만 중력이다.
- palm IMU 축은 손목 축과 다르다: `T_wrist_from_palmimu`의 회전이 x축 90°다(2026-10-15 실측). roll·pitch만 신뢰하고 yaw는 드리프트한다.

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
