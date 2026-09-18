# 구조와 규약

[plan.md](../plan.md)의 설계를 코드 구조와 계약으로 옮긴 문서다. 코드가 생기면 실제 모듈에 맞춰 갱신한다.

## 기본 경로

녹화 PC에서 브리지로 bag을 기록하고, 분석 서버에서 다음 순서로 처리한다.

```
extract → calib check → keypoints(HaMeR) → segment → G1 → shape → V1 → V2 → evaluate / overlay
```

- G1: glove 관절을 MANO 관절각으로 변환한다(손목 좌표계, world 없음).
- V1: 손가락 관절을 G1 값으로 고정하고, 손 전체의 world 6D를 4뷰 RGB-D로 맞춘다.
- V2: 손가락 관절을 G1 값에서 너무 벗어나지 않게 미세조정한다.
- 대체 경로(예: 관절각을 HaMeR로 추정)는 비교 실험용이다. 기본 경로로 바꾸려면 명시적 요청이 있어야 한다.

## 모듈 (예정)

| 위치 | 역할 |
|---|---|
| `wuji_ros/paths.py` | repo 경로, `data/`·`output/` 레이아웃, `assets/`·`calibration/` 경로의 단일 출처 |
| `wuji_ros/bridge/` | 녹화 PC용 ROS2 노드. SDK frame → 메시지 필드 변환은 ROS 없이 테스트 가능한 순수 함수 |
| `wuji_ros/bagio.py` | bag → `data/<session>/` (마스터 클럭 동기) |
| `wuji_ros/rig.py` | 캘리브 로드, 투영/역투영, mocap → world |
| `wuji_ros/keypoints.py` | `third_party/hamer` wrapper (sys.path 주입, `hamer.configs.CACHE_DIR_HAMER`를 `assets/hamer`로 지정) |
| `wuji_ros/segment.py` | 손 점군 분할 |
| `wuji_ros/mano.py` | MANO forward, OpenPose-21 관절 순서 |
| `wuji_ros/glovefit.py` | G1 |
| `wuji_ros/handfit.py` | β, V1, V2 |
| `wuji_ros/viz.py` | 전체 프레임 overlay, H.264 mp4 + gif |
| `wuji_ros/evaluate.py` | 일치도 지표, 프레임별 라벨 |
| `wuji_ros/cli.py`, `scripts/run.py` | 분석 entrypoint, 단계별 subcommand, `--force` |
| `scripts/glove_bridge.py`, `launch/` | 브리지 entrypoint |

## 데이터 레이아웃

```
data/<session>/
  camera{1..4}/color/NNNNNN.png      # rgb8 → png
  camera{1..4}/depth/NNNNNN.png      # aligned_depth_to_color, uint16 mm
  camera{1..4}/intrinsics.json       # bag의 color/camera_info
  object/NNNNNN.npz                  # 물체 rigid body pose + 마커
  glove/NNNNNN.npz                   # skeleton_wrist, emf_poses, joint_angles, gravity_wrist, confidence
  metadata.csv                       # frame_id별 스트림 timestamp와 dt
  manifest.json                      # bag, 캘리브 이름, SDK 정보, 장갑 두께, 추출 설정
output/<session>/<stage>/...         # 단계별 결과, 재실행 시 재사용, --force로 재생성
```

`<session>`은 bag 이름에서 정한다. 출력 레이아웃은 입력 레이아웃을 따른다.

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

### glove (Wuji SDK 문서)

- `hand_skeleton`: MediaPipe 21점, `r_wrist` 프레임, m. MediaPipe-21과 OpenPose-21은 인덱스 순서가 같다.
- `r_wrist` 축: X 요측, Z 근위(손목→팔꿈치), Y = Z × X (오른손은 손바닥 쪽).
- `emf_poses`: `r_hand_emf_tx` 프레임. `T_wrist_from_emftx`는 `tf_static`(평행이동만)이다.
- palm IMU 축 = 손목 축. roll·pitch만 신뢰하고 yaw는 드리프트한다.

## 시간 규약

- 모든 스트림은 `header.stamp`로 맞춘다. 수신 시각은 쓰지 않는다.
- 브리지는 glove `timestamp_us`(장치 EMF 샘플링 시각, UTC)를 `header.stamp`로 싣는다. 장갑과 카메라는 같은 PC(같은 호스트 클럭)에 연결한다.
- 추출의 마스터 클럭은 `camera1` color다. 나머지 스트림은 nearest로 매칭하고 `dt`를 기록한다. 허용오차를 넘으면 결측으로 두고 버리지 않는다.

## 외부 의존성

| 대상 | 방식 | 버전 |
|---|---|---|
| Wuji SDK | 녹화 PC에 PyPI 설치, 예제는 `third_party/wuji-sdk` | `2026.8.31` (submodule `b0e4865`) |
| HaMeR | `third_party/hamer` submodule, 수정 없이 sys.path | upstream `3a01849` |
| MANO, HaMeR checkpoint | `assets/` 심볼릭 링크 | README "자산" |

참고: 이전 DexTouch 결과는 Dyn-HaMR이 수정한 HaMeR 사본(`models/hamer.py`, `mano_head.py`, `mano_wrapper.py` 등이 upstream과 다름)으로 만든 것이다. 그래서 수치를 이 repo 결과와 그대로 비교할 수 없다.
