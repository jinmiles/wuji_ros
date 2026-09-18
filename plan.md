# Wuji Glove 기반 손 pose 추정 — 계획

> 목표: Wuji Glove + 4뷰 RealSense RGB-D + OptiTrack(물체) 녹화에서, 매 프레임 **world 좌표의 MANO 손 pose**를 추정한다.
> 역할 분담: **손가락 관절은 glove**, **손 전체의 world 6D(손목)는 4뷰 RGB-D 정합**으로 얻는다. 손에는 마커를 붙이지 않는다.
>
> **wuji_ros가 메인 프로젝트다**(2026-09-18 결정). 녹화 브리지부터 추정·평가까지 전부 이 repo에 구현한다.
> DexTouch, mocap_ros_py는 **수정하지도 import하지도 않는다.** 아래 §0에 그쪽에서 읽어 둔 사실만 배경 근거로 남긴다.
> 외부 upstream은 `third_party/` submodule로만 들이고 수정하지 않는다.
>
> 이 문서는 계획이다. §6의 미결 항목이 정해지기 전까지 해당 마일스톤의 세부는 바뀔 수 있다.

---

## 0. 확인된 사실 (근거)

### Wuji Glove / wuji-sdk
- 출처: `third_party/wuji-sdk` (submodule, `b0e4865`), Wuji Docs Center의 hand-tracking / coordinate-frames / time-sync / imu / offline-pipeline 문서.
- `third_party/wuji-sdk`에는 **예제와 문서만** 있다. SDK 본체는 PyPI 휠이다.
  PyPI의 `2026.9.14`는 yanked, 최신 유효 버전은 `2026.8.31`이고 submodule 커밋과 일치한다 → **`wuji-sdk==2026.8.31`로 고정**.
  `requires_python >=3.10`, 휠은 `cp310`~`cp314`, `manylinux_2_34` (Ubuntu 22.04 + ROS2 Humble의 python3.10에서 사용 가능).
- 측정량은 **EMF**: 손등 송신 코일(`r_hand_emf_tx`) 기준 손가락 끝 수신 코일 5개의 6D pose (`emf_poses`, 120 Hz, confidence < 0.9는 불신).
  `hand_skeleton` / `tip_poses` / `hand_joint_angles`는 이 EMF를 hand URDF로 IK 한 **파생값**이다.
- `hand_skeleton`: MediaPipe 21점, `frame_id = r_wrist`, 단위 m, EMF 소스 timestamp를 이어받는다.
- `r_wrist` 축: X = 요측(radial), Z = 근위(손목→팔꿈치), Y = 오른손 법칙 → **오른손은 Y+가 손바닥 쪽**.
- `tf_static`: `r_wrist → r_hand_emf_tx`, `r_wrist → r_palm_imu_link` (**평행이동만**, 1 Hz). 즉 palm IMU 축 = 손목 축.
- **절대 위치 없음.** palm IMU 융합 자세는 roll·pitch가 중력 기준으로 장기 안정하다.
  반면 yaw는 자력계가 없어 스트림 시작 시 0에서 출발해 드리프트한다.
  → 손목 6D 소스로 쓰지 않고 **중력 방향(기울기) 검증**에만 쓴다.
- `timestamp_us`: 장치가 **EMF 샘플링 완료 시점**에 찍고, connect 직후 자동 time sync 이후 **UTC µs**. sync 전에는 장치 uptime이다.
  30 s마다 재동기, 단조 증가 보장.
- offline pipeline(`WujiGlove.offline_pipeline(sn, hand_side, urdf_path)`)은 같은 `emf_poses` + 같은 URDF에서 **실시간과 동일한** skeleton / angles / tips를 낸다.
  → raw `emf_poses`를 녹화해 두면 캘리브레이션된 URDF로 다시 계산할 수 있다.
- IK는 SDK 사용자(user)의 URDF를 쓴다. default user는 항상 built-in URDF이고, 피험자별 캘리브레이션은 **named user**를 만든 뒤 `5.calibration.py`로 한다.

### 설계를 정한 배경 측정 (이전 프로젝트 데이터를 읽기만 함)
- **손바닥-위 파지가 흔하다**: 이전 4뷰 녹화의 손 fit에서 손등 법선이 world 위쪽과 120° 넘게 벌어진 프레임은 한 세션에서 77%, 다른 세션에서 56%였다.
  나머지 세션은 0~12%였다. 한 세션은 overlay로 직접 확인했다.
- **손등 마커는 추적이 잘 안 됐다**: mocap_ros_py glove_exp1에서 손등 허브 rigid body가 잡힌 프레임은 왼손 40.5%, 오른손 13.2%였다.
  OptiTrack에 낮은 각도 카메라가 없다(사용자 확인). → 손 마커를 쓰지 않는다.
- **장갑을 끼면 HaMeR가 나빠진다**: 이전 녹화에서 검은 mocap 장갑을 낀 세션은 HaMeR 재투영 10.4 / 12.8 px, 맨손은 7.6–8.8 px였다.
  confidence도 0.80–0.81로 가장 낮았다.
- **bag 녹화의 교훈**(mocap_ros_py `ROSBAG_OUTPUT.md`):
  - 동기화는 반드시 `header.stamp` 기준으로 한다. RealSense는 촬영 시각을 찍고, recv − stamp ≈ 50 ms다.
  - 기록 중 이미지가 드롭되어 뷰당 22–26 fps가 나온다(목표 30).
  - bag의 `/tf_static`은 camera3/4를 camera1에 잇지 않는다. 멀티뷰 extrinsic은 별도 캘리브 파일로 받아야 한다.
  - glove IMU와 OptiTrack의 각속도 상호상관 오프셋은 +9 ms였다.
- **RealSense depth 함정**(DexTouch 문서): 흰색·무늬 없는 물체는 모든 면이 10–15 mm 바깥으로 부풀어 보인다.
  depth 잔차만으로 물체 크기·배치나 손끝 정확도를 결론내지 않는다.
- 연구실 OptiTrack ROS 노드는 개별 마커 발행이 가능하다(사용자 확인). 현재 계획에서는 쓰지 않는다.

### HOGraspNet (Cho et al., ECCV 2024) — 본문 + supplementary §3
- 장비: Azure Kinect **RGB-D 4대**, IR mocap 8대는 **물체만** 추적했다. **손에는 마커가 없다.**
- 손 주석:
  1. 뷰별 MediaPipe 2.5D → 삼각측량(재투영 오차로 이상치 뷰 제거).
  2. depth로 관절 가시성을 구한다.
  3. 마스크는 DeepLabv3로 만든다.
- MANO fit 손실: 2D(가시성 가중), 물체 마커, 마스크 L1, depth L1, pose·shape·시간 정규화, 관통 + 접촉(τ = 8 mm).
- **단계적 fit**: (1) 손 전체 global → (2) 손목에서 바깥쪽 관절 → (3) 손 전체 + 물체.
- 품질: 어느 뷰든 마스크 IoU < 0.6이면 제외하고 육안으로 검수한다. **mm 정확도 보고 없음, fit 코드 비공개.**

---

## 1. 전체 구조

```
[녹화 PC — ROS2 Humble, python3.10, 장갑·카메라 같은 PC]
 Wuji Glove ──wuji-sdk──▶ wuji_ros 브리지 노드 ──▶ /wuji_glove/right/*  (header.stamp = 장치 timestamp_us)
 OptiTrack  ──NatNet──▶ /RigidBody_<object>
 RealSense ×4 ────────▶ /cameraN/cameraN/...
            └──────────▶ ros2 bag record

[분석 서버 — wuji_ros 분석 파이프라인]
 bag ─extract─▶ data/<session>/ (4뷰 color·aligned depth·K, 물체 pose, glove)
     ─calib check─▶ 캘리브 일관성 보고
     ─keypoints─▶ 4뷰 HaMeR 2D          ─segment─▶ 손 점군
     ─G1─▶ θ_glove (손목 좌표계, world 없음)
     ─shape─▶ β (캘리브 클립)
     ─V1─▶ 손 전체 world 6D   ─V2─▶ θ 미세조정
     ─evaluate / overlay─▶ output/<session>/
```

- HOGraspNet 3단계와의 대응: (1) global → **V1**, (2) 손목에서 바깥쪽 관절 → **glove가 대신 공급(G1)**, (3) 손 전체 → **V2**.
- 처음 버전에서는 물체를 두 가지에만 쓴다: 손 점군에서 물체 점을 빼는 것, 그리고 평가.
  물체 관통·접촉 제약은 V1·V2가 선 뒤의 별도 마일스톤(M9)이다.

### 예정 모듈 (M0에서 `docs/ARCHITECTURE.md`로 확정)

| 위치 | 역할 |
|---|---|
| `wuji_ros/paths.py` | repo 경로, `data/`·`output/` 레이아웃, 외부 자산(MANO, HaMeR checkpoint) 경로의 **단일 출처** |
| `wuji_ros/bridge/` | 녹화 PC용 ROS2 노드. SDK frame → 메시지 필드 변환은 ROS 없이 테스트 가능한 순수 함수로 둔다 |
| `wuji_ros/bagio.py` | bag → 세션 데이터셋 (마스터 클럭 동기) |
| `wuji_ros/rig.py` | 캘리브 로드, 투영/역투영, mocap → world |
| `wuji_ros/keypoints.py` | `third_party/hamer` wrapper (sys.path 주입) |
| `wuji_ros/segment.py` | 손 점군 분할 |
| `wuji_ros/mano.py` | MANO forward (OpenPose-21 순서 = MediaPipe-21 순서), fingertip 정점 |
| `wuji_ros/glovefit.py` | G1 |
| `wuji_ros/handfit.py` | β, V1, V2 |
| `wuji_ros/viz.py` | 전체 프레임 overlay, mp4(H.264)/gif |
| `wuji_ros/evaluate.py` | 일치도 지표, 라벨 |
| `wuji_ros/cli.py`, `scripts/run.py` | 분석 entrypoint 하나, 단계별 subcommand, `--force`로 재생성 |
| `scripts/glove_bridge.py` | 브리지 entrypoint (녹화 PC) |

---

## 2. 좌표·시간 계약

### 좌표계
- **world** = 캘리브 파일이 정의하는 world. 축 방향(z-up/down)과 테이블 평면은 받은 캘리브 파일과 실제 샘플로 확인한 뒤 `docs/ARCHITECTURE.md`에 적는다. 추측하지 않는다.
- 카메라: `K`는 bag의 `color/camera_info`에서 읽고, `T_cam_from_world`는 캘리브 파일에서 읽는다. depth는 `aligned_depth_to_color`(uint16 mm)만 쓴다.
- 물체: `T_world_from_object(t) = T_world_from_mocap · T_mocap_from_object(t)`.

### 손 변환 체인 (각 변환은 정확히 한 번만 적용)

```
G1 (프레임별, world 없음):  kp_wrist(t)  ≈  J_MANO(θ_glove(t), β; T_wrist_from_mano(t))
V1 (프레임별, 6-DoF):       T_world_from_wrist(t)  = argmin 4뷰 증거 | 손 모양 = MANO(θ_glove(t), β)
최종:                       joints/vertices_world(t) = T_world_from_wrist(t) · T_wrist_from_mano(t) · MANO(θ(t), β)
EMF:                        p_wrist(t) = T_wrist_from_emftx · p_emftx(t)     # tf_static, 평행이동만
```

- 인덱스 대응: MediaPipe-21 = OpenPose-21 순서라 인덱스 매핑은 항등이다.
  다만 **관절 위치의 의미는 다르다**(Wuji URDF 관절 중심 ≠ MANO 관절 회전 중심). G1 잔차로 측정하고, 같다고 가정하지 않는다.
- EMF 수신 코일은 손가락 끝 **등쪽(손톱 쪽)**에 있어 MANO fingertip 정점과 다른 점이다.
  EMF를 fit에 쓸 때는 대응 MANO 정점 ID를 한 번 정해 고정한다(nearest 재매칭 금지).
- IMU 기울기 검증: palm IMU가 주는 손목 좌표계 중력 방향 `g_wrist(t)`와 V1의 `R_world_from_wrist(t)ᵀ · g_world` 사이 각도를 본다. yaw는 비교하지 않는다.
  `g_world`는 테이블 법선으로 가정하고, 테이블 수평은 E001에서 정지 IMU로 확인한다.

### 시간
- 브리지는 `header.stamp = timestamp_us`(장치 EMF 샘플링 시각, UTC)로 발행한다. 수신 시각을 쓰지 않는다.
- 장갑과 카메라는 **같은 PC**에 연결한다(2026-09-18 결정). glove stamp와 카메라 stamp가 같은 호스트 클럭을 쓴다.
- sync 전 uptime timestamp가 섞이지 않도록 시작 시 `|header.stamp − host now|`를 확인한다. 1 s를 넘으면 발행하지 않고 에러로 끝낸다.
- 추출 시 마스터 클럭은 `camera1` color `header.stamp`다. 나머지 카메라, depth, 물체 pose, glove는 nearest로 매칭하고 스트림별 `dt`를 기록한다.
  허용오차를 넘으면 결측으로 표시하고 버리지 않는다.
- 검증(E001): 손에 마커가 없으므로 **추적되는 물체를 장갑 손으로 단단히 쥐고 회전**시키는 클립을 쓴다.
  palm IMU 각속도 크기와 물체 rigid body 각속도 크기의 상호상관으로 glove↔mocap 오프셋을 잰다.

---

## 3. 마일스톤

### M0 — 문서·환경·의존성
- `README.md`, `docs/ARCHITECTURE.md`, `docs/TODO.md`, `docs/experiments/README.md`, `docs/experiments/001_glove_hand_pose/experiments.md`.
- `third_party/hamer` submodule(upstream `geopavlakos/hamer`, `3a01849`). 수정하지 않고 wrapper에서 sys.path로 쓴다.
- 분석 env `environment.yml`, 자산 심볼릭 링크 `assets/`, 캘리브 `calibration/20260824_v2/` (§6 10–12).
- 녹화 PC: ROS2 Humble python3.10에 `wuji-sdk==2026.8.31` (승인됨). 피험자 named user + `5.calibration.py`로 URDF 확보.

### M1 — 브리지 노드
| 토픽 | 타입 | 내용 |
|---|---|---|
| `/wuji_glove/right/emf_poses` | `sensor_msgs/PointCloud2` | 5점, 필드 `x,y,z,qx,qy,qz,qw,confidence`, `frame_id=r_hand_emf_tx` — **raw, 재계산의 원천** |
| `/wuji_glove/right/hand_skeleton` | `sensor_msgs/PointCloud2` | 21점(MediaPipe 순서), 필드 `x,y,z,confidence`, `frame_id=r_wrist` |
| `/wuji_glove/right/hand_joint_angles` | `sensor_msgs/JointState` | 21 DoF, rad |
| `/wuji_glove/right/imu_raw/palm` | `sensor_msgs/Imu` | 원시 각속도·가속도 (시간 동기 검증 + offline pipeline 입력) |
| `/wuji_glove/right/imu_data/palm` | `sensor_msgs/Imu` | 융합 자세 (기울기 검증용, yaw 불사용) |
| `/wuji_glove/right/info` | `std_msgs/String` (JSON, latched) | 세션 메타: SN, 펌웨어, SDK 버전, SDK user id, URDF 출처·경로·sha256, time sync 결과, 필드·관절 이름, `tf_static` 값 |
| `/tf_static` | `tf2_msgs/TFMessage` | `r_wrist → r_hand_emf_tx`, `r_wrist → r_palm_imu_link` |

- 표준 메시지만 쓴다(6절 결정). IMU 기반 `tf`(`waist → wrist`)는 yaw 드리프트 때문에 발행하지 않는다.
- 녹화를 멈추는 조건:
  - device stamp가 호스트 시계와 `--max-clock-skew`(기본 1 s) 넘게 다를 때. time sync 전에 찍힌 uptime stamp를 막는다.
  - `emf_poses`가 `--stall-timeout`(기본 2 s) 동안 오지 않을 때.
  - 관절·손가락 개수가 계약과 다를 때.
- SDK user는 id만 기록한다. 표시 이름은 피험자 이름일 수 있어서다.
- 뷰어는 따로 만들지 않고 `ros2 topic hz` / `ros2 topic echo`로 확인한다.
- **wuji-sdk는 녹화 PC에서만 돈다.** 휠이 `manylinux_2_34`(glibc ≥ 2.34)인데, 분석 서버는 glibc 2.31이라 import가 실패한다(2026-09-18 확인).
  raw `emf_poses`로 skeleton을 다시 계산하는 offline pipeline도 녹화 PC에서 돌려야 한다.
- 구현(2026-09-18): `wuji_ros/bridge/convert.py`(순수 변환), `wuji_ros/bridge/node.py`, `scripts/glove_bridge.py`, `requirements-bridge.txt`.
  변환 계약 테스트는 `tests/test_bridge_convert.py`. 실제 장갑·ROS 환경에서의 실행은 녹화 PC에서 확인해야 한다.
- 수용 기준: bag에서 `hand_skeleton` ≈ 120 Hz, stamp 단조 증가, `|stamp − recv|` 분포 기록, 단위 m 확인.

### M2 — 실험 세팅·녹화
- 손에는 마커를 붙이지 않는다. OptiTrack은 물체만 추적한다.
- 장갑 두께 `t_glove`를 손등·손바닥·손가락 마디에서 캘리퍼로 잰다(V1 surface 항의 오프셋).
- 금속 근처 EMF 간섭 가능성 때문에, 녹화 전 정지 손으로 `emf_poses` confidence ≥ 0.9를 확인한다.
- `ros2 bag record`: 장갑 토픽, 물체 rigid body(+마커), 카메라 color / aligned_depth / camera_info, `/tf_static`. 카메라 QoS와 드롭률도 기록한다.
- 녹화 클립:
  - (a) **β 캘리브 클립**: 물체 없이 손을 펴고 천천히 회전.
  - (b) **동기 클립**: 추적되는 물체를 쥐고 여러 축으로 회전(E001). 시작 전 손을 테이블에 올려 정지 IMU로 테이블 기울기 확인.
  - (c) **파지 클립**: 파지 과제 + **손바닥이 위를 향하는 파지**.

### M3 — bag → 세션 데이터셋 (`bagio`)
- `data/<session>/camera{1..4}/{color,depth}/NNNNNN.png`, `camera{1..4}/intrinsics.json`
- `data/<session>/object/NNNNNN.npz`: 물체 rigid body pose와 마커.
- `data/<session>/glove/NNNNNN.npz`: `skeleton_wrist`(21,3 m, `r_wrist`), `skeleton_confidence`(21), `emf_poses`(5,7), `emf_confidence`(5), `joint_angles`(5,5 rad), `gravity_wrist`(3).
- `data/<session>/metadata.csv`: frame_id별 스트림 timestamp와 `dt`.
- `data/<session>/manifest.json`: bag 경로, 캘리브 이름, SDK 정보, glove 두께, 추출 설정.
- 프레임 선택은 마스터 카메라 프레임 전체가 기본이고, `--stride`로 줄인다. 연속 프레임이 있으므로 시간 항은 나중에 단일 변경 실험으로 넣을 수 있다.
- 수용 기준: 실제 bag 샘플로 키·shape·frame_id 정렬·단위·스트림별 `dt` 분포를 확인한다. 드롭된 카메라 프레임은 결측으로 표시한다.

### M4 — 캘리브 확인 (`rig`)
- 물체 마커를 4뷰 color에 투영한 overlay와, 테이블 평면(depth 대비 캘리브 world)을 확인한다.
- 이상이 있으면 이후 단계로 가지 않고 보고한다. 보정값이 필요하면 캘리브 계층에 출처와 함께 둔다(물체나 손 상수에 흡수하지 않는다).

### M5 — HaMeR 2D + 손 점군 (`keypoints`, `segment`)
- HaMeR에 넣을 손 박스:
  - 1차: 테이블 위 작업공간 안에서 배경·물체로 설명되지 않는 depth 점을 손 후보로 보고, 각 뷰에 투영한 박스를 쓴다.
  - 연속 프레임에서는 이전 프레임 V1 결과를 투영한 박스로 보강할 수 있다.
- 손 점군은 HaMeR 21점 convex hull(여유 margin) 안의 depth 점에서 테이블 평면과 물체 주변 점을 뺀 것이다. 팔뚝은 hull 밖이라 빠진다.
- 뷰별 HaMeR confidence와 관절별 가시성 가중을 저장한다.

### M6 — G1: glove 관절 → MANO 관절각 (손목 좌표계)
- 변수: 프레임별 θ(45), `T_wrist_from_mano`(6). β는 M7 shape에서 고정.
- 손실:
  - 손바닥 강체 점(MediaPipe 0 wrist, 1 thumb CMC, 5·9·13·17 MCP)의 위치.
  - 손가락 **뼈 방향**의 `1 − cos`. URDF와 MANO의 뼈 길이 불일치가 자세를 왜곡하지 않게 한다.
  - glove confidence 가중, 관절 한계.
- world·카메라를 쓰지 않아 싸게 돌고, vision과 독립이다. 손바닥 강체 점이 `r_wrist`에서 실제로 고정인지 먼저 확인한다.

### M7 — shape β, V1, V2 (`handfit`)
- **shape**: 클립 (a)에서 θ = θ_glove로 두고 β와 프레임별 6D를 함께 푼다. β에는 prior를 둔다. 피험자별로 한 번 정하고 고정한다.
- **V1 (6-DoF)**: 손 모양을 `MANO(θ_glove, β)`로 고정하고 MANO `global_orient`(3), `transl`(3)만 푼다.
  - 재투영 손실: 4뷰 HaMeR 2D, 관절 가시성 가중, Huber 10 px. 2D 소스는 HaMeR만 쓴다(2026-09-18 결정).
  - 표면 손실: 손 점군 → 최근접 MANO 정점, 한 방향, Huber 10 mm.
    MANO 정점은 법선 방향으로 `t_glove`만큼 부풀려 쓴다(점군은 장갑 표면, MANO는 피부).
  - 초기 후보는 4뷰 HaMeR 회전 평균, 뷰별 HaMeR 회전, 그리고 각각을 손 길이축으로 180° 뒤집은 것이다. 이동은 점군 중심에서 잡는다.
    **데이터 손실이 가장 낮은 해**를 고른다.
  - IMU 기울기는 선택에 쓰지 않는다. 검증 채널의 독립성을 지키기 위해서다. 크게 어긋나면 뒤집힘 의심 라벨을 단다.
- **V2**: V1 결과에서 θ를 풀되 `‖θ − θ_glove‖` prior를 둔다.
- 프레임별 라벨:
  - 뷰별 마스크 IoU: MANO를 투영해 래스터화한 마스크 vs 손 픽셀. < 0.6이면 저신뢰로 표시하되 버리지 않는다.
  - IMU 기울기 오차, glove confidence, HaMeR confidence, 유효 뷰 수.
- 출력은 `output/<session>/{glove_g1,v1,v2}/NNNNNN.npz`다. 기존 출력이 있으면 재사용하고 `--force`로 재생성한다.

### M8 — 평가·시각화
- **GT가 없으므로 정확도가 아니라 독립 채널 간 일치도만 말한다**(손끝 GT 클립은 불가, 2026-09-18). 독립 채널은 glove(EMF), 4뷰 RGB-D, palm IMU 기울기다.
- overlay는 전체 카메라 프레임에 그리고(crop 금지), 파일명 라벨은 이미지 아래 띠에 둔다. per-frame 시퀀스마다 H.264 mp4와 looping gif를 함께 낸다.

| 실험 | 한 가지 변경 | 측정 |
|---|---|---|
| E001 | 시간 동기 검증 | IMU vs 물체 RB 각속도 상호상관 오프셋, 정지 IMU로 본 테이블 기울기 |
| E002 | G1 (glove → MANO) | 손목 좌표계 잔차: 손바닥 점 위치, 손가락 뼈 방향 각도, 손끝 위치(mm) |
| E003 | 관절각 소스: HaMeR θ → glove θ (V1 동일) | 뷰별 IoU, HaMeR 2D 재투영, IMU 기울기 오차, 저신뢰 프레임 비율. **손바닥-위 프레임을 따로 집계** |
| E004 | V2 (θ 미세조정 + glove prior) | E003 glove 조건 대비 위 지표 변화, glove θ에서 벗어난 양 |

- E003의 기준 조건(HaMeR θ)은 같은 파이프라인에서 θ만 4뷰 HaMeR 평균으로 바꾼 것이다. glove의 효과를 한 가지 변경으로 본다.
- depth 잔차는 손끝 정확도의 근거로 쓰지 않는다.

### M9 — (V1·V2 이후) 물체 관통·접촉 제약
- 새 녹화 물체의 CAD와 마커 오프셋이 필요하다. 장갑 두께 때문에 관통 허용치를 다시 정해야 한다. 범위와 우선순위는 V1·V2 결과를 보고 정한다.

---

## 4. 완료 조건 (제안 — 결과를 보기 전에 확정한다)

1. 브리지: bag 기준 `hand_skeleton` ≥ 110 Hz, stamp가 UTC이고 단조 증가.
2. E001: glove↔mocap 시간 오프셋 |Δt| ≤ 8.3 ms (EMF 한 주기, 120 Hz).
3. E002: G1 손끝 잔차 중앙값 ≤ 8 mm.
   근거: 접촉 판정 임계 8 mm(HOGraspNet τ, 이전 파이프라인의 contact threshold). MANO가 glove 손끝을 이보다 못 따라가면 glove 손으로 접촉을 판정할 수 없다.
4. E003·E004: 파지 클립의 모든 프레임(결측 제외)에서 V1·V2 결과와 프레임별 라벨이 산출되고, overlay mp4/gif가 있다.
   HaMeR θ 대비 비교가 손바닥-위 프레임을 따로 나눠 보고된다.

조건 3이 실패하면 기준을 내리지 않는다. 원인(G1 모델 차이, URDF, β)을 분리해 다음 가설을 세운다.

---

## 5. 위험·한계

- **정확도의 기준이 없다**: 손 마커도 손끝 GT도 없다. IMU는 기울기(2 DoF)만 검증한다. 보고는 채널 간 일치도까지만 한다.
- **장갑 외형**: HaMeR는 맨손으로 학습됐다. 장갑을 끼면 2D keypoint와 분할이 나빠진다(이전 녹화에서 측정됨).
  V1은 6 DoF만 풀어 덜 민감할 것으로 예상하지만, E003 전까지는 가설이다.
- **장갑 두께**: 점군은 장갑 표면이다. `t_glove` 오프셋이 부정확하면 V1 위치가 표면 법선 방향으로 치우친다.
- **URDF 의존**: 캘리브레이션하지 않은 user는 built-in URDF를 쓴다. raw EMF를 녹화해 두면 재계산할 수 있다.
- **손등 송신기와 손 뼈의 상대 운동**: IK로 추정한 손목·MCP는 장갑이 밀리면 실제 해부학 위치와 어긋난다.
- **EMF 자체 정확도**: 문서에 수치가 없고, 이 계획 안에서는 잴 수단이 없다.
- **재구현 범위**: 이전 프로젝트가 겪은 함정을 다시 밟을 수 있다.
  - RealSense depth 팽창.
  - 캘리브 체인 오프셋을 물체·손 상수에 흡수시키는 실수.
  - bag TF만으로 멀티뷰 extrinsic 구성.
  - 녹화 수신 시각 기준 동기.
  §0의 교훈을 수용 기준에 넣어 막는다.

---

## 6. 결정 사항과 미결 항목

결정됨 (2026-09-18):
1. 손목 pose 소스 = 4뷰 RGB-D 정합. 손 마커 없음(낮은 각도 OptiTrack 카메라 없음, 손바닥-위 파지 필요).
2. glove = 손가락 관절(G1), 4뷰 RGB-D = 손 전체 6D(V1)와 미세조정(V2).
3. 데이터셋은 bag에서 wuji_ros가 직접 추출한다. 장갑과 카메라는 같은 PC에 연결한다.
4. 오른손만.
5. 손끝 GT 클립은 불가. mm 정확도는 주장하지 않는다.
6. 녹화 PC에 `wuji-sdk==2026.8.31` 설치 승인.
7. 브리지는 표준 메시지(PointCloud2 필드로 confidence). colcon 빌드가 없고 `rosbags` 기본 typestore로 읽힌다.
8. 2D keypoint는 HaMeR만 쓴다. Dyn-HaMR은 쓰지 않는다.
9. **wuji_ros가 메인.** DexTouch·mocap_ros_py는 수정도 import도 하지 않는다.

10. **분석 env**: 새로 만든다. conda `wuji_ros`(`environment.yml`, python 3.10, torch 2.0.1+cu118). 분석 서버의 모든 분석 단계를 이 env 하나로 돌린다.
11. **HaMeR checkpoint·MANO**: `assets/` 아래 심볼릭 링크로 서버 파일을 가리킨다(README "자산").
12. **캘리브**: DexTouch의 최신 `20260824_v2`를 `calibration/20260824_v2/`에 복사해 재활용한다.
    `mocap2cam.npy`는 전달값이 아니라 유도한 임시값이다. 새 녹화에서의 유효성은 M4에서 확인한다(`docs/TODO.md`).

미결:
- D. **물체**: 다음 주 실험에서 정해진다. 물체 종류, CAD·마커 오프셋 유무에 따라 M5 분할의 물체 제거와 M9 범위가 정해진다.
