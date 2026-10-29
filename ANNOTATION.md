# 실행 안내: 녹화 폴더 → 물체에 닿게 보정한 손(`contact`)

수집기가 고정 주기로 뽑아 둔 녹화 폴더 하나에서 world 좌표의 오른손 MANO를 만든다. 물체를 들고 있는 프레임에서는 손 위치를 물체에 닿게 보정한다.
각 단계의 내용은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), 결정 사항은 [plan.md](plan.md) §6 18, 실험 결과는 [docs/experiments/001_glove_hand_pose/experiments.md](docs/experiments/001_glove_hand_pose/experiments.md)(E014–E016)에 있다.

## 준비

- conda env `wuji_ros`, `assets/`의 MANO·HaMeR 링크: README "환경", "자산".
- GPU 한 장. `keypoints`와 `fit4`는 CUDA로만 돈다.
- 저장소는 `f6a58a0` 이후 버전.

## 입력 데이터 위치

```
wuji_ros/
├── data/brl_mocap_cam_hand/
│   ├── <세션>/                                ← 수집기 출력 그대로
│   │   ├── image_metadata.csv
│   │   ├── capture_config.json
│   │   ├── wuji_info.json
│   │   ├── camera1/color/000001.png …, camera1/depth/000001.png …   (camera1–camera4)
│   │   ├── wuji/000001.npz …                  (장갑)
│   │   └── mocap/000001.npz …                 (물체 마커)
│   ├── config_rerun_object_mk4_wuji.yaml      ← 카메라 내부 파라미터 + 물체 마커 오프셋
│   └── 01_Hexagonal_Prism_with_tracker.stl    ← 물체 CAD (mm, OBJ 좌표)
└── calibration/20261015/                      ← w2c_ext.npz, t2r_ext.npz, mocap2cam.npy (저장소에 포함, 읽기 전용)
```

- 경로는 명령 인자로 넘기므로 다른 곳에 둬도 된다. 아래 명령은 이 위치를 기준으로 한다.
- `import`는 원본을 건드리지 않고 `data/<세션>/`에 같은 레이아웃으로 정리한다.

## 명령 (`contact`까지)

```bash
cd /home/user/wj/wuji_ros
S=20261022_163051_mocap_mk4_wuji_fixed_rate
B=data/brl_mocap_cam_hand
R="conda run --no-capture-output -n wuji_ros python scripts/run.py"
export CUDA_VISIBLE_DEVICES=0

$R import $B/$S --camera-config $B/config_rerun_object_mk4_wuji.yaml --every 0.5 --force
$R segment   --session $S --force
$R glovefit  --session $S --force
$R keypoints --session $S --force
$R fit4      --session $S --force
$R contact   --session $S --object-config $B/config_rerun_object_mk4_wuji.yaml \
             --object-mesh $B/01_Hexagonal_Prism_with_tracker.stl --force
```

- `--force`를 빼면 이미 있는 단계 출력을 다시 쓴다.
- `20261022_163051`(0.5 s 간격 50프레임)은 RTX 3090 한 장에서 약 5분 걸린다.

그림까지 보려면(선택):

```bash
$R overlay --session $S --hand contact --object-config $B/config_rerun_object_mk4_wuji.yaml \
           --object-mesh $B/01_Hexagonal_Prism_with_tracker.stl --force
```

- `output/$S/overlay_contact_object/`, `overlay_contact_object.{mp4,gif}`가 생긴다. 보정한 손과 물체를 4개 카메라 화면에 그린다.
- `--hand contact`를 빼면 보정 전(`fit4`) 손에 물체를 더한 그림이 `overlay_object/`에 생긴다.
- 그림에 참가자 얼굴이 보이므로 공유할 때 주의한다.

## 결과

`output/<세션>/contact/NNNNNN.npz`가 프레임별 최종 손이다. `fit4`에 결과가 있는 프레임마다 하나씩 있고, 좌표는 캘리브 world, 단위는 m다.

| 키 | 내용 |
|---|---|
| `verts_world` (778,3), `joints_world` (21,3) | MANO 정점과 관절(OpenPose-21 순서) |
| `R_world_from_mano` (3,3), `wrist_world` (3,), `betas` (10,) | 손 회전, 손목 위치, 손 크기 β |
| `T_world_from_wrist` (4,4) | 장갑 `r_wrist` 좌표계 → world (촉각 taxel, 장갑 골격용) |
| `held` | 물체를 들고 있는 프레임으로 판정했는지 |
| `t_world` (3,) | 보정한 손 이동량. 들고 있지 않은 프레임은 0 |
| `T_world_from_obj` (4,4) | 물체 자세(mocap 마커 + `mocap2cam`, camera1 경유) |
| `min_sd_mm_before/after`, `inside_before/after` | 보정 전후 손-물체 최소 부호 거리(음수는 관통 깊이), 물체 안 정점 수 |
| `reproj_px_before`, `reproj_px`, `surface_mm_before`, `surface_mm_median` | 보정 전후 영상 일치(HaMeR 2D 재투영, depth 표면) |

- 손가락 자세 θ는 `output/<세션>/glovefit/NNNNNN.npz`의 `hand_pose`(15,3,3 회전행렬)다. `contact`는 손가락을 바꾸지 않는다.
- MANO 파라미터로 다시 만들 때의 식:
  `verts_world = R_world_from_mano · (MANO(β, θ, 회전 I, 이동 0) − 그 손목 관절) + wrist_world`
- `contact/summary.json`에는 들고 있는 프레임 목록, 이동량, 보정 전후 지표가 있다.

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

1. 녹화 폴더를 `data/brl_mocap_cam_hand/<새 세션>/`에 넣고 `S`를 바꾼다.
2. `wuji_ros/rig.py`의 `SESSION_CALIBRATION`에 `"<새 세션>": "20261015",`를 더한다. 없으면 `segment`에서 멈춘다.
   다른 날짜의 캘리브라면 연구실의 파일을 먼저 받는다. 캘리브는 추정하지도 고치지도 않는다.
3. yaml의 `object.marker_offsets`가 그 녹화의 마커 이름 대응과 맞는지 확인한다. 녹화마다 바뀐 적이 있다.
   `overlay --object-config ... --object-mesh ...`를 한 번 돌려 로그의 "CAD fit RMSE"가 수 mm 이하인지 본다(`20261022_163051`은 0.30 mm).
