# wuji_ros

Wuji Glove + 4뷰 RealSense RGB-D + OptiTrack(물체) 녹화에서 오른손의 MANO pose를 world 좌표로 추정한다.
손가락 관절은 glove가, 손 전체의 world 6D는 4뷰 RGB-D 정합이 맡는다.
계획과 결정 사항은 [plan.md](plan.md), 구조와 좌표 규약은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)에 있다.

현재 상태: M0(문서·환경·의존성). 코드는 아직 없다.

## 구성

| 경로 | 내용 |
|---|---|
| `plan.md` | 목표, 마일스톤, 완료 조건, 결정 사항 |
| `docs/` | 구조(`ARCHITECTURE.md`), 미해결 항목(`TODO.md`), 실험 기록(`experiments/`) |
| `calibration/<name>/` | 리그 캘리브 파일 (커밋). 현재 `20260824_v2` |
| `environment.yml` | 분석 env `wuji_ros` 정의 |
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
