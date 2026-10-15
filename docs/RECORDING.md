# 녹화 PC 운영 절차

Wuji Glove(오른손)를 ROS 2 토픽으로 보내고 bag으로 녹화하는 절차다. 2026-10-15 녹화 PC에서 처음부터 끝까지 확인했다.

- 녹화 PC: Ubuntu 22.04, ROS 2 Humble, repo 위치 `~/ros2_ws/src/wuji_ros`
- 장갑: `WG1KA06260805538`(오른손), 펌웨어 `0.11.1`, `wuji-sdk==2026.8.31`
- 명령은 모두 한 줄이다. 터미널에 붙여넣을 때 줄바꿈이나 둥근 따옴표(`“ ”`)로 깨지지 않게 따옴표를 쓰지 않는다. 한 블록씩 붙여넣는다.
- python은 항상 `python3.10`이다. rclpy가 3.10용이라 그냥 `python3`로 돌리면 `No module named 'rclpy._rclpy_pybind11'`이 난다.

## 1. 처음 한 번

### 1-1. repo와 SDK

```bash
cd ~/ros2_ws/src && git clone https://github.com/jinmiles/wuji_ros.git
```
```bash
cd ~/ros2_ws/src/wuji_ros && git submodule update --init third_party/wuji-sdk
```
```bash
bash scripts/shells/setup_bridge.sh
```

마지막 줄에 `bridge env ok`가 나오면 된다. `wuji-sdk`는 venv 없이 `python3.10 --user`에 깔린다(런타임 의존성이 없어 ROS 패키지는 바뀌지 않는다).

### 1-2. 하드웨어 연결

장갑은 USB가 아니라 이더넷 장치다.

```
장갑 데이터 수집 보드 ─USB-C↔USB-C─ 이더넷 전원 어댑터 ─이더넷─ PC USB 랜 어댑터(enx00e04c681c5d)
                                        └─ USB-A→USB-C 전원
```

- 보드와 어댑터 사이 USB-C 케이블은 양끝 흰색 라벨이 같은 쪽을 향해야 한다. 반대면 링크가 안 잡힌다.
- 어댑터 전원은 USB-A→USB-C 케이블이어야 한다(C→C로는 전원이 안 들어간다).
- 보드 표시등이 녹색을 거쳐 청록색 호흡(4초 주기)이 되면 PC 연결 대기 상태다.

### 1-3. 네트워크

| 항목 | 값 |
|---|---|
| 장갑 IP (오른손 / 왼손) | `192.168.1.101` / `192.168.1.100`, UDP 50000(검색), 50001(데이터), DHCP 없음 |
| 장갑 MAC (오른손) | `02:1b:00:33:00:17` |
| PC 장갑 포트 | USB 랜 어댑터 `enx00e04c681c5d` |
| PC IP | `192.168.1.50/24`, NetworkManager 프로파일 `wuji-glove`. PC에 `.100`/`.101`을 쓰면 안 된다 |
| 건드리지 않는 포트 | `enx00e04c265a20` 연구실 LAN(기본 경로), `enx00e04c345f80` 프로파일 `mocap` |

```bash
sudo nmcli connection add type ethernet con-name wuji-glove ifname enx00e04c681c5d ipv4.method manual ipv4.addresses 192.168.1.50/24 ipv4.never-default yes ipv6.method disabled
```
```bash
sudo nmcli connection modify wuji-glove ipv4.routes 224.0.0.224/32,192.168.1.101/32
```
```bash
sudo nmcli connection up wuji-glove
```

- 첫 줄은 프로파일이 없을 때만 실행한다.
- `224.0.0.224/32`: SDK·Studio의 장치 검색(Zenoh 멀티캐스트)이 연구실 LAN으로 나가지 않게 한다. 없으면 `No devices found`가 난다.
- `192.168.1.101/32`: 다른 포트가 같은 대역을 쓰더라도 장갑 IP로 가는 패킷을 장갑 포트로 고정한다. 장갑 IP를 바꾸면 같이 바꾼다.
- 모두 NetworkManager에 저장되어 재부팅해도 유지된다.
- 랜 어댑터를 바꾸면 이름이 바뀐다. 랜선을 꽂을 때 `ip link`에 새로 나타나는 `enx…`로 다시 지정한다: `sudo nmcli connection modify wuji-glove connection.interface-name <새 이름>`

### 1-4. SDK 사용자와 캘리브레이션 (피험자마다)

캘리브레이션은 SDK 사용자(named user)와 장갑 SN별로 `~/.wuji/sdk/users/` 아래에 저장되고, 브리지가 시작할 때 자동으로 읽는다. 기본(Default) 사용자는 손 모델 캘리브레이션을 저장하지 않는다. 캘리브레이션 중에는 브리지와 Wuji Studio를 끈다.

```bash
source /opt/ros/humble/setup.bash && cd ~/ros2_ws/src/wuji_ros/third_party/wuji-sdk/examples/python/wuji_glove
```
```bash
python3.10 4.user.py --user-name subj01 --list
```
```bash
python3.10 5.calibration.py
```
```bash
python3.10 7.tactile_calibration.py
```
```bash
python3.10 8.tactile_contact_view.py
```

- `4.user.py`는 인자 없이 실행하면 현재 사용자를 보여 주기만 한다. `--user-name`을 줘야 만들고 전환한다. 이름은 피험자 실명 대신 ID를 쓴다.
- `5.calibration.py`: 손 모델(6개 자세). 제조사는 내장 모델로 먼저 써 보고 부족할 때만 보정하라고 권한다.
- `7.tactile_calibration.py`: 촉각(빈손, 4개 동작). 없으면 `tactile_binary`, `tactile_residual`이 나오지 않는다.
- `8.tactile_contact_view.py`: 누른 부위가 `#`로 보이면 된다. `q`로 종료.
- Wuji Studio의 Calibration 화면으로 해도 같은 위치에 저장된다.

## 2. 매번 녹화할 때

### 2-1. 연결 확인

장갑 전원을 넣고 표시등이 청록색 호흡이 되면:

```bash
nmcli device status; ip route get 192.168.1.101
```

`enx00e04c681c5d … connected wuji-glove`와 `192.168.1.101 dev enx00e04c681c5d src 192.168.1.50`이 나오면 된다.

### 2-2. 브리지 실행 (터미널 1)

```bash
cd ~/ros2_ws/src/wuji_ros && git pull && bash scripts/shells/glove_bridge.sh
```

- 정상: `glove: {...}` 다음에 `streaming to /wuji_glove/right/*`, 스트림마다 `first frame` 줄. 표시등이 청록색 심박(짧게 깜빡)으로 바뀐다.
- 손가락 IMU 10개의 `not offered by this glove, skipped` 경고는 정상이다(이 장갑은 손등 IMU 1개뿐이다).
- `Failed to download ELF ... v0.11.1` 경고는 SDK의 장치 로그 해석용이라 데이터와 무관하다.
- `No devices found`면 주소로 바로 연결한다: `bash scripts/shells/glove_bridge.sh --address 192.168.1.101:50001`
- 종료는 Ctrl+C.

### 2-3. 토픽 확인 (터미널 2)

```bash
source /opt/ros/humble/setup.bash && ros2 topic hz /wuji_glove/right/hand_skeleton
```
```bash
source /opt/ros/humble/setup.bash && ros2 topic echo --once --full-length /wuji_glove/right/info
```

- `hand_skeleton`은 약 120 Hz여야 한다.
- `info`에서 `sdk_user_is_default: false`(캘리브레이션한 사용자), `unavailable_streams`(손가락 IMU 10개뿐), `urdf_source`를 확인한다.

### 2-4. 녹화 (터미널 2)

```bash
cd ~/ros2_ws/src/wuji_ros && bash scripts/shells/record_glove.sh subj01_take01
```

- `data/bags/<이름>/`에 sqlite3 bag으로 저장된다. 이름이 이미 있으면 실패하니 매번 새 이름을 쓴다.
- 장갑 토픽 전부와 `/tf_static`을 기록한다. 카메라·mocap 토픽은 뒤에 이름을 붙이면 같이 기록된다: `bash scripts/shells/record_glove.sh subj01_take01 /camera1/color/image_raw /camera1/aligned_depth_to_color/image_raw`
- 브리지를 먼저 띄운다. `info`와 `/tf_static`은 latched라서 녹화를 나중에 시작해도 bag에 들어간다.
- 종료는 Ctrl+C. `Recording stopped`가 나올 때까지 기다린다.

### 2-5. bag 확인

```bash
source /opt/ros/humble/setup.bash && ros2 bag info data/bags/subj01_take01
```

2026-10-15 시험 녹화(12.0 s) 기준 정상값:

| 토픽 | 개수 | 주기 |
|---|---|---|
| `emf_poses`, `tip_poses`, `hand_skeleton`, `hand_joint_angles` | 각 약 1,446 | 120 Hz |
| `tactile`, `tactile_binary`, `tactile_residual`, `tactile_zones/{palm,thumb,index,middle,ring,pinky}`, `tactile_point_cloud` | 각 약 1,446 | 120 Hz |
| `imu_raw/palm`, `imu_data/palm` | 약 9,200 | 약 760~770 Hz |
| `tf` (IMU 기반 `waist → r_wrist`) | 약 9,150 | 약 760 Hz |
| `info`, `/tf_static` | 1 이상 | latched |

- 용량은 장갑만 약 3.7 MB/s(1시간 약 13 GB)다. 녹화 전 디스크 여유를 확인한다: `df -h ~/ros2_ws`
- `Messages: 0`이면 브리지가 꺼져 있었거나 토픽 이름이 다르다. 브리지 터미널과 `ros2 topic list`를 확인한다.

## 3. 문제 해결

| 증상 | 확인 / 조치 |
|---|---|
| 표시등 빨간색 2 Hz | 이더넷 끊김. 랜선 양끝과 어댑터 램프 확인 |
| `cat /sys/class/net/enx00e04c681c5d/carrier`가 `0` | 링크 없음. 랜선 접촉 불량(2026-10-15 원인), USB-C 라벨 방향 확인 |
| `No devices found` | `ip route show 224.0.0.224`가 장갑 포트인지 확인(1-3). 급하면 `--address 192.168.1.101:50001` |
| `Connection timeout` | `ip route get 192.168.1.101`이 장갑 포트·`src 192.168.1.50`인지 확인. 장갑 전원을 껐다 켠다 |
| 그 밖의 연결 문제 | `bash scripts/shells/glove_netcheck.sh`(NIC, ARP 응답, 장갑 패킷 진단. 설정은 바꾸지 않는다) |
| `rclpy._rclpy_pybind11` 없음 | `python3` 대신 `python3.10`으로 실행 |
| 표시등 노란색 0.5 Hz | 연결은 됐지만 데이터를 가져가는 프로그램이 없다. 브리지를 다시 띄운다 |

- 장갑은 ping에 응답하지 않을 수 있다. 연결 여부는 `ip neigh show 192.168.1.101`(`lladdr 02:1b:… REACHABLE` 또는 `STALE`)이나 브리지의 `first frame`으로 판단한다.
- 표시등: 청록색 호흡 = 연결 대기, 청록색 심박 = 데이터 전송 중, 노란색 0.5 Hz = 연결됐지만 데이터를 가져가지 않음, 빨간색 2 Hz = 이더넷 끊김, 빨간색 점등 = 센서 고장.
- 제조사 문의(`support@wuji.tech`)에는 SN, 펌웨어 버전, SDK 버전, `lsb_release -a`, 표시등 패턴, `~/.wuji/logs/` 전체를 함께 보낸다.
