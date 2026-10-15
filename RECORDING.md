# 녹화 PC 명령어

repo: `~/ros2_ws/src/wuji_ros`. 명령은 한 블록씩 붙여넣는다.

## 처음 한 번

설치

```bash
cd ~/ros2_ws/src/wuji_ros && git pull && git submodule update --init third_party/wuji-sdk
```
```bash
bash scripts/shells/setup_bridge.sh
```

네트워크 (장갑 포트 `enx00e04c681c5d`, PC `192.168.1.50`, 장갑 `192.168.1.101`)

```bash
sudo nmcli connection add type ethernet con-name wuji-glove ifname enx00e04c681c5d ipv4.method manual ipv4.addresses 192.168.1.50/24 ipv4.never-default yes ipv6.method disabled
```
```bash
sudo nmcli connection modify wuji-glove ipv4.routes 224.0.0.224/32,192.168.1.101/32
```
```bash
sudo nmcli connection up wuji-glove
```

랜 어댑터를 바꿨을 때만

```bash
sudo nmcli connection modify wuji-glove connection.interface-name enx00e04c681c5d
```

## 피험자마다 (브리지·Studio 끈 상태)

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

## 매번 녹화

1. 연결 확인

```bash
nmcli device status; ip route get 192.168.1.101
```

2. 촉각 확인 (착용 후, 브리지 끈 상태, `q`로 종료)

```bash
source /opt/ros/humble/setup.bash && cd ~/ros2_ws/src/wuji_ros/third_party/wuji-sdk/examples/python/wuji_glove && python3.10 8.tactile_contact_view.py
```

3. 브리지 (터미널 1)

```bash
cd ~/ros2_ws/src/wuji_ros && bash scripts/shells/glove_bridge.sh
```

검색이 안 될 때

```bash
cd ~/ros2_ws/src/wuji_ros && bash scripts/shells/glove_bridge.sh --address 192.168.1.101:50001
```

4. 토픽 확인 (터미널 2)

```bash
source /opt/ros/humble/setup.bash && ros2 topic hz /wuji_glove/right/hand_skeleton
```
```bash
source /opt/ros/humble/setup.bash && ros2 topic list
```

5. 녹화 (터미널 2, Ctrl+C로 종료, 이름은 매번 새로)

카메라·mocap 포함 전체

```bash
cd ~/ros2_ws/src/wuji_ros && bash scripts/shells/record_glove.sh subj01_take01 --all
```

장갑만

```bash
cd ~/ros2_ws/src/wuji_ros && bash scripts/shells/record_glove.sh subj01_take01
```

6. bag 확인

```bash
cd ~/ros2_ws/src/wuji_ros && bash scripts/shells/check_bag.sh subj01_take01
```
```bash
df -h ~/ros2_ws
```

## 문제 해결

```bash
cat /sys/class/net/enx00e04c681c5d/carrier
```
```bash
ip neigh show 192.168.1.101
```
```bash
cd ~/ros2_ws/src/wuji_ros && bash scripts/shells/glove_netcheck.sh
```
