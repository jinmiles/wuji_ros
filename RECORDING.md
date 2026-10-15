# 녹화 PC 명령어

repo: `~/ros2_ws/src/wuji_ros`. 명령은 한 블록씩 붙여넣는다.

## 처음 한 번

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

## 캘리브레이션 (Wuji Studio, 사람이 바뀔 때마다)

사람이 바뀌면 손 모델·촉각 캘리브레이션을 둘 다 다시 한다. 브리지는 끈 상태에서 한다.

```bash
wuji-studio
```

1. 장치 카드 → Connect.
2. Calibration: 안내에 따라 6개 자세(엄지-검지/중지/약지/새끼 끝 맞대기, 네 손가락 90°, 손 펴기).
3. Tactile Calibration: 아무것도 만지지 않은 상태로 안내되는 4개 동작. 끝나면 누른 부위가 빨간색으로 표시되는지 확인.
4. Studio를 닫는다.

## 매번 녹화

1. 연결 확인

```bash
nmcli device status; ip route get 192.168.1.101
```

2. 촉각 확인 (착용 후, 브리지 끈 상태)

```bash
wuji-studio
```

Connect → Enter Visualization. 아무것도 만지지 않았는데 촉각 히트맵이 노란색이거나 값이 높으면 Tactile Calibration을 다시 한다. 확인 후 Studio를 닫는다.

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
