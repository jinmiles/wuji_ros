#!/usr/bin/env bash
# Recording PC: find which NIC and IP the Wuji Glove answers on. Read-only: it changes no
# network settings, only pings 192.168.1.0/24 and runs one SDK device scan while
# capturing packets that the PC itself did not send. Needs sudo for tcpdump.
# Usage: bash scripts/shells/glove_netcheck.sh      (report also saved under scratch/)
set -eo pipefail
cd "$(dirname "$0")/../.."
mkdir -p scratch
log="scratch/glove_netcheck_$(date +%Y%m%d_%H%M%S).log"
cap_dir="$(mktemp -d)"
exec > >(tee "$log") 2>&1

wired=()
echo "== 1. NICs (carrier 1 = cable link up)"
for dev in /sys/class/net/*; do
  name="$(basename "$dev")"
  [[ "$name" == lo || ! -e "$dev/device" ]] && continue  # skip loopback and virtual NICs
  carrier="$(cat "$dev/carrier" 2>/dev/null || echo 0)"
  speed="$(cat "$dev/speed" 2>/dev/null || echo -)"
  addrs="$(ip -4 -br addr show dev "$name" | awk '{$1=$2=""; print $0}')"
  printf '%-18s carrier=%s speed=%-6s mac=%s ipv4=%s\n' "$name" "$carrier" "$speed" "$(cat "$dev/address")" "${addrs:- none}"
  if [[ "$carrier" == 1 ]]; then wired+=("$name"); fi
done
echo "default route: $(ip route show default | head -1)"
for ip in 192.168.1.100 192.168.1.101; do echo "route to $ip: $(ip route get "$ip" | head -1)"; done

echo
echo "== 2. ARP sweep of 192.168.1.0/24 on NICs that have a 192.168.1.x address"
for name in "${wired[@]}"; do
  src="$(ip -4 -o addr show dev "$name" | awk '{print $4}' | cut -d/ -f1 | grep '^192\.168\.1\.' | head -1 || true)"
  [[ -z "$src" ]] && continue
  echo "-- $name (src $src)"
  # wait on the pings only: a bare `wait` would also wait for the tee logger
  ping_pids=()
  for i in $(seq 1 254); do ping -c1 -W1 -I "$src" "192.168.1.$i" >/dev/null 2>&1 & ping_pids+=($!); done
  wait "${ping_pids[@]}" 2>/dev/null || true
  ip neigh show dev "$name" | grep -vE "FAILED|INCOMPLETE" | grep -v "^$src " || echo "   no device answered"
done

echo
echo "== 3. Packets from other devices during an SDK scan (NICs with link: ${wired[*]:-none})"
sudo -v  # ask for the password once, before tcpdump runs in the background
pids=()
for name in "${wired[@]}"; do
  mac="$(cat "/sys/class/net/$name/address")"
  sudo timeout 20 tcpdump -eni "$name" -c 40 "not ether src $mac and not port 22" \
    > "$cap_dir/$name.txt" 2>/dev/null &
  pids+=($!)
done
sleep 2
source /opt/ros/humble/setup.bash
python3.10 - <<'EOF' || true
from wuji_sdk import SdkManager
for _ in range(3):
    devices = SdkManager.instance().scan()
    print("sdk scan:", [(d.sn, d.address, str(d.device_type)) for d in devices] or "no device found")
    if devices:
        break
EOF
wait "${pids[@]}" 2>/dev/null || true
for name in "${wired[@]}"; do
  echo "-- $name"
  if [[ -s "$cap_dir/$name.txt" ]]; then head -20 "$cap_dir/$name.txt"; else echo "   nothing received"; fi
done
rm -rf "$cap_dir"
echo
echo "report saved to $log"
