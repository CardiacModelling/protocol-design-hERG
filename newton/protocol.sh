PROTOCOLS=(
        "staircaseramp"
        "sis"
        "sisi"
        "hh3step"
        "wang3step"
        "squarewave"
        "maxdiff"
        "spacefill10"
        "spacefill19"
        "spacefill26"
        "hhbrute3gstep"
        "wangbrute3gstep"
        "hhsobol3step"
        "wangsobol3step"
        "longap"
        "manualppx"
        )

for protocol in "${PROTOCOLS[@]}"
do
    echo "$protocol"
    python protocol.py $protocol hh_ikr_rt wang_ikr_rt
done
