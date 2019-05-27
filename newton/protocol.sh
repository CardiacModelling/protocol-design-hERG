PROTOCOLS=(
        "staircaseramp"
        "sis"
        "sisi"
        "hh3step"
        "wang3step"
        "squarewave"
        "maxdiff"
        "longap"
        "manualppx"
        )

for protocol in "${PROTOCOLS[@]}"
do
    echo "$protocol"
    python protocol.py $protocol hh_ikr_rt wang_ikr_rt
done
