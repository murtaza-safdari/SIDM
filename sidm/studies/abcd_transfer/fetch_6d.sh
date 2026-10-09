#!/usr/bin/env bash
# Copy the merged inclusive 6D ABCD histograms from EOS to local disk on the LPC.
#
# Source: Dongyub Lee's campaign ABCD_6D_inclusive_eHEM_skimv1 (merged outputs, read-only).
# Destination: $ABCD6D_LOCAL (default /uscms_data/d3/murtazas/abcd6d_local), one directory
# per kind (data, bkg, signal), each .coffea file next to its .meta.yaml sidecar.
# Files already present with the same size as on EOS are skipped.  Needs a VOMS proxy.
set -euo pipefail

REDIRECTOR=root://cmseos.fnal.gov
CAMPAIGN=/store/user/dlee3/sidm_condor/ABCD_6D_inclusive_eHEM_skimv1
DEST=${ABCD6D_LOCAL:-/uscms_data/d3/murtazas/abcd6d_local}
KINDS=${*:-data bkg signal}

for kind in $KINDS; do
    src="$CAMPAIGN/ABCD_6D_inclusive_eHEM_skimv1_v1_${kind}_merged_samples_v1"
    dst="$DEST/$kind"
    mkdir -p "$dst"
    copied=0; skipped=0
    while read -r size path; do
        name=$(basename "$path")
        case "$name" in *.coffea|*.meta.yaml) ;; *) continue ;; esac
        if [ -f "$dst/$name" ] && [ "$(stat -c %s "$dst/$name")" = "$size" ]; then
            skipped=$((skipped + 1)); continue
        fi
        xrdcp --silent --force "$REDIRECTOR/$path" "$dst/$name"
        [ "$(stat -c %s "$dst/$name")" = "$size" ] || { echo "size mismatch after copy: $name" >&2; exit 1; }
        copied=$((copied + 1))
    done < <(xrdfs "$REDIRECTOR" ls -l "$src" | awk '{print $4, $NF}')
    echo "$kind: $copied copied, $skipped already present -> $dst"
done

{
    echo "source: $REDIRECTOR$CAMPAIGN"
    echo "fetched: $(date -u +%Y-%m-%dT%H:%MZ)"
    for kind in data bkg signal; do
        [ -d "$DEST/$kind" ] && find "$DEST/$kind" -name '*.coffea' -printf '%s %p\n' | sort -k2
    done
} > "$DEST/MANIFEST.txt"
echo "manifest: $DEST/MANIFEST.txt ($(grep -c coffea "$DEST/MANIFEST.txt") files)"
