#!/bin/bash
# extract frames from a video into a dir you can pass to ./build/dvlt.
# defaults to fps sampling (dvlt samples video ~2 fps); -n gives an exact uniformly-spaced count.
# -s picks the sharpest frame per time bucket (needs ffmpeg blurdetect, else falls back).
# usage: ./tools/v2f.sh -i input.mp4 [-f fps | -n count] [-s] [-o inputs/name/]

set -euo pipefail

FPS=2
N=""
OUTDIR=""
VIDEO=""
SHARP=0

while [ $# -gt 0 ]; do
    case "$1" in
        -i) VIDEO="$2"; shift 2 ;;
        -f) FPS="$2"; N=""; shift 2 ;;
        -n) N="$2"; shift 2 ;;
        -s) SHARP=1; shift ;;
        -o) OUTDIR="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 -i input.mp4 [-f fps | -n count] [-s] [-o output_dir]"
            echo "  -i <path>   input video (required)"
            echo "  -f <fps>    sample at this frame rate (default: 2, matches dvlt)"
            echo "  -n <int>    instead, extract exactly N uniformly-spaced frames"
            echo "  -s          pick the sharpest frame per bucket (needs blurdetect)"
            echo "  -o <dir>    output directory (default: inputs/<video name>/)"
            echo "then: ./build/dvlt inputs/<video name>/"
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$VIDEO" ]; then
    echo "error: -i <input video> is required" >&2
    echo "usage: $0 -i input.mp4 [-f fps | -n count] [-o output_dir]" >&2
    exit 1
fi
if ! command -v ffmpeg &>/dev/null; then echo "error: ffmpeg not found" >&2; exit 1; fi
if ! command -v ffprobe &>/dev/null; then echo "error: ffprobe not found" >&2; exit 1; fi
if [ ! -f "$VIDEO" ]; then echo "error: $VIDEO not found" >&2; exit 1; fi

# default output: inputs/<video name without extension>/
if [ -z "$OUTDIR" ]; then
    NAME=$(basename "$VIDEO"); OUTDIR="inputs/${NAME%.*}"
fi

# sharp mode needs the blurdetect filter; fall back to plain sampling if missing.
# (read filters into a var first: piping into grep -q trips pipefail via SIGPIPE.)
if [ "$SHARP" -eq 1 ]; then
    FILTERS=$(ffmpeg -hide_banner -filters 2>/dev/null || true)
    if ! grep -q blurdetect <<<"$FILTERS"; then
        echo "warning: this ffmpeg has no blurdetect filter, falling back to plain sampling" >&2
        SHARP=0
    fi
fi

mkdir -p "$OUTDIR"

if [ "$SHARP" -eq 1 ]; then
    # pass 1: score every frame's blur (higher = blurrier) with its index and time.
    SCORES=$(mktemp); PARSED=$(mktemp)
    trap 'rm -f "$SCORES" "$PARSED"' EXIT
    ffmpeg -v error -i "$VIDEO" -vf "blurdetect,metadata=mode=print:file=$SCORES" -f null - 2>/dev/null
    awk '
        /^frame:/ { fn=""; pt=""; nf=split($0, f, /[: \t]+/);
                    for (i=1;i<=nf;i++){ if(f[i]=="frame") fn=f[i+1]; if(f[i]=="pts_time") pt=f[i+1] } }
        /lavfi.blur=/ { split($0, b, "="); print fn, pt, b[2] }
    ' "$SCORES" > "$PARSED"
    if [ ! -s "$PARSED" ]; then
        echo "error: blurdetect produced no scores for $VIDEO" >&2; exit 1
    fi
    # pass 2: keep the sharpest (min-blur) frame in each bucket.
    #  -n: N even index buckets over all frames;  -f: one bucket per 1/FPS seconds.
    if [ -n "$N" ]; then
        SELECTED=$(awk -v N="$N" '
            { fn[NR]=$1; bl[NR]=$3 }
            END {
                t=NR; if (N>t) N=t; if (N<1) N=1;
                for (i=0;i<N;i++){
                    lo=int(i*t/N)+1; hi=int((i+1)*t/N); if(hi<lo)hi=lo;
                    bv=1e30; bf=fn[lo];
                    for (j=lo;j<=hi;j++) if (bl[j]+0<bv){ bv=bl[j]+0; bf=fn[j] }
                    print bf
                }
            }' "$PARSED")
    else
        SELECTED=$(awk -v FPS="$FPS" '
            { b=int($2*FPS); if(!(b in seen) || $3+0<bv[b]){ seen[b]=1; bv[b]=$3+0; bf[b]=$1 } }
            END { m=-1; for(b in seen) if(b+0>m)m=b+0;
                  for(b=0;b<=m;b++) if(b in seen) print bf[b] }' "$PARSED")
    fi
    SELECT=$(echo "$SELECTED" | awk '{ if(NR>1) printf "+"; printf "eq(n\\,%d)", $1 }')
    ffmpeg -v warning -i "$VIDEO" -vf "select='$SELECT'" -vsync vfr "$OUTDIR/frame_%04d.jpg"
elif [ -n "$N" ]; then
    # exact count: select N uniformly-spaced frame indices.
    TOTAL=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "$VIDEO")
    if [ -z "$TOTAL" ] || [ "$TOTAL" -lt 1 ]; then
        echo "error: could not count frames in $VIDEO" >&2; exit 1
    fi
    if [ "$N" -gt "$TOTAL" ]; then
        echo "warning: requested $N frames but video only has $TOTAL, extracting all" >&2
        N="$TOTAL"
    fi
    if [ "$N" -le 1 ]; then
        SELECT="eq(n\\,0)"
    else
        INTERVAL=$(awk "BEGIN{printf \"%.6f\", ($TOTAL - 1) / ($N - 1)}")
        SELECT=""
        for i in $(seq 0 $((N - 1))); do
            FRAME=$(awk "BEGIN{printf \"%d\", $i * $INTERVAL + 0.5}")
            if [ -n "$SELECT" ]; then SELECT="$SELECT+"; fi
            SELECT="${SELECT}eq(n\\,$FRAME)"
        done
    fi
    ffmpeg -v warning -i "$VIDEO" -vf "select='$SELECT'" -vsync vfr "$OUTDIR/frame_%04d.jpg"
else
    # fps sampling (dvlt default): keep ~FPS frames per second.
    ffmpeg -v warning -i "$VIDEO" -vf "fps=$FPS" "$OUTDIR/frame_%04d.jpg"
fi

COUNT=$(ls "$OUTDIR"/frame_*.jpg 2>/dev/null | wc -l)
echo "extracted $COUNT frames from $VIDEO -> $OUTDIR/"
echo "run: ./build/dvlt $OUTDIR/"
