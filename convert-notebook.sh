#!/bin/bash
# convert-notebook.sh
# Usage: ./convert-notebook.sh my-notebook.ipynb [--name=SUFFIX] [param1=value1 param2=value2 ...]
 
# Input notebook
IN=$1
shift
 
# default suffix used for output file (previously hard-coded to OUT)
SUFFIX="OUT"
EXPERIMENT_NAME=""
 
# Collect remaining param-like args unless a --name is provided.
REMAINING=()
while [ $# -gt 0 ]; do
    case "$1" in
        --name=*)
            SUFFIX="${1#--name=}"
            shift
            ;;
        --name)
            shift
            if [ $# -gt 0 ]; then
                SUFFIX="$1"
                shift
            else
                echo "Error: --name requires a value" >&2
                exit 2
            fi
            ;;
        experiment_name=*)
            EXPERIMENT_NAME="${1#experiment_name=}"
            REMAINING+=("$1")
            shift
            ;;
        *)
            REMAINING+=("$1")
            shift
            ;;
    esac
done
 
# If no explicit --name was provided, decide on suffix
if [ "${SUFFIX}" = "OUT" ] && [ ${#REMAINING[@]} -gt 0 ]; then
    # If experiment_name parameter is present, use it as suffix
    if [ -n "$EXPERIMENT_NAME" ]; then
        SUFFIX="$EXPERIMENT_NAME"
    else
        # Otherwise build a suffix from all params by joining with '+'
        SUFFIX=$(printf "%s+" "${REMAINING[@]}")
        SUFFIX=${SUFFIX%+}
    fi
fi
 
# Sanitize suffix: allow only letters, numbers and a few safe punctuation characters
# (keep '+', '=', '.', '_', and '-') and replace any other char with '_'.
SANITIZED_SUFFIX=$(printf "%s" "$SUFFIX" | sed 's/[^A-Za-z0-9._+=-]/_/g')
# collapse multiple underscores
SANITIZED_SUFFIX=$(printf "%s" "$SANITIZED_SUFFIX" | sed 's/_\+/_/g')
# trim leading/trailing dots or underscores
SANITIZED_SUFFIX=$(printf "%s" "$SANITIZED_SUFFIX" | sed 's/^[_\.]*//; s/[_\.]*$//')
# fallback if empty after sanitization
if [ -z "$SANITIZED_SUFFIX" ]; then
    SANITIZED_SUFFIX="OUT"
fi
SUFFIX="$SANITIZED_SUFFIX"
 
# Place exported notebooks in an `out` subfolder next to the input notebook
IN_DIR="$(dirname "$IN")"
OUTDIR="$IN_DIR/out"
mkdir -p "$OUTDIR"
OUT_BASENAME="$(basename "${IN%.ipynb}")-$SUFFIX.ipynb"
OUT="$OUTDIR/$OUT_BASENAME"
 
# If parameters are given, use papermill; otherwise use nbconvert
if [ ${#REMAINING[@]} -gt 0 ]; then
    # Convert key=value args into papermill -p flags
    PARAMS=()
    for arg in "${REMAINING[@]}"; do
        KEY="${arg%%=*}"
        VAL="${arg#*=}"
        PARAMS+=("-p" "$KEY" "$VAL")
    done
    papermill "$IN" "$OUT" "${PARAMS[@]}"
else
    # nbconvert: pass basename and --output-dir to place file into OUTDIR
    jupyter nbconvert --to notebook --execute "$IN" --output "$OUT_BASENAME" --output-dir "$OUTDIR"
fi