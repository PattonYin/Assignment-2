#!/bin/zsh
# fix_robostack_macos.sh — Fix zsh/sh incompatibility in robostack ROS2
#
# Usage:  conda activate <your_env> && zsh fix_robostack_macos.sh

set -e

if [ -z "$CONDA_PREFIX" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "ERROR: Activate your ROS2 conda env first."; exit 1
fi

SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/ros-jazzy-ros-workspace_activate.sh"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: ros-jazzy-ros-workspace not found."; exit 1
fi

cat > "$SCRIPT" << 'EOF'
_CONDA_BIN="$CONDA_PREFIX/bin"
if [ "$CONDA_BUILD" = "1" -a "$target_platform" != "$build_platform" ]; then
    echo "Not activating ROS when cross-compiling";
else
    if [ -n "$ZSH_VERSION" ] && [ -f "$CONDA_PREFIX/setup.zsh" ]; then
        source "$CONDA_PREFIX/setup.zsh"
    else
        source "$CONDA_PREFIX/setup.sh"
    fi
fi
export PATH="$_CONDA_BIN:$PATH"
unset _CONDA_BIN
case "$OSTYPE" in
  darwin*)  export ROS_OS_OVERRIDE="conda:osx";;
  linux*)   export ROS_OS_OVERRIDE="conda:linux";;
esac
export ROS_ETC_DIR=$CONDA_PREFIX/etc/ros
export AMENT_PREFIX_PATH=$CONDA_PREFIX
EOF

echo "Done. Now: conda deactivate && conda activate $CONDA_DEFAULT_ENV"