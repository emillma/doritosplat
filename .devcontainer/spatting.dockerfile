## Unofficial Dockerfile for 3D Gaussian Splatting for Real-Time Radiance Field Rendering
## Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
## https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel


# NOTE:
# Building the libraries for this repository requires cuda *DURING BUILD PHASE*, therefore:
# - The default-runtime for container should be set to "nvidia" in the deamon.json file. See this: https://github.com/NVIDIA/nvidia-docker/issues/1033
# - For the above to work, the nvidia-container-runtime should be installed in your host. Tested with version 1.14.0-rc.2
# - Make sure NVIDIA's drivers are updated in the host machine. Tested with 525.125.06

ENV DEBIAN_FRONTEND=noninteractive

# Update and install tzdata separately
# RUN apt update && apt install -y tzdata

# Install necessary packages
# RUN apt install -y git && \
# apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev && \
# apt clean && apt install wget && rm -rf /var/lib/apt/lists/*


# RUN apt install -y cmake pkg-config mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev libglew-dev libglfw3-dev libglm-dev libao-dev libmpg123-dev

# RUN conda env create --file environment.yml && conda init bash && exec bash && conda activate gaussian_splatting

WORKDIR /include
RUN apt update
RUN apt install -y \
    build-essential \
    cmake \
    freeglut3-dev \
    git \
    libao-dev \
    libassimp-dev \
    libavcodec-dev \
    libavdevice-dev \
    libboost-all-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libceres-dev \
    libcgal-dev \
    libeigen3-dev \
    libembree-dev \
    libflann-dev \
    libfreeimage-dev \
    libglew-dev \
    libglfw3-dev \
    libglm-dev \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgtk-3-dev \
    libmetis-dev \
    libmpg123-dev \
    libopencv-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libxxf86vm-dev \
    mesa-common-dev \
    mesa-utils \
    ninja-build \
    pkg-config \
    qtbase5-dev \
    tzdata


# WORKDIR /include
# RUN git clone https://github.com/colmap/colmap
# WORKDIR /include/colmap/build
# RUN cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
# RUN ninja
# RUN ninja install


# # Tweak the CMake file for matching the existing OpenCV version. Fix the naming of FindEmbree.cmake
# WORKDIR /workspaces/gaussian-splatting/SIBR_viewers
# RUN sed -i 's/find_package(OpenCV 4\.5 REQUIRED)/find_package(OpenCV 4.2 REQUIRED)/g' cmake/linux/dependencies.cmake
# RUN sed -i 's/find_package(embree 3\.0 )/find_package(EMBREE)/g' cmake/linux/dependencies.cmake
# RUN mv cmake/linux/Modules/FindEmbree.cmake cmake/linux/Modules/FindEMBREE.cmake
# RUN sed -i 's/\bembree\b/embree3/g' src/core/raycaster/CMakeLists.txt

# # Fix the naming of the embree library in the rayscaster's cmake

# # Ready to build the viewer now.
# WORKDIR /workspace/gaussian-splatting/SIBR_viewers 
# RUN cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j24 --target install
