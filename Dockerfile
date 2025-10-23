FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# System dependencies & build tools
RUN apt-get update && apt-get install -y \
    # Compilers:
    gcc-11 \
    g++-11 \
    # Build tools:
    cmake \
    make \
    # Vesion control:
    git \
    # Python (for PenGym env)
    python3.10 \
    python3-pip \
    # Debugging tools:
    gdb \
    valgrind \
    linux-tools-generic \
    # Monitoring tools:
    htop \
    vim \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*



