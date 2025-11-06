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

# Set gcc-11 and g++-11 as the default compilers    
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    gymnasium \
    pybind11

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create working directory
WORKDIR /app

CMD ["/bin/bash"]