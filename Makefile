# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -pthread -Wall -Wextra -g
INCLUDES = -Iinclude -Isrc/CPU/environment -Isrc/CPU/queues -Isrc/CPU/worker_threads

# Directories
SRC_DIR = src/CPU
TEST_DIR = tests
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Portable directory creation helper
# On Unix-like systems use `mkdir -p`; on Windows (cmd.exe) use `if not exist`.
ifeq ($(OS),Windows_NT)
define MKDIR_DIR
	if not exist "$(subst /,\\,$1)" mkdir "$(subst /,\\,$1)"
endef
else
define MKDIR_DIR
	mkdir -p $1
endef
endif

# Source files
ENV_SOURCES = $(SRC_DIR)/environment/pentest_env.cpp
QUEUE_SOURCES = $(SRC_DIR)/queues/request_queue.cpp \
                $(SRC_DIR)/queues/response_queue.cpp
WORKER_SOURCES = $(SRC_DIR)/worker_threads/worker_thread.cpp

ALL_SOURCES = $(ENV_SOURCES) $(QUEUE_SOURCES) $(WORKER_SOURCES)

# Object files
ENV_OBJS = $(patsubst $(SRC_DIR)/environment/%.cpp,$(OBJ_DIR)/environment/%.o,$(ENV_SOURCES))
QUEUE_OBJS = $(patsubst $(SRC_DIR)/queues/%.cpp,$(OBJ_DIR)/queues/%.o,$(QUEUE_SOURCES))
WORKER_OBJS = $(patsubst $(SRC_DIR)/worker_threads/%.cpp,$(OBJ_DIR)/worker_threads/%.o,$(WORKER_SOURCES))

ALL_OBJS = $(ENV_OBJS) $(QUEUE_OBJS) $(WORKER_OBJS)

# Test executables
TEST_ENV = $(BIN_DIR)/test_env
TEST_QUEUES = $(BIN_DIR)/test_queues
TEST_WORKERS = $(BIN_DIR)/test_worker_threads

# Default target
all: directories $(TEST_ENV) $(TEST_QUEUES) $(TEST_WORKERS)
	@echo ""
	@echo "Build complete!"
	@echo ""
	@echo "Run tests:"
	@echo "  make test        - Run all tests"
	@echo "  make test-env    - Test environment only"
	@echo "  make test-queues - Test queues only"
	@echo "  make test-workers - Test worker threads only"

# Create necessary directories
directories:
	@$(call MKDIR_DIR,$(OBJ_DIR)/environment)
	@$(call MKDIR_DIR,$(OBJ_DIR)/queues)
	@$(call MKDIR_DIR,$(OBJ_DIR)/worker_threads)
	@$(call MKDIR_DIR,$(BIN_DIR))

# Compile environment objects
$(OBJ_DIR)/environment/%.o: $(SRC_DIR)/environment/%.cpp
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile queue objects
$(OBJ_DIR)/queues/%.o: $(SRC_DIR)/queues/%.cpp
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile worker thread objects
$(OBJ_DIR)/worker_threads/%.o: $(SRC_DIR)/worker_threads/%.cpp
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build test_env
$(TEST_ENV): $(ENV_OBJS) $(TEST_DIR)/test_env.cpp
	@echo "Building test_env..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_DIR)/test_env.cpp $(ENV_OBJS) -o $@

# Build test_queues
$(TEST_QUEUES): $(ENV_OBJS) $(QUEUE_OBJS) $(TEST_DIR)/test_queues.cpp
	@echo "Building test_queues..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_DIR)/test_queues.cpp $(ENV_OBJS) $(QUEUE_OBJS) -o $@

# Build test_worker_threads
$(TEST_WORKERS): $(ALL_OBJS) $(TEST_DIR)/test_worker_threads.cpp
	@echo "Building test_worker_threads..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_DIR)/test_worker_threads.cpp $(ALL_OBJS) -o $@

# Run all tests
test: $(TEST_ENV) $(TEST_QUEUES) $(TEST_WORKERS)
	@echo ""
	@echo "Running all tests..."
	@echo ""
	@echo "Environment Tests:"
	@$(TEST_ENV)
	@echo ""
	@echo "Queue Tests:"
	@$(TEST_QUEUES)
	@echo ""
	@echo "Worker Thread Tests:"
	@$(TEST_WORKERS)
	@echo ""
	@echo "All tests completed!"

# Run individual tests
test-env: $(TEST_ENV)
	@echo "Running environment tests..."
	@$(TEST_ENV)

test-queues: $(TEST_QUEUES)
	@echo "Running queue tests..."
	@$(TEST_QUEUES)

test-workers: $(TEST_WORKERS)
	@echo "Running worker thread tests..."
	@$(TEST_WORKERS)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

# Clean and rebuild
rebuild: clean all

# Help target
help:
	@echo "Available targets:"
	@echo "  all            - Build all test executables (default)"
	@echo "  test           - Run all tests"
	@echo "  test-env       - Run environment tests only"
	@echo "  test-queues    - Run queue tests only"
	@echo "  test-workers   - Run worker thread tests only"
	@echo "  clean          - Remove build artifacts"
	@echo "  rebuild        - Clean and rebuild"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make           # Build everything"
	@echo "  make test      # Run all tests"
	@echo "  make clean     # Clean up"

.PHONY: all directories test test-env test-queues test-workers clean rebuild help