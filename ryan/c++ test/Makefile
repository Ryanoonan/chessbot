# Makefile

# Path to your PyTorch install; set this to the folder that contains:
#   include/torch/script.h
TORCH_DIR ?= /Users/ryannoonan/Desktop/Cornell\ Classes/Spring\ 2025/CS-4701/chessbot/venv/lib/python3.13/site-packages/torch

CXX       = g++
CXXFLAGS  = -std=c++17 \
             -I$(TORCH_DIR)/include \
             -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS   = -L$(TORCH_DIR)/lib \
             -ltorch -ltorch_cpu -lc10 \
             -Wl,-rpath,$(TORCH_DIR)/lib

# 1) Find all .cpp in this directory
SRC_CPP   := $(wildcard *.cpp)

# 2) Create matching .o names
OBJS      := $(SRC_CPP:.cpp=.o)

# 3) Pick up all headers so that changing any will recompile
SRC_HDR   := $(wildcard *.h)

# Binaries we want
ENGINE_EXE = engine_vs_stockfish
AGENT_EXE  = chess_agent

all: $(ENGINE_EXE) $(AGENT_EXE)

# Link engine_vs_stockfish from exactly the objects you have
$(ENGINE_EXE): $(filter-out chess_agent.o, $(OBJS))
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Link chess_agent
$(AGENT_EXE): $(filter-out main.o engine_vs_stockfish.o, $(OBJS))
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile rule: any .cpp → .o when that file or any header changes
%.o: %.cpp $(SRC_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(ENGINE_EXE) $(AGENT_EXE)
