# ============================
#   Makefile for CUDA Sudoku
# ============================

# --- CONFIG ---
NVCC = nvcc
CFLAGS = -O3 -std=c++17
TARGET = kernel.exe
SRC = kernel.cu

# --- RULES ---
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS)

# Sprzątanie
clean:
	rm -f $(TARGET)

# Dodatkowe reguły użytkowe
run:
	./$(TARGET)

.PHONY: clean run
