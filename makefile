# Makefile для Neural ASM

ASM = nasm
LD = ld
ASM_FLAGS = -f elf64 -g
LD_FLAGS = -static

SRC_DIR = src
CORE_DIR = $(SRC_DIR)/core
LAYERS_DIR = $(SRC_DIR)/layers
LOSS_DIR = $(SRC_DIR)/loss
OPTIM_DIR = $(SRC_DIR)/optim

SOURCES = main.asm $(wildcard src/*/*.asm)
OBJECTS = $(SOURCES:.asm=.o)

TARGET = neural_asm

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LD) $(LD_FLAGS) -o $@ $^

%.o: %.asm
	$(ASM) $(ASM_FLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean