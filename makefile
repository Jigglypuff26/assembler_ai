ASM = nasm
LD = ld
ASM_FLAGS = -f elf64 -g
LD_FLAGS = -static

SRC_DIR = src
CORE_DIR = $(SRC_DIR)/core
LAYERS_DIR = $(SRC_DIR)/layers
LOSS_DIR = $(SRC_DIR)/loss
OPTIM_DIR = $(SRC_DIR)/optim
UTILS_DIR = $(SRC_DIR)/utils

SOURCES = main.asm \
          $(CORE_DIR)/memory.asm \
          $(CORE_DIR)/math.asm \
          $(CORE_DIR)/linear.asm \
          $(LAYERS_DIR)/dense.asm \
          $(LAYERS_DIR)/activation.asm \
          $(LAYERS_DIR)/conv.asm \
          $(LAYERS_DIR)/pooling.asm \
          $(LAYERS_DIR)/rnn.asm \
          $(LAYERS_DIR)/lstm.asm \
          $(LOSS_DIR)/mse.asm \
          $(OPTIM_DIR)/sgd.asm \
          $(OPTIM_DIR)/adam.asm \
          $(OPTIM_DIR)/rmsprop.asm \
          $(UTILS_DIR)/serialization.asm \
          $(UTILS_DIR)/im2col.asm

OBJECTS = $(SOURCES:.asm=.o)

TARGET = neural_net

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LD) $(LD_FLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.asm
	$(ASM) $(ASM_FLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

test: $(TARGET)
	./$(TARGET)

.PHONY: all clean test