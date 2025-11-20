; Системные вызовы Linux
%define SYS_WRITE   1
%define SYS_EXIT    60
%define STDOUT      1

; Размеры данных
%define FLOAT_SIZE  4
%define DOUBLE_SIZE 8

; Флаги выравнивания памяти
%define ALIGN_16    16
%define ALIGN_32    32

; Функции активации
%define ACTIVATION_RELU    0
%define ACTIVATION_SIGMOID 1
%define ACTIVATION_TANH    2
%define ACTIVATION_NONE    3

; Флаги оптимизаторов
%define OPTIM_SGD         0
%define OPTIM_SGD_MOMENTUM 1
%define OPTIM_SGD_NESTEROV 2

; Параметры по умолчанию
%define DEFAULT_LEARNING_RATE 0.01
%define DEFAULT_MOMENTUM      0.9
%define DEFAULT_WEIGHT_DECAY  0.0001