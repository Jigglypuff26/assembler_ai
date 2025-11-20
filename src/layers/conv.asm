section .data
align 32
    zero: dd 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

section .text
    global conv2d_forward, conv2d_backward
    global max_pool2d_forward, avg_pool2d_forward

; Структура Conv2D слоя
; typedef struct {
;     float* kernels;      // +0 [out_channels, in_channels, kernel_h, kernel_w]
;     float* biases;       // +8 [out_channels]
;     float* input;        // +16
;     float* output;       // +24  
;     size_t in_channels;  // +32
;     size_t out_channels; // +40
;     size_t kernel_h;     // +48
;     size_t kernel_w;     // +56
;     size_t stride;       // +64
;     size_t padding;      // +72
; } Conv2DLayer;

; void conv2d_forward(float* input, float* output, float* kernels, float* biases,
;                    size_t batch, size_t in_channels, size_t out_channels,
;                    size_t input_h, size_t input_w, size_t kernel_h, size_t kernel_w,
;                    size_t stride, size_t padding)
conv2d_forward:
    push r15
    push r14
    push r13
    push r12
    
    ; Расчет размеров выхода
    mov rax, [rsp+72]    ; input_h
    mov rbx, [rsp+80]    ; kernel_h
    mov rcx, [rsp+96]    ; padding
    shl rcx, 1           ; padding * 2
    add rax, rcx
    sub rax, rbx
    inc rax              ; output_h = (input_h + 2*padding - kernel_h)/stride + 1
    
    ; Аналогично для output_w
    
    ; Основной цикл свертки
    xor r15, r15 ; batch
.batch_loop:
    xor r14, r14 ; out_channel
.out_channel_loop:
    xor r13, r13 ; output_y
.output_y_loop:
    xor r12, r12 ; output_x
.output_x_loop:
    
    ; Вычисление свертки для одной позиции
    call single_convolution
    
    inc r12
    cmp r12, output_w
    jl .output_x_loop
    
    inc r13
    cmp r13, output_h
    jl .output_y_loop
    
    inc r14
    cmp r14, out_channels
    jl .out_channel_loop
    
    inc r15
    cmp r15, batch
    jl .batch_loop
    
    pop r12
    pop r13
    pop r14
    pop r15
    ret

single_convolution:
    ; Реализация одной свертки с AVX
    vxorps ymm0, ymm0, ymm0 ; аккумулятор
    
    mov rcx, kernel_h
.kernel_y_loop:
    mov rdx, kernel_w
.kernel_x_loop:
    
    ; Загрузка данных ядра и входного окна
    vmovups ymm1, [kernel_ptr]
    vmovups ymm2, [input_ptr]
    vmulps ymm3, ymm1, ymm2
    vaddps ymm0, ymm0, ymm3
    
    add kernel_ptr, 32
    add input_ptr, 32
    
    dec rdx
    jnz .kernel_x_loop
    
    dec rcx
    jnz .kernel_y_loop
    
    ; Добавление смещения и сохранение
    vmovss xmm1, [bias_ptr]
    vaddps ymm0, ymm0, ymm1
    vmovups [output_ptr], ymm0
    
    ret