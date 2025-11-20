section .data
align 32
    zero: dd 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    one: dd 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

section .text
    global conv2d_forward, conv2d_backward
    global conv2d_init, conv2d_free

; Структура Conv2D слоя
; typedef struct {
;     float* kernels;      // +0 [out_channels, in_channels, kernel_h, kernel_w]
;     float* biases;       // +8 [out_channels]
;     float* input;        // +16
;     float* output;       // +24  
;     float* dkernels;     // +32
;     float* dbiases;      // +40
;     float* dinput;       // +48
;     size_t in_channels;  // +56
;     size_t out_channels; // +64
;     size_t kernel_h;     // +72
;     size_t kernel_w;     // +80
;     size_t stride;       // +88
;     size_t padding;      // +96
; } Conv2DLayer;

; Conv2DLayer* conv2d_init(size_t in_ch, size_t out_ch, size_t k_h, size_t k_w, size_t stride, size_t padding)
conv2d_init:
    push r12
    push r13
    push r14
    push r15
    
    mov r12, rdi  ; in_ch
    mov r13, rsi  ; out_ch
    mov r14, rdx  ; k_h
    mov r15, rcx  ; k_w
    
    ; Выделяем память под структуру (104 байта)
    mov rdi, 104
    call malloc_asm
    test rax, rax
    jz .error
    
    mov r8, rax   ; сохраняем указатель
    
    ; Вычисляем размеры массивов
    ; kernels: out_ch * in_ch * k_h * k_w
    mov rax, r13
    mul r12
    mul r14
    mul r15
    shl rax, 2
    mov rdi, rax
    call malloc_asm
    mov [r8], rax
    
    ; biases: out_ch
    mov rdi, r13
    shl rdi, 2
    call malloc_asm
    mov [r8 + 8], rax
    
    ; dkernels: такой же размер как kernels
    mov rax, r13
    mul r12
    mul r14
    mul r15
    shl rax, 2
    mov rdi, rax
    call malloc_asm
    mov [r8 + 32], rax
    
    ; dbiases: out_ch
    mov rdi, r13
    shl rdi, 2
    call malloc_asm
    mov [r8 + 40], rax
    
    ; Инициализируем веса
    mov rdi, [r8]        ; kernels
    mov rsi, r13
    imul rsi, r12
    imul rsi, r14
    imul rsi, r15        ; total elements
    call initialize_weights_xavier
    
    ; Инициализируем смещения нулями
    mov rdi, [r8 + 8]    ; biases
    mov rsi, r13
    call initialize_biases
    
    ; Сохраняем параметры
    mov [r8 + 56], r12   ; in_channels
    mov [r8 + 64], r13   ; out_channels
    mov [r8 + 72], r14   ; kernel_h
    mov [r8 + 80], r15   ; kernel_w
    mov [r8 + 88], r8    ; stride
    mov [r8 + 96], r9    ; padding
    
    mov rax, r8
    jmp .end

.error:
    xor rax, rax

.end:
    pop r15
    pop r14
    pop r13
    pop r12
    ret

; void conv2d_forward(Conv2DLayer* layer, float* input, float* output, 
;                    size_t batch, size_t input_h, size_t input_w)
conv2d_forward:
    push r15
    push r14
    push r13
    push r12
    push r11
    push r10
    push r9
    push r8
    
    mov r15, rdi        ; layer
    mov r14, rsi        ; input
    mov r13, rdx        ; output
    mov r12, rcx        ; batch
    mov r11, r8         ; input_h
    mov r10, r9         ; input_w
    
    ; Сохраняем вход
    mov [r15 + 16], rsi
    
    ; Расчет размеров выхода
    mov rax, r11        ; input_h
    add rax, [r15 + 96] ; + padding
    add rax, [r15 + 96] ; + padding
    sub rax, [r15 + 72] ; - kernel_h
    mov rbx, [r15 + 88] ; stride
    xor rdx, rdx
    div rbx
    inc rax             ; output_h = (input_h + 2*padding - kernel_h)/stride + 1
    mov r9, rax         ; сохраняем output_h
    
    mov rax, r10        ; input_w
    add rax, [r15 + 96] ; + padding
    add rax, [r15 + 96] ; + padding
    sub rax, [r15 + 80] ; - kernel_w
    mov rbx, [r15 + 88] ; stride
    xor rdx, rdx
    div rbx
    inc rax             ; output_w
    mov r8, rax         ; сохраняем output_w
    
    ; Основные циклы
    xor rbx, rbx        ; batch index
.batch_loop:
    cmp rbx, r12
    jge .end_batch
    
    xor rcx, rcx        ; out_channel
.out_channel_loop:
    cmp rcx, [r15 + 64]
    jge .end_out_channel
    
    xor rdx, rdx        ; output_y
.output_y_loop:
    cmp rdx, r9
    jge .end_output_y
    
    xor rsi, rsi        ; output_x
.output_x_loop:
    cmp rsi, r8
    jge .end_output_x
    
    ; Вычисление одной свертки
    push r8
    push r9
    push r10
    push r11
    push r12
    
    mov rdi, r15        ; layer
    mov rsi, rsi        ; output_x
    mov rdx, rdx        ; output_y
    mov rcx, rcx        ; out_channel
    mov r8, rbx         ; batch
    call single_convolution
    
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    
    inc rsi
    jmp .output_x_loop

.end_output_x:
    inc rdx
    jmp .output_y_loop

.end_output_y:
    inc rcx
    jmp .out_channel_loop

.end_out_channel:
    inc rbx
    jmp .batch_loop

.end_batch:
    pop r8
    pop r9
    pop r10
    pop r11
    pop r12
    pop r13
    pop r14
    pop r15
    ret

; Вычисление одной свертки
single_convolution:
    vxorps ymm0, ymm0, ymm0 ; аккумулятор
    
    mov rax, [rdi + 56]  ; in_channels
    mov rbx, [rdi + 72]  ; kernel_h
    mov rcx, [rdi + 80]  ; kernel_w
    
    xor r8, r8           ; in_channel
.in_channel_loop:
    cmp r8, rax
    jge .end_in_channel
    
    xor r9, r9           ; kernel_y
.kernel_y_loop:
    cmp r9, rbx
    jge .end_kernel_y
    
    xor r10, r10         ; kernel_x
.kernel_x_loop:
    cmp r10, rcx
    jge .end_kernel_x
    
    ; Вычисление позиций
    call compute_positions
    
    ; Загрузка и умножение
    vmovups ymm1, [kernel_ptr]
    vmovups ymm2, [input_ptr]
    vmulps ymm3, ymm1, ymm2
    vaddps ymm0, ymm0, ymm3
    
    add kernel_ptr, 32
    add input_ptr, 32
    
    inc r10
    jmp .kernel_x_loop

.end_kernel_x:
    inc r9
    jmp .kernel_y_loop

.end_kernel_y:
    inc r8
    jmp .in_channel_loop

.end_in_channel:
    ; Добавление смещения
    vmovss xmm1, [bias_ptr]
    vaddps ymm0, ymm0, ymm1
    
    ; Сохранение результата
    vmovups [output_ptr], ymm0
    
    ret

; void conv2d_backward(Conv2DLayer* layer, float* doutput)
conv2d_backward:
    ; Реализация обратного распространения для свертки
    push r15
    mov r15, rdi
    
    ; Вычисление градиентов для ядер
    call compute_kernel_gradients
    
    ; Вычисление градиентов для смещений
    call compute_bias_gradients
    
    ; Вычисление градиентов для входа
    call compute_input_gradients
    
    pop r15
    ret

compute_kernel_gradients:
    ; Реализация вычисления dkernels
    ret

compute_bias_gradients:
    ; dbiases = sum(doutput, axis=(0,2,3))
    ret

compute_input_gradients:
    ; dinput = full_convolution(doutput, rotated_kernels)
    ret

; void conv2d_free(Conv2DLayer* layer)
conv2d_free:
    test rdi, rdi
    jz .end
    
    push r12
    mov r12, rdi
    
    ; Освобождаем все массивы
    mov rdi, [r12]       ; kernels
    call free_asm
    
    mov rdi, [r12 + 8]   ; biases
    call free_asm
    
    mov rdi, [r12 + 32]  ; dkernels
    call free_asm
    
    mov rdi, [r12 + 40]  ; dbiases
    call free_asm
    
    ; Освобождаем структуру
    mov rdi, r12
    call free_asm
    
    pop r12

.end:
    ret