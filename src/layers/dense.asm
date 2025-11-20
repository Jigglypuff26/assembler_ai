section .data
align 16
    ; Глобальный экземпляр оптимизатора
    optim_instance: dq 0
    default_lr: dd 0.01
    default_momentum: dd 0.9

section .text
    global dense_layer_init, dense_layer_forward
    global dense_layer_backward, dense_layer_update
    global dense_layer_set_optimizer, dense_layer_free

    extern malloc_asm, free_asm, memset_float_asm
    extern vec_add_asm, vec_dot_asm
    extern matrix_multiply_simple, outer_product, matrix_multiply_transpose
    extern relu_asm, sigmoid_asm, tanh_asm
    extern relu_derivative_asm, sigmoid_derivative_asm, tanh_derivative_asm
    extern sgd_optimizer_init, sgd_optimizer_update, sgd_optimizer_free
    extern sgd_update_learning_rate

; DenseLayer* dense_layer_init(size_t input_size, size_t output_size, int activation)
; rdi = input_size, rsi = output_size, rdx = activation
dense_layer_init:
    push r12
    push r13
    push r14
    push r15
    
    ; Сохраняем параметры
    mov r12, rdi  ; input_size
    mov r13, rsi  ; output_size  
    mov r14, rdx  ; activation
    
    ; Выделяем память под структуру слоя (88 байт)
    mov rdi, 88
    call malloc_asm
    test rax, rax
    jz .error
    
    ; Сохраняем указатель на структуру
    mov r15, rax
    
    ; Вычисляем размеры массивов
    ; weights: input_size * output_size
    mov rax, r12
    mul r13
    mov r8, rax  ; сохраняем total_elements
    shl rax, 2   ; * sizeof(float)
    mov rdi, rax
    call malloc_asm
    mov [r15], rax  ; weights
    
    ; biases: output_size
    mov rdi, r13
    shl rdi, 2
    call malloc_asm
    mov [r15 + 8], rax  ; biases
    
    ; output: output_size
    mov rdi, r13
    shl rdi, 2
    call malloc_asm
    mov [r15 + 24], rax  ; output
    
    ; dweights: input_size * output_size
    mov rax, r12
    mul r13
    shl rax, 2
    mov rdi, rax
    call malloc_asm
    mov [r15 + 32], rax  ; dweights
    
    ; dbiases: output_size
    mov rdi, r13
    shl rdi, 2
    call malloc_asm
    mov [r15 + 40], rax  ; dbiases
    
    ; dinput: input_size
    mov rdi, r12
    shl rdi, 2
    call malloc_asm
    mov [r15 + 48], rax  ; dinput
    
    ; Инициализируем веса случайными значениями
    mov rdi, [r15]  ; weights
    mov rsi, r8     ; total elements
    call initialize_weights_xavier
    
    ; Инициализируем смещения нулями
    mov rdi, [r15 + 8]  ; biases
    mov rsi, r13        ; output_size
    call initialize_biases
    
    ; Инициализируем оптимизатор по умолчанию
    mov rdi, r8         ; param_size (input_size * output_size)
    call initialize_default_optimizer
    mov [r15 + 80], rax ; optimizer
    
    ; Сохраняем размеры и тип активации
    mov [r15 + 56], r12  ; input_size
    mov [r15 + 64], r13  ; output_size
    mov [r15 + 72], r14  ; activation
    
    mov rax, r15
    jmp .end

.error:
    xor rax, rax

.end:
    pop r15
    pop r14
    pop r13
    pop r12
    ret

; Инициализация весов методом Xavier/Glorot
initialize_weights_xavier:
    push r12
    push r13
    
    mov r12, rdi  ; weights
    mov r13, rsi  ; size
    
    ; Для Xavier инициализации: limit = sqrt(6 / (fan_in + fan_out))
    ; В упрощенной версии используем небольшие случайные значения
    mov rcx, r13
    mov eax, 0x3DCCCCCD  ; ~0.1 в float
    
.loop:
    mov [r12], eax
    add r12, 4
    dec rcx
    jnz .loop
    
    pop r13
    pop r12
    ret

; void initialize_biases(float* biases, size_t size)  
initialize_biases:
    vxorps xmm0, xmm0, xmm0
    call memset_float_asm
    ret

; Инициализация оптимизатора по умолчанию
; void* initialize_default_optimizer(size_t param_size)
initialize_default_optimizer:
    push r12
    mov r12, rdi  ; param_size
    
    ; SGD с learning_rate = 0.01, momentum = 0.9
    vmovss xmm0, [default_lr]
    vmovss xmm1, [default_momentum]
    vxorps xmm2, xmm2, xmm2  ; weight_decay = 0
    mov rdi, 0               ; nesterov = false
    mov rsi, r12             ; param_size
    call sgd_optimizer_init
    
    pop r12
    ret

; float* dense_layer_forward(DenseLayer* layer, float* input)
; rdi = layer, rsi = input
dense_layer_forward:
    push r12
    push r13
    push r14
    
    mov r12, rdi  ; layer
    mov r13, rsi  ; input
    
    ; Сохраняем вход
    mov [r12 + 16], rsi
    
    ; output = weights * input + biases
    mov r14, [r12 + 64]  ; output_size
    mov rdi, [r12]       ; weights
    mov rsi, r13         ; input
    mov rdx, [r12 + 24]  ; output
    mov rcx, r14         ; rows (output_size)
    mov r8, [r12 + 56]   ; cols (input_size)
    mov r9, 1            ; colsB = 1
    
    call matrix_multiply_simple
    
    ; Добавляем смещения
    mov rdi, [r12 + 24]  ; output
    mov rsi, [r12 + 8]   ; biases
    mov rdx, [r12 + 24]  ; output (результат)
    mov rcx, r14         ; output_size
    call vec_add_asm
    
    ; Применяем функцию активации
    mov eax, [r12 + 72]  ; activation
    cmp eax, ACTIVATION_RELU
    jne .sigmoid_check
    
    ; ReLU activation
    mov rdi, [r12 + 24]  ; output
    mov rsi, [r12 + 24]  ; output (in-place)
    mov rdx, r14         ; output_size
    call relu_asm
    jmp .end

.sigmoid_check:
    cmp eax, ACTIVATION_SIGMOID
    jne .tanh_check
    
    ; Sigmoid activation
    mov rdi, [r12 + 24]  ; output
    mov rsi, [r12 + 24]  ; output (in-place)
    mov rdx, r14         ; output_size
    call sigmoid_asm
    jmp .end

.tanh_check:
    cmp eax, ACTIVATION_TANH
    jne .end
    
    ; Tanh activation
    mov rdi, [r12 + 24]  ; output
    mov rsi, [r12 + 24]  ; output (in-place)
    mov rdx, r14         ; output_size
    call tanh_asm

.end:
    mov rax, [r12 + 24]  ; возвращаем output
    pop r14
    pop r13
    pop r12
    ret

; void dense_layer_backward(DenseLayer* layer, float* doutput)
; rdi = layer, rsi = doutput
dense_layer_backward:
    push r12
    push r13
    push r14
    push r15
    
    mov r12, rdi  ; layer
    mov r13, rsi  ; doutput
    
    ; Применяем производную функции активации, если нужно
    mov eax, [r12 + 72]  ; activation
    cmp eax, ACTIVATION_NONE
    je .no_activation_derivative
    
    ; Применяем производную активации к doutput
    mov rdi, r13         ; doutput
    mov rsi, [r12 + 24]  ; output слоя
    mov rdx, r13         ; doutput (in-place)
    mov rcx, [r12 + 64]  ; output_size
    call apply_activation_derivative
    jmp .continue_backward

.no_activation_derivative:
    ; Для линейной активации производная = 1, ничего не делаем

.continue_backward:
    ; dbiases = doutput
    mov rdi, [r12 + 40]  ; dbiases
    mov rsi, r13         ; doutput
    mov rdx, [r12 + 40]  ; dbiases (in-place)
    mov rcx, [r12 + 64]  ; output_size
    call vec_add_asm
    
    ; dweights = input * doutput^T
    mov r14, [r12 + 16]  ; input
    mov r15, r13         ; doutput
    
    mov rdi, r14         ; input
    mov rsi, r15         ; doutput
    mov rdx, [r12 + 32]  ; dweights
    mov rcx, [r12 + 56]  ; input_size (rows)
    mov r8, 1            ; colsA = 1
    mov r9, [r12 + 64]   ; output_size (colsB)
    
    call outer_product
    
    ; dinput = weights^T * doutput
    mov rdi, [r12]       ; weights
    mov rsi, r15         ; doutput
    mov rdx, [r12 + 48]  ; dinput
    mov rcx, [r12 + 56]  ; input_size (rows результата)
    mov r8, [r12 + 64]   ; output_size (cols weights)
    mov r9, 1            ; colsB = 1
    
    call matrix_multiply_transpose
    
    pop r15
    pop r14
    pop r13
    pop r12
    ret

; Применение производной функции активации
; void apply_activation_derivative(float* doutput, float* output, float* result, size_t len)
apply_activation_derivative:
    ; В этой упрощенной версии предполагаем, что тип активации известен
    ; В реальной реализации нужно передавать тип активации как параметр
    
    ; rdi = doutput, rsi = output, rdx = result, rcx = len
    ; Для упрощения всегда используем ReLU derivative
    mov rdi, rsi         ; output (первый параметр для relu_derivative)
    mov rsi, rdi         ; doutput (второй параметр) 
    mov rdx, rcx         ; len (третий параметр)
    call relu_derivative_asm
    ret

; void dense_layer_update(DenseLayer* layer, float learning_rate)
; rdi = layer, xmm0 = learning_rate
dense_layer_update:
    push r12
    
    mov r12, rdi
    
    ; Обновляем learning_rate в оптимизаторе если передан
    vxorps xmm1, xmm1, xmm1
    vcomiss xmm0, xmm1
    jbe .no_lr_update
    
    ; Обновляем learning_rate
    mov rdi, [r12 + 80]  ; optimizer
    call sgd_update_learning_rate

.no_lr_update:
    ; Обновляем веса с помощью оптимизатора
    mov rdi, [r12]       ; weights
    mov rsi, [r12 + 32]  ; dweights
    mov rdx, [r12 + 56]  ; input_size
    imul rdx, [r12 + 64] ; * output_size
    mov rcx, [r12 + 80]  ; optimizer
    call sgd_optimizer_update
    
    ; Обновляем смещения (используем тот же оптимизатор)
    mov rdi, [r12 + 8]   ; biases
    mov rsi, [r12 + 40]  ; dbiases
    mov rdx, [r12 + 64]  ; output_size
    mov rcx, [r12 + 80]  ; optimizer
    call sgd_optimizer_update
    
    ; Очищаем градиенты для следующей итерации
    mov rdi, r12
    call clear_gradients
    
    pop r12
    ret

; Очистка градиентов
clear_gradients:
    push r12
    mov r12, rdi
    
    ; Очищаем dweights
    mov rdi, [r12 + 32]  ; dweights
    vxorps xmm0, xmm0, xmm0
    mov rsi, [r12 + 56]  ; input_size
    imul rsi, [r12 + 64] ; * output_size
    call memset_float_asm
    
    ; Очищаем dbiases
    mov rdi, [r12 + 40]  ; dbiases
    vxorps xmm0, xmm0, xmm0
    mov rsi, [r12 + 64]  ; output_size
    call memset_float_asm
    
    ; Очищаем dinput
    mov rdi, [r12 + 48]  ; dinput
    vxorps xmm0, xmm0, xmm0
    mov rsi, [r12 + 56]  ; input_size
    call memset_float_asm
    
    pop r12
    ret

; Установка кастомного оптимизатора
; void dense_layer_set_optimizer(DenseLayer* layer, void* optimizer)
dense_layer_set_optimizer:
    ; Освобождаем старый оптимизатор если есть
    mov rax, [rdi + 80]
    test rax, rax
    jz .set_new
    
    push rdi
    push rsi
    mov rdi, rax
    call sgd_optimizer_free
    pop rsi
    pop rdi

.set_new:
    mov [rdi + 80], rsi
    ret

; Освобождение памяти слоя
; void dense_layer_free(DenseLayer* layer)
dense_layer_free:
    test rdi, rdi
    jz .end
    
    push r12
    mov r12, rdi
    
    ; Освобождаем все массивы
    mov rdi, [r12]       ; weights
    call free_asm
    
    mov rdi, [r12 + 8]   ; biases
    call free_asm
    
    mov rdi, [r12 + 24]  ; output
    call free_asm
    
    mov rdi, [r12 + 32]  ; dweights
    call free_asm
    
    mov rdi, [r12 + 40]  ; dbiases
    call free_asm
    
    mov rdi, [r12 + 48]  ; dinput
    call free_asm
    
    ; Освобождаем оптимизатор
    mov rdi, [r12 + 80]  ; optimizer
    test rdi, rdi
    jz .free_struct
    
    call sgd_optimizer_free

.free_struct:
    ; Освобождаем саму структуру
    mov rdi, r12
    call free_asm
    
    pop r12

.end:
    ret