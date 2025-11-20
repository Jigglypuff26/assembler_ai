section .data
align 32
    ; Константы для оптимизатора
    momentum_default: dd 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9
    zero_float: dd 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

section .bss
    ; Глобальные переменные для momentum
    global velocity_weights, velocity_biases
    velocity_weights: resq 1  ; указатель на velocity для весов
    velocity_biases: resq 1   ; указатель на velocity для смещений

section .text
    global sgd_optimizer_init, sgd_optimizer_update
    global sgd_momentum_update, sgd_nesterov_update

; Структура SGD оптимизатора
; typedef struct {
;     float learning_rate;    // +0
;     float momentum;         // +4  
;     float weight_decay;     // +8
;     int nesterov;           // +12
;     size_t param_size;      // +16
; } SGDOptimizer;

; SGDOptimizer* sgd_optimizer_init(float lr, float momentum, float weight_decay, int nesterov, size_t param_size)
; xmm0 = lr, xmm1 = momentum, xmm2 = weight_decay, rdi = nesterov, rsi = param_size
sgd_optimizer_init:
    push r12
    push r13
    
    ; Сохраняем параметры
    mov r12, rdi  ; nesterov
    mov r13, rsi  ; param_size
    
    ; Выделяем память под структуру (24 байта)
    mov rdi, 24
    call malloc_asm
    test rax, rax
    jz .error
    
    ; Сохраняем указатель на структуру
    mov r15, rax
    
    ; Заполняем структуру
    vmovss [rax], xmm0        ; learning_rate
    vmovss [rax + 4], xmm1    ; momentum
    vmovss [rax + 8], xmm2    ; weight_decay
    mov [rax + 12], r12       ; nesterov
    mov [rax + 16], r13       ; param_size
    
    ; Инициализируем velocity buffers если используется momentum
    vxorps xmm3, xmm3, xmm3
    vcomiss xmm1, xmm3
    jbe .no_momentum          ; если momentum == 0, пропускаем
    
    ; Выделяем память для velocity весов
    mov rdi, r13
    shl rdi, 2  ; * sizeof(float)
    call malloc_asm
    mov [velocity_weights], rax
    
    ; Инициализируем нулями
    mov rdi, rax
    vxorps xmm0, xmm0, xmm0
    mov rsi, r13
    call memset_float_asm
    
    ; Выделяем память для velocity смещений (если нужно)
    mov rdi, r13
    shl rdi, 2
    call malloc_asm
    mov [velocity_biases], rax
    
    mov rdi, rax
    vxorps xmm0, xmm0, xmm0
    mov rsi, r13
    call memset_float_asm

.no_momentum:
    mov rax, r15
    jmp .end

.error:
    xor rax, rax

.end:
    pop r13
    pop r12
    ret

; Базовый SGD update
; void sgd_optimizer_update(float* params, float* grads, size_t len, SGDOptimizer* optim)
; rdi = params, rsi = grads, rdx = len, rcx = optim
sgd_optimizer_update:
    test rdx, rdx
    jz .end
    
    ; Проверяем использование momentum
    vmovss xmm3, [rcx + 4]    ; momentum
    vxorps xmm4, xmm4, xmm4
    vcomiss xmm3, xmm4
    ja .use_momentum           ; если momentum > 0
    
    ; Базовый SGD без momentum
    vmovss xmm0, [rcx]        ; learning_rate
    call sgd_basic_update
    jmp .end

.use_momentum:
    ; Проверяем Nesterov momentum
    mov eax, [rcx + 12]       ; nesterov flag
    test eax, eax
    jnz .use_nesterov
    
    ; SGD с классическим momentum
    vmovss xmm0, [rcx]        ; learning_rate
    vmovss xmm1, [rcx + 4]    ; momentum
    mov r8, [velocity_weights]
    call sgd_momentum_update
    jmp .end

.use_nesterov:
    ; SGD с Nesterov momentum
    vmovss xmm0, [rcx]        ; learning_rate
    vmovss xmm1, [rcx + 4]    ; momentum
    mov r8, [velocity_weights]
    call sgd_nesterov_update

.end:
    ret

; Базовый SGD: param = param - lr * grad
; void sgd_basic_update(float* params, float* grads, size_t len, float lr)
; rdi = params, rsi = grads, rdx = len, xmm0 = lr
sgd_basic_update:
    test rdx, rdx
    jz .end
    
    ; Дублируем learning_rate по всему регистру
    vshufps xmm0, xmm0, xmm0, 0
    vinsertf128 ymm0, ymm0, xmm0, 1
    
    mov r8, rdx
    shr r8, 3
    jz .process_remainder

.process_8:
    vmovups ymm1, [rsi]       ; grads
    vmovups ymm2, [rdi]       ; params
    
    vmulps ymm1, ymm1, ymm0   ; lr * grads
    vsubps ymm2, ymm2, ymm1   ; params - lr * grads
    
    vmovups [rdi], ymm2       ; обновляем params
    
    add rdi, 32
    add rsi, 32
    dec r8
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end

.process_1:
    vmovss xmm1, [rsi]        ; grad
    vmovss xmm2, [rdi]        ; param
    
    vmulss xmm1, xmm1, xmm0   ; lr * grad
    vsubss xmm2, xmm2, xmm1   ; param - lr * grad
    
    vmovss [rdi], xmm2
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

; SGD с Momentum: 
; velocity = momentum * velocity - lr * grad
; param = param + velocity
; void sgd_momentum_update(float* params, float* grads, size_t len, float lr, float momentum, float* velocity)
; rdi = params, rsi = grads, rdx = len, xmm0 = lr, xmm1 = momentum, r8 = velocity
sgd_momentum_update:
    push r12
    mov r12, r8  ; сохраняем velocity
    
    test rdx, rdx
    jz .end
    
    ; Дублируем коэффициенты
    vshufps xmm0, xmm0, xmm0, 0
    vinsertf128 ymm0, ymm0, xmm0, 1  ; lr
    vshufps xmm1, xmm1, xmm1, 0
    vinsertf128 ymm1, ymm1, xmm1, 1  ; momentum
    
    mov r8, rdx
    shr r8, 3
    jz .process_remainder

.process_8:
    ; Загружаем данные
    vmovups ymm2, [r12]       ; velocity
    vmovups ymm3, [rsi]       ; grads
    vmovups ymm4, [rdi]       ; params
    
    ; velocity = momentum * velocity - lr * grads
    vmulps ymm5, ymm1, ymm2   ; momentum * velocity
    vmulps ymm6, ymm0, ymm3   ; lr * grads
    vsubps ymm2, ymm5, ymm6   ; новое velocity
    
    ; param = param + velocity
    vaddps ymm4, ymm4, ymm2
    
    ; Сохраняем результаты
    vmovups [r12], ymm2       ; обновляем velocity
    vmovups [rdi], ymm4       ; обновляем params
    
    add rdi, 32
    add rsi, 32
    add r12, 32
    dec r8
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end

.process_1:
    vmovss xmm2, [r12]        ; velocity
    vmovss xmm3, [rsi]        ; grad
    vmovss xmm4, [rdi]        ; param
    
    ; velocity = momentum * velocity - lr * grad
    vmulss xmm5, xmm1, xmm2   ; momentum * velocity
    vmulss xmm6, xmm0, xmm3   ; lr * grad
    vsubss xmm2, xmm5, xmm6   ; новое velocity
    
    ; param = param + velocity
    vaddss xmm4, xmm4, xmm2
    
    ; Сохраняем результаты
    vmovss [r12], xmm2
    vmovss [rdi], xmm4
    
    add rdi, 4
    add rsi, 4
    add r12, 4
    dec rdx
    jnz .process_1

.end:
    pop r12
    vzeroupper
    ret

; SGD с Nesterov Momentum:
; velocity_prev = velocity
; velocity = momentum * velocity - lr * grad
; param = param - momentum * velocity_prev + (1 + momentum) * velocity
; void sgd_nesterov_update(float* params, float* grads, size_t len, float lr, float momentum, float* velocity)
; rdi = params, rsi = grads, rdx = len, xmm0 = lr, xmm1 = momentum, r8 = velocity
sgd_nesterov_update:
    push r12
    mov r12, r8  ; сохраняем velocity
    
    test rdx, rdx
    jz .end
    
    ; Дублируем коэффициенты
    vshufps xmm0, xmm0, xmm0, 0
    vinsertf128 ymm0, ymm0, xmm0, 1  ; lr
    vshufps xmm1, xmm1, xmm1, 0
    vinsertf128 ymm1, ymm1, xmm1, 1  ; momentum
    
    ; Вычисляем (1 + momentum)
    vmovaps ymm7, [one]
    vaddps ymm7, ymm7, ymm1   ; 1 + momentum
    
    mov r8, rdx
    shr r8, 3
    jz .process_remainder

.process_8:
    ; Загружаем данные
    vmovups ymm2, [r12]       ; velocity
    vmovups ymm3, [rsi]       ; grads
    vmovups ymm4, [rdi]       ; params
    
    ; Сохраняем старый velocity
    vmovaps ymm5, ymm2        ; velocity_prev = velocity
    
    ; velocity = momentum * velocity - lr * grads
    vmulps ymm2, ymm1, ymm2   ; momentum * velocity
    vmulps ymm6, ymm0, ymm3   ; lr * grads
    vsubps ymm2, ymm2, ymm6   ; новое velocity
    
    ; param = param - momentum * velocity_prev + (1 + momentum) * velocity
    vmulps ymm5, ymm1, ymm5   ; momentum * velocity_prev
    vsubps ymm4, ymm4, ymm5   ; param - momentum * velocity_prev
    
    vmulps ymm6, ymm7, ymm2   ; (1 + momentum) * velocity
    vaddps ymm4, ymm4, ymm6   ; final param
    
    ; Сохраняем результаты
    vmovups [r12], ymm2       ; обновляем velocity
    vmovups [rdi], ymm4       ; обновляем params
    
    add rdi, 32
    add rsi, 32
    add r12, 32
    dec r8
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end

.process_1:
    vmovss xmm2, [r12]        ; velocity
    vmovss xmm3, [rsi]        ; grad
    vmovss xmm4, [rdi]        ; param
    
    ; Сохраняем старый velocity
    vmovss xmm5, xmm2         ; velocity_prev
    
    ; velocity = momentum * velocity - lr * grad
    vmulss xmm2, xmm1, xmm2   ; momentum * velocity
    vmulss xmm6, xmm0, xmm3   ; lr * grad
    vsubss xmm2, xmm2, xmm6   ; новое velocity
    
    ; param = param - momentum * velocity_prev + (1 + momentum) * velocity
    vmulss xmm5, xmm1, xmm5   ; momentum * velocity_prev
    vsubss xmm4, xmm4, xmm5   ; param - momentum * velocity_prev
    
    vmovss xmm6, [one]
    vaddss xmm6, xmm6, xmm1   ; 1 + momentum
    vmulss xmm6, xmm6, xmm2   ; (1 + momentum) * velocity
    vaddss xmm4, xmm4, xmm6   ; final param
    
    ; Сохраняем результаты
    vmovss [r12], xmm2
    vmovss [rdi], xmm4
    
    add rdi, 4
    add rsi, 4
    add r12, 4
    dec rdx
    jnz .process_1

.end:
    pop r12
    vzeroupper
    ret

; SGD с L2 регуляризацией (weight decay)
; void sgd_with_weight_decay(float* params, float* grads, size_t len, float lr, float weight_decay)
; rdi = params, rsi = grads, rdx = len, xmm0 = lr, xmm1 = weight_decay
sgd_with_weight_decay:
    test rdx, rdx
    jz .end
    
    ; Дублируем коэффициенты
    vshufps xmm0, xmm0, xmm0, 0
    vinsertf128 ymm0, ymm0, xmm0, 1  ; lr
    vshufps xmm1, xmm1, xmm1, 0
    vinsertf128 ymm1, ymm1, xmm1, 1  ; weight_decay
    
    mov r8, rdx
    shr r8, 3
    jz .process_remainder

.process_8:
    vmovups ymm2, [rsi]       ; grads
    vmovups ymm3, [rdi]       ; params
    
    ; Добавляем weight decay к градиентам: grad = grad + weight_decay * param
    vmulps ymm4, ymm1, ymm3   ; weight_decay * param
    vaddps ymm2, ymm2, ymm4   ; grad + weight_decay * param
    
    ; param = param - lr * grad
    vmulps ymm2, ymm2, ymm0   ; lr * grad
    vsubps ymm3, ymm3, ymm2   ; param - lr * grad
    
    vmovups [rdi], ymm3
    
    add rdi, 32
    add rsi, 32
    dec r8
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end

.process_1:
    vmovss xmm2, [rsi]        ; grad
    vmovss xmm3, [rdi]        ; param
    
    ; grad = grad + weight_decay * param
    vmulss xmm4, xmm1, xmm3   ; weight_decay * param
    vaddss xmm2, xmm2, xmm4   ; grad + weight_decay * param
    
    ; param = param - lr * grad
    vmulss xmm2, xmm2, xmm0   ; lr * grad
    vsubss xmm3, xmm3, xmm2   ; param - lr * grad
    
    vmovss [rdi], xmm3
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

; Обновление learning rate (learning rate decay)
; void sgd_update_learning_rate(SGDOptimizer* optim, float new_lr)
; rdi = optim, xmm0 = new_lr
sgd_update_learning_rate:
    vmovss [rdi], xmm0
    ret

; Освобождение памяти оптимизатора
; void sgd_optimizer_free(SGDOptimizer* optim)
sgd_optimizer_free:
    test rdi, rdi
    jz .end
    
    ; Освобождаем velocity buffers если они есть
    mov rax, [velocity_weights]
    test rax, rax
    jz .no_velocity
    
    push rdi
    mov rdi, rax
    call free_asm
    pop rdi
    
    mov qword [velocity_weights], 0

.no_velocity:
    mov rax, [velocity_biases]
    test rax, rax
    jz .free_optim
    
    push rdi
    mov rdi, rax
    call free_asm
    pop rdi
    mov qword [velocity_biases], 0

.free_optim:
    call free_asm

.end:
    ret

section .data
align 16
one: dd 1.0, 1.0, 1.0, 1.0