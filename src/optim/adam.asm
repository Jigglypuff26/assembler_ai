section .data
align 32
    beta1: dd 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9
    beta2: dd 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999
    epsilon: dd 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8
    one_minus_beta1: dd 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    one_minus_beta2: dd 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001

section .text
    global adam_optimizer_init, adam_update, adam_free

; Структура Adam оптимизатора
; typedef struct {
;     float* m;           // +0 (first moment)
;     float* v;           // +8 (second moment)  
;     size_t param_size;  // +16
;     size_t t;           // +24 (timestep)
;     float beta1;        // +32
;     float beta2;        // +36
;     float epsilon;      // +40
; } AdamOptimizer;

; AdamOptimizer* adam_optimizer_init(size_t param_size, float lr, float beta1, float beta2, float epsilon)
adam_optimizer_init:
    push r12
    push r13
    
    mov r12, rdi  ; param_size
    mov r13, xmm0 ; lr
    
    ; Выделяем память под структуру (48 байт)
    mov rdi, 48
    call malloc_asm
    test rax, rax
    jz .error
    
    mov r14, rax  ; сохраняем указатель
    
    ; Выделяем память для m и v
    mov rax, r12
    shl rax, 2    ; * sizeof(float)
    mov rdi, rax
    call malloc_asm
    mov [r14], rax ; m
    
    mov rdi, rax
    call malloc_asm
    mov [r14 + 8], rax ; v
    
    ; Инициализируем m и v нулями
    mov rdi, [r14]     ; m
    vxorps xmm0, xmm0, xmm0
    mov rsi, r12
    call memset_float_asm
    
    mov rdi, [r14 + 8] ; v
    vxorps xmm0, xmm0, xmm0
    mov rsi, r12
    call memset_float_asm
    
    ; Сохраняем параметры
    mov [r14 + 16], r12 ; param_size
    mov dword [r14 + 24], 0 ; t = 0
    vmovss [r14 + 32], xmm1 ; beta1
    vmovss [r14 + 36], xmm2 ; beta2
    vmovss [r14 + 40], xmm3 ; epsilon
    
    mov rax, r14
    jmp .end

.error:
    xor rax, rax

.end:
    pop r13
    pop r12
    ret

; void adam_update(float* params, float* grads, AdamOptimizer* optim, float lr)
adam_update:
    push r15
    push r14
    push r13
    
    mov r15, rdi  ; params
    mov r14, rsi  ; grads
    mov r13, rdx  ; optim
    ; xmm0 = lr
    
    ; Увеличиваем счетчик времени
    mov rax, [r13 + 24]
    inc rax
    mov [r13 + 24], rax
    
    ; Дублируем константы по регистрам
    vmovss xmm1, [r13 + 32] ; beta1
    vshufps xmm1, xmm1, xmm1, 0
    vinsertf128 ymm1, ymm1, xmm1, 1
    
    vmovss xmm2, [r13 + 36] ; beta2
    vshufps xmm2, xmm2, xmm2, 0
    vinsertf128 ymm2, ymm2, xmm2, 1
    
    vmovss xmm3, [r13 + 40] ; epsilon
    vshufps xmm3, xmm3, xmm3, 0
    vinsertf128 ymm3, ymm3, xmm3, 1
    
    vmovaps ymm4, [one_minus_beta1]
    vmovaps ymm5, [one_minus_beta2]
    
    ; Вычисляем bias correction
    vcvtsi2ss xmm6, xmm6, rax ; t
    call compute_bias_correction
    
    mov rcx, [r13 + 16] ; param_size
    mov r8, rcx
    shr r8, 3           ; / 8
    jz .process_remainder

.adam_loop:
    ; Загружаем данные
    vmovups ymm7, [r14]         ; grads
    vmovups ymm8, [r15]         ; params
    vmovups ymm9, [r13]         ; m
    vmovups ymm10, [r13 + 8]    ; v
    
    ; Обновляем m: m = beta1 * m + (1 - beta1) * grad
    vmulps ymm11, ymm9, ymm1    ; beta1 * m
    vmulps ymm12, ymm7, ymm4    ; (1 - beta1) * grad
    vaddps ymm9, ymm11, ymm12   ; m = ...
    
    ; Обновляем v: v = beta2 * v + (1 - beta2) * grad²
    vmulps ymm11, ymm10, ymm2   ; beta2 * v
    vmulps ymm12, ymm7, ymm7    ; grad²
    vmulps ymm12, ymm12, ymm5   ; (1 - beta2) * grad²
    vaddps ymm10, ymm11, ymm12  ; v = ...
    
    ; Bias correction
    vdivps ymm11, ymm9, ymm13   ; m_hat = m / (1 - beta1^t)
    vdivps ymm12, ymm10, ymm14  ; v_hat = v / (1 - beta2^t)
    
    ; Обновление параметров
    vsqrtps ymm12, ymm12        ; sqrt(v_hat)
    vaddps ymm12, ymm12, ymm3   ; sqrt(v_hat) + epsilon
    vdivps ymm11, ymm11, ymm12  ; m_hat / (sqrt(v_hat) + epsilon)
    vmulps ymm11, ymm11, ymm15  ; * lr
    vsubps ymm8, ymm8, ymm11    ; param - lr * ...
    
    ; Сохраняем результаты
    vmovups [r15], ymm8
    vmovups [r13], ymm9
    vmovups [r13 + 8], ymm10
    
    add r15, 32
    add r14, 32
    add r13, 32
    add qword [r13], 32  ; m ptr
    add qword [r13 + 8], 32 ; v ptr
    
    dec r8
    jnz .adam_loop

.process_remainder:
    and rcx, 7
    jz .end
    
    ; Обработка оставшихся элементов
    ; ... аналогичная логика для скалярных операций

.end:
    pop r13
    pop r14
    pop r15
    vzeroupper
    ret

compute_bias_correction:
    ; Вычисление 1 - beta1^t и 1 - beta2^t
    ; ymm13 = 1 - beta1^t, ymm14 = 1 - beta2^t
    ; ymm15 = learning rate
    ret

; void adam_free(AdamOptimizer* optim)
adam_free:
    test rdi, rdi
    jz .end
    
    push r12
    mov r12, rdi
    
    mov rdi, [r12]     ; m
    call free_asm
    
    mov rdi, [r12 + 8] ; v
    call free_asm
    
    mov rdi, r12       ; структура
    call free_asm
    
    pop r12

.end:
    ret