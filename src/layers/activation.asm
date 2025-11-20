section .data
align 32
    ; Константы для математических функций
    one:        dd 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    half:       dd 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    neg_one:    dd -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    zero:       dd 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    exp_coeff:  dd 1.0, 1.0, 0.5, 0.1666667, 0.04166667, 0.00833333  ; Ряд Тейлора для exp

section .text
    global sigmoid_asm, sigmoid_derivative_asm
    global tanh_asm, tanh_derivative_asm
    global softmax_asm

; void sigmoid_asm(float* input, float* output, size_t len)
; rdi = input, rsi = output, rdx = len
sigmoid_asm:
    test rdx, rdx
    jz .end
    
    vmovaps ymm4, [one]      ; 1.0
    vmovaps ymm5, [neg_one]  ; -1.0
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]      ; загружаем input
    vmulps ymm0, ymm0, ymm5  ; -input
    call exp_approx_asm      ; exp(-input)
    
    vaddps ymm1, ymm0, ymm4  ; 1 + exp(-input)
    vdivps ymm2, ymm4, ymm1  ; 1 / (1 + exp(-input))
    
    vmovups [rsi], ymm2
    
    add rdi, 32
    add rsi, 32
    dec rcx
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end
    
.process_1:
    vmovss xmm0, [rdi]       ; input
    vmulss xmm0, xmm0, [neg_one] ; -input
    call exp_approx_scalar   ; exp(-input)
    
    vmovss xmm1, [one]       ; 1.0
    vaddss xmm2, xmm0, xmm1  ; 1 + exp(-input)
    vdivss xmm3, xmm1, xmm2  ; 1 / (1 + exp(-input))
    
    vmovss [rsi], xmm3
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

; void sigmoid_derivative_asm(float* input, float* output, size_t len)
; output = sigmoid(input) * (1 - sigmoid(input))
sigmoid_derivative_asm:
    test rdx, rdx
    jz .end
    
    vmovaps ymm4, [one]
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    ; Сначала вычисляем sigmoid
    vmovups ymm0, [rdi]
    vmovaps ymm5, [neg_one]
    vmulps ymm0, ymm0, ymm5
    call exp_approx_asm      ; exp(-input)
    
    vaddps ymm1, ymm0, ymm4  ; 1 + exp(-input)
    vdivps ymm2, ymm4, ymm1  ; sigmoid = 1 / (1 + exp(-input))
    
    ; Вычисляем производную: sigmoid * (1 - sigmoid)
    vsubps ymm3, ymm4, ymm2  ; 1 - sigmoid
    vmulps ymm2, ymm2, ymm3  ; sigmoid * (1 - sigmoid)
    
    vmovups [rsi], ymm2
    
    add rdi, 32
    add rsi, 32
    dec rcx
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end
    
.process_1:
    vmovss xmm0, [rdi]
    vmulss xmm0, xmm0, [neg_one]
    call exp_approx_scalar
    
    vmovss xmm1, [one]
    vaddss xmm2, xmm0, xmm1
    vdivss xmm3, xmm1, xmm2  ; sigmoid
    
    vsubss xmm4, xmm1, xmm3  ; 1 - sigmoid
    vmulss xmm3, xmm3, xmm4  ; производная
    
    vmovss [rsi], xmm3
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

; void tanh_asm(float* input, float* output, size_t len)
tanh_asm:
    test rdx, rdx
    jz .end
    
    vmovaps ymm4, [one]
    vmovaps ymm5, [neg_one]
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]      ; input
    
    ; tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    vmulps ymm0, ymm0, ymm4  ; 2x (пока просто x, потом умножим)
    vmulps ymm0, ymm0, ymm4  ; 2x
    
    ; exp(2x)
    call exp_approx_asm      ; ymm0 = exp(2x)
    
    vmovaps ymm1, ymm0       ; копируем exp(2x)
    vsubps ymm2, ymm0, ymm4  ; exp(2x) - 1
    vaddps ymm3, ymm1, ymm4  ; exp(2x) + 1
    vdivps ymm0, ymm2, ymm3  ; (exp(2x) - 1) / (exp(2x) + 1)
    
    vmovups [rsi], ymm0
    
    add rdi, 32
    add rsi, 32
    dec rcx
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end
    
.process_1:
    vmovss xmm0, [rdi]       ; x
    vaddss xmm0, xmm0, xmm0  ; 2x
    
    call exp_approx_scalar   ; exp(2x)
    
    vmovss xmm1, [one]
    vsubss xmm2, xmm0, xmm1  ; exp(2x) - 1
    vaddss xmm3, xmm0, xmm1  ; exp(2x) + 1
    vdivss xmm0, xmm2, xmm3  ; tanh(x)
    
    vmovss [rsi], xmm0
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

; void tanh_derivative_asm(float* input, float* output, size_t len)
; derivative = 1 - tanh²(x)
tanh_derivative_asm:
    test rdx, rdx
    jz .end
    
    vmovaps ymm4, [one]
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    ; Сначала вычисляем tanh
    vmovups ymm0, [rdi]
    
    ; Вычисляем tanh (упрощенная версия)
    vmovaps ymm5, ymm0
    vmulps ymm5, ymm5, ymm4
    vmulps ymm5, ymm5, ymm4  ; 2x
    call exp_approx_asm      ; exp(2x)
    
    vmovaps ymm1, ymm0
    vsubps ymm2, ymm0, ymm4
    vaddps ymm3, ymm1, ymm4
    vdivps ymm0, ymm2, ymm3  ; tanh(x)
    
    ; Вычисляем производную: 1 - tanh²(x)
    vmulps ymm1, ymm0, ymm0  ; tanh²(x)
    vsubps ymm2, ymm4, ymm1  ; 1 - tanh²(x)
    
    vmovups [rsi], ymm2
    
    add rdi, 32
    add rsi, 32
    dec rcx
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end
    
.process_1:
    vmovss xmm0, [rdi]
    vaddss xmm0, xmm0, xmm0  ; 2x
    call exp_approx_scalar   ; exp(2x)
    
    vmovss xmm1, [one]
    vsubss xmm2, xmm0, xmm1
    vaddss xmm3, xmm0, xmm1
    vdivss xmm0, xmm2, xmm3  ; tanh(x)
    
    vmulss xmm1, xmm0, xmm0  ; tanh²(x)
    vsubss xmm2, [one], xmm1 ; 1 - tanh²(x)
    
    vmovss [rsi], xmm2
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

; void softmax_asm(float* input, float* output, size_t len)
softmax_asm:
    push r12
    push r13
    
    mov r12, rdi  ; input
    mov r13, rsi  ; output
    mov r14, rdx  ; len
    
    test rdx, rdx
    jz .end
    
    ; Шаг 1: Находим максимум (для численной стабильности)
    vmovss xmm7, [r12]  ; max_value = input[0]
    mov rcx, 1
.find_max:
    cmp rcx, r14
    jge .found_max
    
    vmovss xmm0, [r12 + rcx*4]
    vmaxss xmm7, xmm7, xmm0
    inc rcx
    jmp .find_max

.found_max:
    ; Дублируем максимум по всему регистру
    vshufps xmm7, xmm7, xmm7, 0
    
    ; Шаг 2: Вычисляем exp(input - max_value)
    vxorps ymm6, ymm6, ymm6  ; sum = 0
    mov rcx, 0
    mov r8, r14
    shr r8, 3
    jz .exp_remainder

.exp_loop_8:
    vmovups ymm0, [r12 + rcx*4]  ; input
    vsubps ymm0, ymm0, ymm7      ; input - max_value
    
    call exp_approx_asm          ; exp(input - max_value)
    
    vmovups [r13 + rcx*4], ymm0  ; сохраняем exp
    vaddps ymm6, ymm6, ymm0      ; sum += exp
    
    add rcx, 8
    dec r8
    jnz .exp_loop_8

.exp_remainder:
    mov r8, r14
    and r8, 7
    jz .normalize
    
.exp_loop_1:
    vmovss xmm0, [r12 + rcx*4]
    vsubss xmm0, xmm0, xmm7
    
    call exp_approx_scalar       ; exp(input - max_value)
    
    vmovss [r13 + rcx*4], xmm0
    vaddss xmm6, xmm6, xmm0
    
    inc rcx
    dec r8
    jnz .exp_loop_1

.normalize:
    ; Горизонтальное суммирование для ymm6
    vhaddps ymm6, ymm6, ymm6
    vhaddps ymm6, ymm6, ymm6
    vextractf128 xmm0, ymm6, 1
    vaddps xmm6, xmm6, xmm0
    
    ; Шаг 3: Делим каждый элемент на сумму
    mov rcx, 0
    mov r8, r14
    shr r8, 3
    jz .norm_remainder

.norm_loop_8:
    vmovups ymm0, [r13 + rcx*4]  ; exp values
    vdivps ymm0, ymm0, ymm6      ; / sum
    vmovups [r13 + rcx*4], ymm0
    
    add rcx, 8
    dec r8
    jnz .norm_loop_8

.norm_remainder:
    mov r8, r14
    and r8, 7
    jz .end
    
.norm_loop_1:
    vmovss xmm0, [r13 + rcx*4]
    vdivss xmm0, xmm0, xmm6
    vmovss [r13 + rcx*4], xmm0
    
    inc rcx
    dec r8
    jnz .norm_loop_1

.end:
    pop r13
    pop r12
    vzeroupper
    ret

; ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

; Приближенное вычисление exp для вектора (ymm0 -> ymm0)
exp_approx_asm:
    ; Используем приближение рядом Тейлора: exp(x) ≈ 1 + x + x²/2 + x³/6
    vmovaps ymm1, ymm0              ; x
    vmovaps ymm2, [one]             ; 1.0
    
    ; 1 + x
    vaddps ymm3, ymm2, ymm1
    
    ; x²
    vmulps ymm4, ymm1, ymm1
    vmulps ymm4, ymm4, [half]       ; x²/2
    vaddps ymm3, ymm3, ymm4
    
    ; x³
    vmulps ymm4, ymm1, ymm1
    vmulps ymm4, ymm4, ymm1
    vmulps ymm4, ymm4, [rel third]  ; x³/6
    vaddps ymm0, ymm3, ymm4
    
    ret

; Приближенное вычисление exp для скаляра (xmm0 -> xmm0)
exp_approx_scalar:
    vmovaps xmm1, xmm0              ; x
    vmovss xmm2, [one]              ; 1.0
    
    ; 1 + x
    vaddss xmm3, xmm2, xmm1
    
    ; x²
    vmulss xmm4, xmm1, xmm1
    vmulss xmm4, xmm4, [half]       ; x²/2
    vaddss xmm3, xmm3, xmm4
    
    ; x³
    vmulss xmm4, xmm1, xmm1
    vmulss xmm4, xmm4, xmm1
    vmulss xmm4, xmm4, [rel third]  ; x³/6
    vaddss xmm0, xmm3, xmm4
    
    ret

section .data
align 16
third: dd 0.1666667, 0.1666667, 0.1666667, 0.1666667