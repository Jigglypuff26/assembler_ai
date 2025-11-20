section .text
    global mse_loss, mse_loss_derivative

; float mse_loss(float* predicted, float* target, size_t len)
; rdi = predicted, rsi = target, rdx = len
mse_loss:
    test rdx, rdx
    jz .zero
    
    vxorps ymm7, ymm7, ymm7  ; аккумулятор
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]  ; predicted
    vmovups ymm1, [rsi]  ; target
    vsubps ymm2, ymm0, ymm1  ; diff
    vmulps ymm2, ymm2, ymm2  ; diff²
    vaddps ymm7, ymm7, ymm2
    
    add rdi, 32
    add rsi, 32
    dec rcx
    jnz .process_8

    ; Горизонтальное суммирование
    vhaddps ymm7, ymm7, ymm7
    vhaddps ymm7, ymm7, ymm7
    vextractf128 xmm0, ymm7, 1
    vaddps xmm7, xmm7, xmm0

.process_remainder:
    and rdx, 7
    jz .finalize
    
.process_1:
    vmovss xmm0, [rdi]
    vmovss xmm1, [rsi]
    vsubss xmm2, xmm0, xmm1
    vmulss xmm2, xmm2, xmm2
    vaddss xmm7, xmm7, xmm2
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.finalize:
    ; Делим на количество элементов
    vcvtsi2ss xmm1, xmm1, [rsp+8]  ; len из параметра
    vdivss xmm0, xmm7, xmm1
    ret

.zero:
    vxorps xmm0, xmm0, xmm0
    ret

; void mse_loss_derivative(float* predicted, float* target, float* derivative, size_t len)
; derivative = 2 * (predicted - target) / len
mse_loss_derivative:
    test rcx, rdx
    jz .end
    
    ; Вычисляем масштаб: 2 / len
    mov eax, 2
    vcvtsi2ss xmm2, xmm2, eax
    vcvtsi2ss xmm3, xmm3, rcx
    vdivss xmm2, xmm2, xmm3
    vshufps xmm2, xmm2, xmm2, 0
    vinsertf128 ymm2, ymm2, xmm2, 1
    
    mov r8, rcx
    shr r8, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]  ; predicted
    vmovups ymm1, [rsi]  ; target
    vsubps ymm3, ymm0, ymm1  ; predicted - target
    vmulps ymm3, ymm3, ymm2  ; * (2 / len)
    vmovups [rdx], ymm3
    
    add rdi, 32
    add rsi, 32
    add rdx, 32
    dec r8
    jnz .process_8

.process_remainder:
    and rcx, 7
    jz .end

.process_1:
    vmovss xmm0, [rdi]
    vmovss xmm1, [rsi]
    vsubss xmm3, xmm0, xmm1
    vmulss xmm3, xmm3, xmm2
    vmovss [rdx], xmm3
    
    add rdi, 4
    add rsi, 4
    add rdx, 4
    dec rcx
    jnz .process_1

.end:
    vzeroupper
    ret