section .text
    global vec_add_asm, vec_mul_asm, vec_dot_asm
    global relu_asm, relu_derivative_asm
    global sigmoid_asm, sigmoid_derivative_asm

; void vec_add_asm(float* a, float* b, float* result, size_t len)
; rdi = a, rsi = b, rdx = result, rcx = len
vec_add_asm:
    test rcx, rcx
    jz .end
    
    ; Обрабатываем по 8 элементов с AVX
    mov r8, rcx
    shr r8, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]
    vmovups ymm1, [rsi]
    vaddps ymm2, ymm0, ymm1
    vmovups [rdx], ymm2
    
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
    vaddss xmm2, xmm0, xmm1
    vmovss [rdx], xmm2
    
    add rdi, 4
    add rsi, 4
    add rdx, 4
    dec rcx
    jnz .process_1

.end:
    vzeroupper
    ret

; void vec_mul_asm(float* a, float* b, float* result, size_t len)
; rdi = a, rsi = b, rdx = result, rcx = len  
vec_mul_asm:
    test rcx, rcx
    jz .end
    
    mov r8, rcx
    shr r8, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]
    vmovups ymm1, [rsi]
    vmulps ymm2, ymm0, ymm1
    vmovups [rdx], ymm2
    
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
    vmulss xmm2, xmm0, xmm1
    vmovss [rdx], xmm2
    
    add rdi, 4
    add rsi, 4
    add rdx, 4
    dec rcx
    jnz .process_1

.end:
    vzeroupper
    ret

; float vec_dot_asm(float* a, float* b, size_t len)
; rdi = a, rsi = b, rdx = len
vec_dot_asm:
    test rdx, rdx
    jz .zero
    
    vxorps ymm7, ymm7, ymm7  ; аккумулятор = 0
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]
    vmovups ymm1, [rsi]
    vmulps ymm2, ymm0, ymm1
    vaddps ymm7, ymm7, ymm2
    
    add rdi, 32
    add rsi, 32
    dec rcx
    jnz .process_8

    ; Горизонтальное суммирование ymm7
    vhaddps ymm7, ymm7, ymm7
    vhaddps ymm7, ymm7, ymm7
    vextractf128 xmm0, ymm7, 1
    vaddps xmm7, xmm7, xmm0

.process_remainder:
    and rdx, 7
    jz .end
    
.process_1:
    vmovss xmm0, [rdi]
    vmovss xmm1, [rsi]
    vmulss xmm2, xmm0, xmm1
    vaddss xmm7, xmm7, xmm2
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vmovss xmm0, xmm7
    vzeroupper
    ret

.zero:
    vxorps xmm0, xmm0, xmm0
    ret

; void relu_asm(float* input, float* output, size_t len)
; rdi = input, rsi = output, rdx = len
relu_asm:
    test rdx, rdx
    jz .end
    
    vxorps xmm1, xmm1, xmm1  ; 0.0
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]
    vmaxps ymm2, ymm0, ymm1  ; max(x, 0)
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
    vmaxss xmm2, xmm0, xmm1
    vmovss [rsi], xmm2
    
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

; void relu_derivative_asm(float* input, float* output, size_t len)
; rdi = input, rsi = output, rdx = len
relu_derivative_asm:
    test rdx, rdx
    jz .end
    
    vxorps xmm1, xmm1, xmm1  ; 0.0
    vmovaps xmm2, [rel_one]   ; 1.0
    
    mov rcx, rdx
    shr rcx, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]
    vcmpgtps ymm3, ymm0, ymm1  ; маска: x > 0.0
    vblendvps ymm4, ymm1, ymm2, ymm3  ; 1.0 если x > 0, иначе 0.0
    vmovups [rsi], ymm4
    
    add rdi, 32
    add rsi, 32
    dec rcx
    jnz .process_8

.process_remainder:
    and rdx, 7
    jz .end
    
.process_1:
    vmovss xmm0, [rdi]
    vcomiss xmm0, xmm1
    jbe .zero
    vmovss [rsi], xmm2
    jmp .next
.zero:
    vmovss [rsi], xmm1
.next:
    add rdi, 4
    add rsi, 4
    dec rdx
    jnz .process_1

.end:
    vzeroupper
    ret

section .data
align 16
rel_one: dd 1.0, 1.0, 1.0, 1.0