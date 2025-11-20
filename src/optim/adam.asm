section .data
align 32
    beta1: dd 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9
    beta2: dd 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999
    epsilon: dd 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8

section .text
    global adam_optimizer_init, adam_update

; void adam_update(float* params, float* grads, float* m, float* v,
;                 size_t len, float lr, size_t t)
adam_update:
    ; m = beta1 * m + (1 - beta1) * grad
    ; v = beta2 * v + (1 - beta2) * grad²
    ; m_hat = m / (1 - beta1^t)
    ; v_hat = v / (1 - beta2^t) 
    ; param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    vmovaps ymm4, [beta1]
    vmovaps ymm5, [beta2]
    vmovaps ymm6, [epsilon]
    
    mov rcx, len
.adam_loop:
    vmovups ymm0, [m]           ; m
    vmovups ymm1, [v]           ; v
    vmovups ymm2, [grads]       ; grad
    vmovups ymm3, [params]      ; param
    
    ; Обновление m и v
    vmulps ymm7, ymm2, [one_minus_beta1]
    vmulps ymm0, ymm0, ymm4
    vaddps ymm0, ymm0, ymm7     ; m = beta1*m + (1-beta1)*grad
    
    vmulps ymm2, ymm2, ymm2     ; grad²
    vmulps ymm7, ymm2, [one_minus_beta2]
    vmulps ymm1, ymm1, ymm5
    vaddps ymm1, ymm1, ymm7     ; v = beta2*v + (1-beta2)*grad²
    
    ; Коррекция bias
    call compute_bias_correction
    
    ; Обновление параметров
    vdivps ymm7, ymm0, ymm8     ; m_hat / (sqrt(v_hat) + epsilon)
    vmulps ymm7, ymm7, ymm9     ; * lr
    vsubps ymm3, ymm3, ymm7     ; param - lr * ...
    
    vmovups [params], ymm3
    vmovups [m], ymm0
    vmovups [v], ymm1
    
    add params, 32
    add grads, 32
    add m, 32
    add v, 32
    
    sub rcx, 8
    jg .adam_loop
    
    ret