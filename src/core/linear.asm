section .text
    global matrix_multiply_asm, matrix_transpose_asm

; void matrix_multiply_asm(float* A, float* B, float* C, 
;                         size_t rowsA, size_t colsA, size_t colsB)
; rdi = A, rsi = B, rdx = C, rcx = rowsA, r8 = colsA, r9 = colsB
matrix_multiply_asm:
    push r12
    push r13
    push r14
    push r15
    
    ; r12 = A, r13 = B, r14 = C
    mov r12, rdi
    mov r13, rsi  
    mov r14, rdx
    
    ; Внешний цикл по строкам A (i)
    xor r15, r15          ; i = 0
.outer_loop:
    cmp r15, rcx
    jge .end_outer
    
    ; Внутренний цикл по столбцам B (j)  
    xor rdi, rdi          ; j = 0
.middle_loop:
    cmp rdi, r9
    jge .end_middle
    
    ; Вычисляем C[i][j] = dot(A[i], B[:,j])
    ; rsi = указатель на строку i в A
    mov rax, r15
    mul r8                ; rax = i * colsA
    shl rax, 2           ; умножаем на sizeof(float)
    add rax, r12         ; rax = &A[i][0]
    
    ; rdx = указатель на столбец j в B
    mov rdx, rdi
    shl rdx, 2           ; rdx = j * sizeof(float)
    
    ; Вызываем vec_dot для вычисления скалярного произведения
    push rdi
    push rsi
    push rcx
    push r8
    push r9
    
    ; Подготовка параметров для vec_dot
    mov rdi, rax         ; A[i]
    mov rsi, r13         ; B[0]
    add rsi, rdx         ; B[0][j]
    mov rdx, r8          ; длина = colsA
    mov rcx, r9          ; шаг для B = colsB
    
    call matrix_dot_product
    
    ; Сохраняем результат в C[i][j]
    ; Вычисляем индекс в C: i * colsB + j
    mov rax, r15
    mul r9
    add rax, [rsp+8]     ; + j
    shl rax, 2
    add rax, r14
    vmovss [rax], xmm0
    
    pop r9
    pop r8
    pop rcx
    pop rsi
    pop rdi
    
    inc rdi              ; j++
    jmp .middle_loop

.end_middle:
    inc r15              ; i++
    jmp .outer_loop

.end_outer:
    pop r15
    pop r14
    pop r13
    pop r12
    ret

; Вспомогательная функция для скалярного произведения с шагом
; float matrix_dot_product(float* a, float* b, size_t len, size_t stride)
; rdi = a, rsi = b, rdx = len, rcx = stride
matrix_dot_product:
    test rdx, rdx
    jz .zero
    
    vxorps ymm7, ymm7, ymm7
    shl rcx, 2           ; stride в байтах
    
    mov r8, rdx
    shr r8, 3
    jz .process_remainder

.process_8:
    vmovups ymm0, [rdi]  ; загружаем 8 элементов из a
    
    ; Загружаем 8 элементов из b с шагом
    vmovss xmm1, [rsi]
    lea rax, [rsi + rcx]
    vmovss xmm2, [rax]
    lea rax, [rax + rcx]
    vmovss xmm3, [rax]
    lea rax, [rax + rcx]
    vmovss xmm4, [rax]
    
    ; Собираем в один регистр (упрощенная версия)
    vinsertps xmm1, xmm1, xmm2, 0x10
    vinsertps xmm1, xmm1, xmm3, 0x20
    vinsertps xmm1, xmm1, xmm4, 0x30
    
    lea rax, [rax + rcx]
    vmovss xmm2, [rax]
    lea rax, [rax + rcx]
    vmovss xmm3, [rax]
    lea rax, [rax + rcx]
    vmovss xmm4, [rax]
    lea rax, [rax + rcx]
    vmovss xmm5, [rax]
    
    vinsertps xmm2, xmm2, xmm3, 0x10
    vinsertps xmm2, xmm2, xmm4, 0x20
    vinsertps xmm2, xmm2, xmm5, 0x30
    
    vinsertf128 ymm1, ymm1, xmm2, 1
    
    vmulps ymm0, ymm0, ymm1
    vaddps ymm7, ymm7, ymm0
    
    add rdi, 32
    add rsi, 32
    dec r8
    jnz .process_8

    ; Горизонтальное суммирование
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
    add rsi, rcx
    dec rdx
    jnz .process_1

.end:
    vmovss xmm0, xmm7
    vzeroupper
    ret

.zero:
    vxorps xmm0, xmm0, xmm0
    ret