section .text
    global matrix_multiply_asm, matrix_transpose_asm
    global matrix_multiply_simple, matrix_multiply_transpose

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
    
    ; Вызываем strided_dot_product для вычисления скалярного произведения
    push rdi
    push rsi
    push rcx
    push r8
    push r9
    
    ; Подготовка параметров для strided_dot_product
    mov rdi, rax         ; A[i]
    mov rsi, r13         ; B[0]
    add rsi, rdx         ; B[0][j]
    mov rdx, r8          ; длина = colsA
    mov rcx, r9          ; шаг для B = colsB
    
    call strided_dot_product
    
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

; Упрощенное матричное умножение для векторов
; void matrix_multiply_simple(float* A, float* B, float* C, size_t rows, size_t cols)
; A[rows][cols] * B[cols] = C[rows]
matrix_multiply_simple:
    push r15
    push r14
    
    mov r15, rdi  ; A
    mov r14, rsi  ; B
    
    xor r10, r10  ; i = 0
.rows_loop:
    cmp r10, rcx
    jge .end_rows
    
    ; Вычисляем C[i] = dot(A[i], B)
    mov rdi, r15  ; &A[i][0]
    mov rsi, r14  ; B
    mov rdx, r8   ; cols
    call vec_dot_asm
    
    ; Сохраняем результат в C[i]
    mov [rdx], eax
    add rdx, 4
    
    ; Переходим к следующей строке A
    mov rax, r8
    shl rax, 2
    add r15, rax
    
    inc r10
    jmp .rows_loop

.end_rows:
    pop r14
    pop r15
    ret

; void matrix_multiply_transpose(float* A, float* B, float* C, size_t rows, size_t colsA, size_t colsB)
; A^T * B = C, где A[colsA][rows], B[colsB], C[rows]
matrix_multiply_transpose:
    push r15
    push r14
    push r13
    push r12
    
    mov r15, rdi  ; A
    mov r14, rsi  ; B  
    mov r13, rdx  ; C
    mov r12, rcx  ; rows
    
    xor r10, r10  ; i = 0
.rows_loop:
    cmp r10, r12
    jge .end_rows
    
    ; C[i] = dot(A[:,i], B) - но A хранится по строкам
    mov rdi, r15        ; A[0] + i (транспонированный доступ)
    mov rsi, r14        ; B
    mov rdx, r8         ; colsA (длина вектора)
    call vec_dot_asm
    
    ; Сохраняем результат в C[i]
    mov [r13 + r10*4], eax
    
    ; Переходим к следующей "строке" в транспонированной A
    add r15, 4          ; следующий элемент в столбце
    
    inc r10
    jmp .rows_loop

.end_rows:
    pop r12
    pop r13
    pop r14
    pop r15
    ret

; Вспомогательная функция для скалярного произведения с шагом
; float strided_dot_product(float* a, float* b, size_t len, size_t stride)
strided_dot_product:
    test rdx, rdx
    jz .zero
    
    vxorps ymm7, ymm7, ymm7
    shl rcx, 2           ; stride в байтах
    
    mov r8, rdx
    shr r8, 3
    jz .process_remainder

.process_8:
    ; Загружаем 8 элементов из a (последовательно)
    vmovups ymm0, [rdi]
    add rdi, 32
    
    ; Загружаем 8 элементов из b с шагом
    vmovss xmm1, [rsi]
    lea rax, [rsi + rcx]
    vmovss xmm2, [rax]
    lea rax, [rax + rcx]
    vmovss xmm3, [rax]
    lea rax, [rax + rcx]
    vmovss xmm4, [rax]
    
    ; Собираем в один регистр
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

; void matrix_transpose_asm(float* A, float* B, size_t rows, size_t cols)
matrix_transpose_asm:
    ; Простая реализация транспонирования
    push r15
    push r14
    
    mov r15, rdi  ; A
    mov r14, rsi  ; B
    
    xor r10, r10  ; i = 0
.outer_loop:
    cmp r10, rcx
    jge .end_outer
    
    xor r11, r11  ; j = 0
.inner_loop:
    cmp r11, r8
    jge .end_inner
    
    ; B[j][i] = A[i][j]
    mov rax, r10
    mul r8
    add rax, r11
    shl rax, 2
    vmovss xmm0, [r15 + rax]
    
    mov rax, r11
    mul rcx
    add rax, r10
    shl rax, 2
    vmovss [r14 + rax], xmm0
    
    inc r11
    jmp .inner_loop

.end_inner:
    inc r10
    jmp .outer_loop

.end_outer:
    pop r14
    pop r15
    ret