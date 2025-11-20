section .text
    global malloc_asm, free_asm, memcpy_asm, memset_float_asm

; void* malloc_asm(size_t size)
; rdi = размер в байтах
malloc_asm:
    push rdi
    push rsi
    push rdx
    
    ; Вызов системного brk для выделения памяти
    mov rax, 12         ; sys_brk
    xor rdi, rdi        ; получить текущий brk
    syscall
    
    mov rdx, rax        ; сохраняем текущий brk
    add rax, [rsp+24]   ; добавляем запрошенный размер
    mov rdi, rax
    mov rax, 12         ; sys_brk с новым значением
    syscall
    
    cmp rax, rdi        ; проверяем успешность
    jne .error
    
    mov rax, rdx        ; возвращаем указатель на начало
    jmp .end
    
.error:
    xor rax, rax        ; возвращаем NULL при ошибке
    
.end:
    pop rdx
    pop rsi
    pop rdi
    ret

; void free_asm(void* ptr)
; В упрощенной реализации просто сохраняем для совместимости
free_asm:
    ret

; void memcpy_asm(void* dest, void* src, size_t n)
; rdi = назначение, rsi = источник, rdx = количество байт
memcpy_asm:
    test rdx, rdx
    jz .end
    
    mov rcx, rdx
    shr rcx, 3          ; копируем по 8 байт
    jz .remainder
    
.loop_8:
    mov rax, [rsi]
    mov [rdi], rax
    add rsi, 8
    add rdi, 8
    dec rcx
    jnz .loop_8

.remainder:
    mov rcx, rdx
    and rcx, 7          ; оставшиеся байты
    jz .end
    
.loop_1:
    mov al, [rsi]
    mov [rdi], al
    inc rsi
    inc rdi
    dec rcx
    jnz .loop_1

.end:
    ret

; void memset_float_asm(float* ptr, float value, size_t count)
; rdi = указатель, xmm0 = значение, rsi = количество
memset_float_asm:
    test rsi, rsi
    jz .end
    
    ; Дублируем значение по всему регистру ymm
    vshufps xmm0, xmm0, xmm0, 0
    vinsertf128 ymm0, ymm0, xmm0, 1
    
    ; Обрабатываем по 8 элементов за итерацию
    mov rcx, rsi
    shr rcx, 3
    jz .process_remainder
    
.process_8:
    vmovups [rdi], ymm0
    add rdi, 32         ; 8 * 4 байта
    dec rcx
    jnz .process_8

.process_remainder:
    and rsi, 7
    jz .end
    
    ; Обрабатываем оставшиеся элементы
    mov rcx, rsi
.process_1:
    vmovss [rdi], xmm0
    add rdi, 4
    dec rcx
    jnz .process_1

.end:
    vzeroupper
    ret