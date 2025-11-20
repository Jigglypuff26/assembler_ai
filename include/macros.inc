; Макрос для выравнивания стека
%macro ALIGN_STACK 1
    push rbp
    mov rbp, rsp
    and rsp, -%1
%endmacro

; Макрос для восстановления стека
%macro RESTORE_STACK 0
    mov rsp, rbp
    pop rbp
%endmacro

; Макрос для отладки - вывод числа float
%macro DEBUG_FLOAT 1
    push rax
    push rdi
    push rsi
    push rdx
    sub rsp, 8
    movss [rsp], %1
    call debug_print_float
    add rsp, 8
    pop rdx
    pop rsi
    pop rdi
    pop rax
%endmacro