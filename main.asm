%include "include/constants.inc"
%include "include/macros.inc"

section .data
align 16
    ; Данные для задачи XOR
    xor_inputs: 
        dd 0.0, 0.0
        dd 0.0, 1.0  
        dd 1.0, 0.0
        dd 1.0, 1.0
    xor_targets:
        dd 0.0
        dd 1.0
        dd 1.0
        dd 0.0
    
    epochs: dd 1000
    lr: dd 0.1
    
    ; Сообщения для вывода
    msg_epoch: db "Epoch: ", 0
    msg_loss: db " Loss: ", 0
    msg_newline: db 10, 0
    msg_training: db "Training XOR network...", 10, 0
    msg_results: db "Results:", 10, 0
    msg_input: db "Input: ", 0
    msg_output: db " Output: ", 0
    msg_target: db " Target: ", 0

section .bss
    layer1: resq 1
    layer2: resq 1
    output: resq 1

section .text
    global _start
    extern dense_layer_init, dense_layer_forward, dense_layer_backward
    extern dense_layer_update, mse_loss, mse_loss_derivative
    extern malloc_asm, free_asm

_start:
    ; Инициализация сети: 2 -> 4 -> 1
    mov rdi, 2  ; input_size
    mov rsi, 4  ; output_size  
    mov rdx, ACTIVATION_RELU
    call dense_layer_init
    mov [layer1], rax
    
    mov rdi, 4  ; input_size
    mov rsi, 1  ; output_size
    mov rdx, ACTIVATION_NONE  ; no activation (linear output)
    call dense_layer_init
    mov [layer2], rax
    
    ; Выделяем память для вывода
    mov rdi, 4  ; 1 float
    call malloc_asm
    mov [output], rax
    
    ; Выводим сообщение о начале обучения
    mov rsi, msg_training
    call print_string
    
    ; Цикл обучения
    mov ecx, [epochs]
.epoch_loop:
    push rcx
    
    ; Показываем прогресс каждые 100 эпох
    mov eax, [epochs]
    sub eax, ecx
    mov edx, 0
    mov ebx, 100
    div ebx
    test edx, edx
    jnz .forward_pass
    
    ; Выводим номер эпохи
    mov rsi, msg_epoch
    call print_string
    mov eax, [epochs]
    sub eax, ecx
    call print_int
    call print_newline

.forward_pass:
    mov r15, 0  ; индекс примера
    vxorps xmm15, xmm15, xmm15  ; аккумулятор потерь
    
.sample_loop:
    ; Прямое распространение через первый слой
    mov rax, [layer1]
    lea rsi, [xor_inputs + r15*8]  ; input (2 float)
    mov rdi, rax
    call dense_layer_forward
    
    ; Прямое распространение через второй слой
    mov rdi, [layer2]
    mov rsi, rax  ; выход первого слоя
    call dense_layer_forward
    mov [output], rax
    
    ; Вычисляем потери
    mov rdi, [output]
    lea rsi, [xor_targets + r15*4]  ; target
    mov rdx, 1  ; len
    call mse_loss
    vaddss xmm15, xmm15, xmm0
    
    ; Обратное распространение
    ; Вычисляем производную потерь
    mov rdi, [output]
    lea rsi, [xor_targets + r15*4]
    mov rdx, rdi  ; derivative (in-place)
    mov rcx, 1
    call mse_loss_derivative
    
    ; Backward через второй слой
    mov rdi, [layer2]
    mov rsi, [output]  ; doutput
    call dense_layer_backward
    
    ; Backward через первый слой
    ; dinput второго слоя становится doutput для первого
    mov rax, [layer2]
    mov rsi, [rax + 48]  ; dinput из второго слоя
    mov rdi, [layer1]
    call dense_layer_backward
    
    ; Обновляем веса
    vmovss xmm0, [lr]
    mov rdi, [layer1]
    call dense_layer_update
    
    vmovss xmm0, [lr]
    mov rdi, [layer2]
    call dense_layer_update
    
    inc r15
    cmp r15, 4
    jl .sample_loop
    
    pop rcx
    dec ecx
    jnz .epoch_loop
    
    ; Выводим финальные результаты
    mov rsi, msg_results
    call print_string
    
    mov r15, 0
.test_loop:
    mov rsi, msg_input
    call print_string
    
    ; Выводим вход
    mov eax, [xor_inputs + r15*8]
    call print_float
    mov eax, [xor_inputs + r15*8 + 4]
    call print_float
    
    ; Прямой проход
    mov rax, [layer1]
    lea rsi, [xor_inputs + r15*8]
    mov rdi, rax
    call dense_layer_forward
    
    mov rdi, [layer2]
    mov rsi, rax
    call dense_layer_forward
    
    ; Выводим результат
    mov rsi, msg_output
    call print_string
    mov eax, [rax]
    call print_float
    
    ; Выводим целевое значение
    mov rsi, msg_target
    call print_string
    mov eax, [xor_targets + r15*4]
    call print_float
    
    call print_newline
    
    inc r15
    cmp r15, 4
    jl .test_loop
    
    ; Завершение программы
    mov rax, 60
    xor rdi, rdi
    syscall

; Вспомогательные функции для вывода
print_string:
    mov rdx, 0
.length_loop:
    cmp byte [rsi + rdx], 0
    je .print
    inc rdx
    jmp .length_loop
.print:
    mov rax, 1
    mov rdi, 1
    syscall
    ret

print_newline:
    mov rax, 1
    mov rdi, 1
    mov rsi, msg_newline
    mov rdx, 1
    syscall
    ret

print_int:
    ; Упрощенная реализация - выводит только последнюю цифру
    add eax, '0'
    push rax
    mov rax, 1
    mov rdi, 1
    mov rsi, rsp
    mov rdx, 1
    syscall
    pop rax
    ret

print_float:
    ; Упрощенная реализация - выводит как целое
    push rax
    mov rax, 1
    mov rdi, 1
    mov rsi, rsp
    mov rdx, 4
    syscall
    pop rax
    ret