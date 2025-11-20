section .data
    magic_header: db "ASM_AIv1", 0
    layer_type_dense: db "DENSE", 0
    layer_type_conv: db "CONV2D", 0
    layer_type_lstm: db "LSTM", 0

section .text
    global save_model, load_model
    global save_weights, load_weights

; void save_model(char* filename, void** layers, size_t num_layers)
save_model:
    push r15
    push r14
    push r13
    
    mov r15, rdi  ; filename
    mov r14, rsi  ; layers
    mov r13, rdx  ; num_layers
    
    ; Открываем файл для записи
    mov rax, 2          ; sys_open
    mov rdi, r15
    mov rsi, 0x241      ; O_WRONLY|O_CREAT|O_TRUNC
    mov rdx, 0644o      ; permissions
    syscall
    
    cmp rax, 0
    jl .error
    
    mov r12, rax        ; file descriptor
    
    ; Записываем заголовок
    mov rax, 1          ; sys_write
    mov rdi, r12
    mov rsi, magic_header
    mov rdx, 8
    syscall
    
    ; Записываем количество слоев
    mov rax, 1
    mov rdi, r12
    mov rsi, r13
    mov rdx, 8
    syscall
    
    ; Записываем каждый слой
    xor rbx, rbx
.layer_loop:
    cmp rbx, r13
    jge .end_layers
    
    mov rdi, [r14 + rbx*8] ; слой
    mov rsi, r12           ; файл
    call serialize_layer
    
    inc rbx
    jmp .layer_loop

.end_layers:
    ; Закрываем файл
    mov rax, 3          ; sys_close
    mov rdi, r12
    syscall
    
    jmp .end

.error:
    mov rax, -1

.end:
    pop r13
    pop r14
    pop r15
    ret

serialize_layer:
    ; Сериализация одного слоя
    ; Определяем тип слоя и вызываем соответствующую функцию
    ret

serialize_dense_layer:
    ; Сериализация Dense слоя
    ret

serialize_conv_layer:
    ; Сериализация Conv2D слоя
    ret

serialize_lstm_layer:
    ; Сериализация LSTM слоя
    ret

; void* load_model(char* filename)
load_model:
    ; Загрузка модели из файла
    ret

; void save_weights(char* filename, void** layers, size_t num_layers)
save_weights:
    ; Сохранение только весов
    ret

; void load_weights(char* filename, void** layers, size_t num_layers)
load_weights:
    ; Загрузка весов в существующие слои
    ret