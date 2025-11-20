section .text
    global save_model, load_model
    global save_weights, load_weights

; void save_model(char* filename, void** layers, size_t num_layers)
save_model:
    ; Открытие файла
    mov rax, 2          ; sys_open
    mov rdi, filename
    mov rsi, 0x241      ; O_WRONLY|O_CREAT|O_TRUNC
    mov rdx, 0644o      ; permissions
    syscall
    
    mov r15, rax        ; file descriptor
    
    ; Запись заголовка
    mov rax, 1          ; sys_write
    mov rdi, r15
    mov rsi, magic_header
    mov rdx, 8
    syscall
    
    ; Запись информации о слоях
    mov rcx, num_layers
.layer_loop:
    mov rdi, [layers + rcx*8]
    call serialize_layer
    dec rcx
    jnz .layer_loop
    
    ; Закрытие файла
    mov rax, 3          ; sys_close
    mov rdi, r15
    syscall
    ret

serialize_layer:
    ; Сериализация одного слоя
    ; Запись типа слоя, размеров, весов
    ret