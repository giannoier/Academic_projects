.data
msg: .asciiz "DWSE ENAN ARNHTIKO ARITHMO:"
.text
.globl main

main:
loop:
addi $v0,$0,4
la $a0, msg
syscall
addi $v0, $0, 5
syscall
add $t0, $v0, $0
bgtz,$t0,loop
add $t3,$t0,1
for:
beqz $t3,endfor
add $t0,$t0,$t3
add $t3,$t3,1
j for
endfor:
move $a0,$t0
li $v0,1
syscall
add $t0,$v0,$0
li $v0,10
syscall
