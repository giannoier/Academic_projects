.data

msg: .asciiz "DWSTE ENAN THETIKO ARITHMO:"

.text
.globl main

main:
addi $v0,$0,4
la $a0, msg
syscall
add $t2,$0,$0
addi $t1,$0,3
addi $v0, $0, 5 
syscall
add $t0, $v0, $0
ble $t0,$t1,else
loop:
addi $t2,$t2,1
addi $t1,$t1,3
blt,$t1,$t0,loop
else:
move $a0,$t2
li $v0,1
syscall
add $t0,$v0,$0
li $v0,10
syscall
