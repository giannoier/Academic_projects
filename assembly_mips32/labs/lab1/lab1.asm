.text
.globl main
main:

addi $t1,$0,20
loop:
addi $t0,$t0,2
move $a0,$t0
li $v0,1
li $v0,1
syscall
bne $t0,$t1,loop
li $v0,10
syscall
