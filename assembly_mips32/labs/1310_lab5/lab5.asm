.data
msg1:.asciiz "Dwse to plithos:  "
msg2:.asciiz "Dwse ton aritmhmo: "
.align 2
pin1: .space 100
.text
.globl main
main:
addi $v0,$0,4
la $a0,msg1
syscall
addi $v0,$0,5
syscall
add $t0,$v0,0
addi $v0,$0,4
la $a0,msg2
syscall
addi $v0,$0,5
syscall
add $s2,$v0,0
addi $s2,$s2,1
addi $t4,$0,0
loop1:
addi $v0,$0,5
syscall 
add $t1,$0,$v0
sw $t1,pin1($t3)
addi $t3,$t3,4
addi $t4,$t4,1
blt $t4,$t0,loop1
la $a0,pin1
move $a1,$t0
move $a2,$s1
jal func
move $a0,$s1
li $v0,1
syscall
##exit##
li $v0,10
syscall
func:
addi $sp,$sp,-8
sw $ra,4($sp)
lw $t5,($a0)
addi $a1,$a1,-1
slt $t2,$t5,$s2
sw $t2,0($sp)
bne $a1,$0,L1
jr $ra
L1:
addi $a0,$a0,4
jal func
LL:
lw $t2,0($sp)
lw $ra,4($sp)
addi $sp,$sp,8
add $s1,$t2,$s1
jr $ra


