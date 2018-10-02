.data
pin:.byte 'a','b','c','c','d','d','d','d','d','d','b','b','c','c','c','c','a',0x1B
.align 2
pin2:.byte 
.space 100
.text
.globl main
main:
addi $t8,$0,27
la $a0,pin
la $a1,pin2
jal comprss


add $t0,$0,$0

la $a0,pin2($s5)
li $v0, 4
syscall
addi $t0,$t0,1

exit:
li $v0,10
syscall


#######comprsss###########
comprss:
addi $t8,$0,27
addi $t2,$0,1
lw $t0,($a0)
addi $a0,$a0,4
lb $t1,($a0)
loop2:
beq $t0,$t1,print2
bgt $t2,4,print3
addi $t2,$0,1
sw $t0,($a1)
addi $a1,$a1,1
add $t0,$0,$t1
addi $a0,$a0,1
lb $t1,($a0)
beq $t1,$t8,exit2
j loop2
print2:
addi $t2,$t2,1
sw $t0,($a1)
addi $a1,$a1,1
add $t0,$0,$t1
addi $a0,$a0,1
lb $t1,($a0)
j loop2
print3:
sll $t3,$t2,2
sub $a1,$a1,$t3
sw $t8,($a1)
sw $t0,4($a1)
sw $t2,8($a1)
addi $a1,$a1,1
add $t0,$0,$t1
addi $a0,$a0,1
lb $t1,($a0)

exit2:

jr $ra
