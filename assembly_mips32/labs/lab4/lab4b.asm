.data
msg1:.asciiz "Dwse to plithos twn stoixeiwn tou pinaka A: "
msg2:.asciiz "Dwse to plithos twn stoixeiwn tou pinaka B: "
msg3:.asciiz "Dwse tous arithmous pou uparxoun ston pinaka A: "
msg4:.asciiz "Dwse tous arithmous pou uparxoun ston pinaka B: "

.align 2
p1: .space 100
p2: .space 100
p3: .space 144
.text	
.globl main
	
main:

add $t6,$0,$0
add $t0,$0,$0
addi $v0,$0,4
la $a0, msg1
syscall
addi $v0,$0,5 		 ####diavazei to plithos gia ton pinakaA
syscall
add $t1,$v0,$0
addi $v0,$0,4
la $a0, msg2
syscall
addi $v0,$0,5 		 ####diavazei to plithos gia ton pinakaB
syscall
add $t8,$v0,$0
beqz $t1,loo
addi $v0,$0,4
la $a0, msg3
syscall


loop1:
addi $v0,$0,5    	 #diavazei k apothikeuei tous arithmous gia ton pinaka A
syscall 
add $t3,$0,$v0
sw $t3,p1($t6)      	 	
addi $t6,$t6,4
addi $t0,$t0,1
blt $t0,$t1,loop1
add $t0,$0,$0
loo:
beqz $t8,loo2
addi $v0,$0,4
la $a0, msg4
syscall
loop2:
addi $v0,$0,5 		 #diavazei tous arithmous gia ton pinaka B
syscall 
add $t4,$0,$v0
sw $t4,p2($t7)      	 #tous apothikeuei stous pinakes
addi $t7,$t7,4
addi $t0,$t0,1
blt $t0,$t8,loop2
loo2:
add $s4,$t8,$t1
la $a0,p1
move $a1,$t1
la $a2,p2
la $a3,p3
addi $sp,$sp,-4
sw $t8,4($sp)


jal sunxoneusi
add $t1,$0,$0

ektupwsi:
lw $a0,p3($s5)
li $v0,1
syscall
addi $t1,$t1,1
addi $s5,$s5,4
blt $t1,$s4,ektupwsi

exit:
li $v0,10
syscall

#####sinxoneysi####
sunxoneusi:
lw $t8,4($sp)
add $t0,$0,$0
add $t1,$a1,$t8
add $t5,$0,$0
add $t6,$0,$0
loop4:

lw $t2,($a0)
lw $t3,($a2)
bge $t5,$a1,periptwsi2
bge $t6,$t8,periptwsi1
slt ,$t4,$t2,$t3
beqz $t4,periptwsi2

periptwsi1:
sw $t2,($a3)
addi $a0,$a0,4
addi $t5,$t5,1
j exitsunx

periptwsi2:
bge $t6,$t8,exit2
sw $t3,($a3)
addi $a2,$a2,4
addi $t6,$t6,1
exitsunx:
addi $a3,$a3,4
addi $t0,$t0,1
blt $t0,$t1,loop4
exit2:
addi $sp,$sp,4
jr $ra
