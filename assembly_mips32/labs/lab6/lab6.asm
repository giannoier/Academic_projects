.data
first:.word 0
msg1:.asciiz "Dwse 8etiko ari8mo" 
.text
.globl main

main:

li $v0,9                   ####dunamikh paraxwrish mnhmhs###
li $a0,8
syscall
move $s1,$v0
sw $s1,first
addi $v0,$0,4
la $a0, msg1
syscall
addi $v0,$0,5
syscall 
add $t0,$0,$v0
sw $t0,0($s1)
loop1:

addi $v0,$0,4
la $a0, msg1
syscall
addi $v0,$0,5
syscall 
add $t0,$0,$v0
move $a0,$t0
move $a1,$s1
jal InsertElement
beqz $a1,end               ###elegxei ti dieuthinsi, an einai iso me miden termatizei to programma####
bnez $t0,loop1             ###otan dwthei i timh 0 termatizei i eisodos arithmwn###
lw $s1,first
move $a0,$s1
jal Print
end:
li $v0,10
syscall

InsertElement:
move $t1,$a0
li $v0,9
li $a0,8
syscall
move $s1,$a1 
sw $v0,4($s1)		
move $s1,$v0
sw $t1,($s1)
jr $ra

Print:

loop2:
lw $t0,($s1)
move $a0,$t0
li $v0,1
syscall 
lw $s1,4($s1)
beqz $a0,end
jal loop2
