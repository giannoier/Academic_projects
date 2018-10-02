.data
msg: .asciiz "Dwse enan akeraio arithmo apo 0-100: "
msgmi: .asciiz "Too small!"
msgme: .asciiz "Too big!"
msgequal: .asciiz "You guessed it with "
msgfa: .asciiz "But you are far away"
msg10: .asciiz "But you are very close"
msg0: "But you are close"
msgenter: "\n"
msgg: " tries!"
.text
.globl main

main:
add $t3,$0,$0
addi $t5,$0,25
addi $t4,$0,10
addi $v0,$0,5
syscall
add $t0,$v0,$0
loop:
addi,$t3,$t3,1
addi $v0,$0,4
la $a0,msg
syscall
addi $v0,$0,5

syscall
add $t1,$v0,$0
beq $t0,$t1,equal
bgt $t1,$t0,loopme
addi $v0,$0,4
la $a0,msgmi
syscall
j endif
loopme:
addi $v0,$0,4
la $a0,msgme
syscall
endif:
add $t2,$0,$0
sub $t2,$t0,$t1
bltz $t2,loop3
j diafora
loop3:
sub $t2,$0,$t2
diafora:
ble $t2,$t4,loop10
j loop25
loop10:
addi $v0,$0,4
la $a0,msg10
syscall
addi $v0,$0,4
la $a0,msgenter
syscall
j end
loop25:
bge $t2,$t5,loopfa
addi $v0,$0,4
la $a0,msg0
syscall
addi $v0,$0,4
la $a0,msgenter
syscall
j end
loopfa:
addi $v0,$0,4
la $a0,msgfa
syscall
addi $v0,$0,4
la $a0,msgenter
syscall
end:
bne $t0,$t1,loop
equal:
addi $v0,$0,4
la $a0,msgequal
syscall
li $v0, 1 
add $a0, $t3, $0 
syscall
addi $v0,$0,4
la $a0,msgg
syscall
li $v0,10
syscall










