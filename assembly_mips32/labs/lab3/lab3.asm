.data
msg1: .asciiz "o arithmos vrethike!"
msg2: .asciiz "dwse to plhthos ton arithmwn ston pinaka: "
msg3: .asciiz "dwse arithmo: "
msg4: .asciiz "dn vrethike"
.align 2
pinakas: .space 24
.text

main:
addi $v0,$0,4
la $a0, msg2
syscall

add $t1,$0,$0
addi $t3,$0,0
addi $v0,$0,5 #diavazei to plhthos twn  stoixeiwn toy pinaka
syscall
add $t0,$v0,$0
add $t1,$0,$0
for:
beq $t0,$t1,endfor
addi $v0,$0,4
la $a0, msg3
syscall
addi $t1,$t1,1
addi $v0,$0,5
syscall
add $t2,$0,$v0
sw $t2,pinakas($t3) #apothikeuontai oi times ston pinaka
addi $t3,$t3,4
j for
endfor:
add $t7,$0,$0
add $t8,$0,$0
addi $v0,$0,5
syscall
add $t4,$0,$v0   #diavazei thn timh pou psaxnoume ston pinaka

loop:

lw $t6,pinakas($t7)
beq $t6,$t4,vrethike

addi $t7,$t7,4
addi $t8,$t8,1
beq $t8,$t0,dnvrethike
j loop
vrethike:
addi $v0,$0,4
la $a0, msg1
syscall
#fdgffdgd

j end
dnvrethike:
addi $v0,$0,4
la $a0, msg4
syscall
end:
add $t0,$v0,$0
li $v0,10
syscall


