def hanoi(n,src,dest,spare):

    if n > 0 and src!=dest: #something to do here
        fmoves = hanoi(n-1,src,spare,dest)

        smoves = hanoi(n-1,spare,dest,src)
        return fmoves + 1 + smoves
    else:
        return 0
data=[]
datas=0
count=0
first=True
p=0

#while True:

while(True):

    line=input()
    data=line.split(sep=",")
    if len(data)>2:
        print(hanoi(int(data[0]), data[1], data[2], 2))
    first=False


#move 3 disks from Peg 1 to Peg 3. Hence, Peg1 is the source or 'fromPeg'. Peg 3 is the desination or 'toPeg' and Peg 2 i
