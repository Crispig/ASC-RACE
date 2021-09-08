import os

path1 = "./dev/shm/RACE/test/high/"
path2 = "./dev/shm/RACE/test/middle/"
path3 = "./result/answers.json"

file_list1 = os.listdir(path1)
file_list3 = os.listdir(path2)

errors = 0

with open(path3, "r") as f:
    ff = f.read()
    content = ff.split('{')[-1].split('}')[0]
    content_list = content.split("], ")
    for c in content_list:
        fname = c.split('\"')[1]
        fanswers = c.split(': [')[1]
        fnum_a = len(fanswers.split(', '))
        if fname in file_list1:
            fname = path1 + fname 
        else:
            fname = path2 + fname
        with open(fname,"r") as pt:
            cont = pt.read()
            answers = cont.split('[')[1].split("]")[0]
            num_a = len(answers.split(", "))
            if fnum_a != num_a:
                print("error:",fname)
                print(num_a,"->",fnum_a)
                errors += 1
print("finish!",errors,"error",end="")
if errors > 1:
    print("s",end="\n")
else:
    print("\n")