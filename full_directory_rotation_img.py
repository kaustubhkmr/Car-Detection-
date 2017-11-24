import os
os.chdir("validation")
i=0
for f in os.listdir():
    FILENAME,EXT = os.path.splitext(f)
    new_name = str(i)+'{}'.format(EXT)
    os.rename(f,new_name)
    i=i+1
            
