import csv

csvfile = open('Details_diffTol_diffPCA.csv', 'w', newline='') 
fieldnames = ['Iterations', 'Tolerane','Mae']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
f = open("Details_diffTol_diffPCA.txt","r")
f1 = f.readlines()
iterations =0 
mae =0
tol =0
for x in f1:
     splits = x.split(":")
     if splits[0] == "Iteration":
          iterations = int(splits[1])
          continue
     else:
          split_fur = splits[1].split(" ")
          mae = float(splits[2])
          tol = float(split_fur[0])
     print("tol:",tol," mae:",mae)
     writer.writerow({'Iterations': iterations, 'Tolerane': tol,'Mae':mae})
