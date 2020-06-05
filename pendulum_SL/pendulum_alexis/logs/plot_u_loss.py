import matplotlib.pyplot as plt


loss = []
ep_id = []
fin = open('Loss_report_tf.out')
for line in fin:
    if 'u_loss' in line:
        #line = next(fin)
        epoch = 500
        e = 0
        for e in range(epoch):
            #print(line.split())
            line = next(fin)
            print(line.split()[0])
            #line = line.split()
            loss.append(float(line.split()[0]))
            ep_id.append(e)
            e += 1
fin.close()

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("U_loss")
plt.plot(ep_id,loss)
plt.savefig("U_loss.png")




