w = list(model.parameters())
a = []
for i in range(10): 
 ab = w[i].cpu().detach().numpy()
 ab0= ab.flatten()
 a = np.append(a,ab0)