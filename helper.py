import numpy as np
def custom_name_transformer(x:np.array):
  a = np.asarray(x).reshape(-1,1)
  for i in range(len(a)):
    titles = ['Mr', 'Miss', 'Mrs', 'Master']
    for title in titles:
      if title in a[i][0]:
        a[i][0] = title
    if(a[i][0] not in titles):
      a[i][0] = 'Unkown'
  return a

def family_size(x:np.array):
  return (np.asarray(x).sum(axis=1) + 1).reshape(-1,1)
def remove_inf(x:np.array):
  return np.where(x == -np.inf, 0, x)
 