import numpy as np
import matplotlib.pyplot as plt

class Grid():
    def __init__(self):
        self.gamma = 0.9
        self.initial_state = [0,0]
        self.grid = np.zeros((5,5),"str")
        self.initialize_env()

    def initialize_env (self):
        for i in range(5):
            for j in range(5):
                if (i==2 and j==2):
                    self.grid[i][j] = "o"
                elif (i==3 and j==2):
                    self.grid[i][j] = "o"
                elif (i==4 and j==2):
                    self.grid[i][j] = "w"
                elif (i==4 and j==4):
                    self.grid[i][j] = "g"
                else:
                    self.grid[i][j] = "e"
        
    def nextstate(self,s,a):
        x,y = s
        x1,y1 = x,y
        if a == "AL":   
                y1=y1-1          
        elif a == "AU":
                x1=x1-1        
        elif a == "AR":   
                y1=y1+1       
        elif a == "AD":         
                x1=x1+1
           
        # wall or obstacle 
        if (x1<0 or x1>4 or y1<0 or y1>4 or self.grid[x1][y1]=="o"):
            x1 = x
            y1 = y
                
        return (x1,y1)

    def reward(self,s1):
        x,y = s1
        s1 = self.grid[x][y]
        if s1=="w":
            r = -10
        elif s1 == "g":
            r = 10
        else: 
            r = 0

        return r

    def terminal(self,s):
        x,y = s
        s = self.grid[x][y]
        if s=="g":
            return 1
        else:
            return 0

    def p(self,s,a):
        x,y = s
        p = np.zeros((5,5)) 

        # case 1 same direction
        if a == "AL":
            x1 = x
            y1 = y-1
        elif a == "AU":
            x1 = x-1
            y1 = y
        elif a == "AR":
            x1 = x
            y1 = y+1
        else: 
            x1 = x+1
            y1 = y
            
        if (x1<0 or x1>4 or y1<0 or y1>4 or self.grid[x1][y1]=="o"):
            x1 = x
            y1 = y      
        p[x1][y1] = 1

        
        return p 
    
    
def value_iteration(gamma):
    theta = 0.0001
    V = np.zeros((5,5))
    V1 = np.copy(V)
    A = ["AL","AU","AR","AD"]
    t = 0
    
    while True:
        delta = 0
        for i in range(5):
            for j in range(5):
                if grid.terminal((i,j)):
                    V1[i][j] = 0
                elif grid.grid[i][j]=="o":
                    V1[i][j] = 0
                else:
                    max_value = -np.inf
                    for a in A:
                        temp  = 0
                        prob = grid.p((i,j),a)
                        for p in range(5):
                            for q in range(5):
                                temp+= (prob[p][q])*(grid.reward((p,q))+gamma*V[p][q])
                                
                        max_value = max(max_value, temp)        
                    V1[i][j] = max_value
    
                
                delta = max(delta, np.abs(V[i][j]-V1[i][j]))
    
        V = np.copy(V1)
        t+=1
        if delta < theta:
            break

    # policy
    policy = np.zeros((5,5),"str")
    for i in range(5):
        for j in range(5):
                if grid.terminal((i,j)):
                    policy[i][j] = "G"
                elif grid.grid[i][j]=="o":
                    policy[i][j] = " "
                else:
                    opt_policy = "None"
                    max_value = -np.inf
                    for a in A:
                        temp  = 0
                        prob = grid.p((i,j),a)
                        for p in range(5):
                            for q in range(5):
                                temp+= (prob[p][q])*(grid.reward((p,q))+gamma*V[p][q])
    
                        if max_value < temp:
                            max_value = temp
                            opt_policy = a

                    if opt_policy == "AL":
                        policy[i][j] = "←"
                    elif opt_policy == "AU":
                        policy[i][j] = "↑"
                    elif opt_policy == "AR":
                        policy[i][j] = "→"
                    elif opt_policy == "AD":
                        policy[i][j] = "↓"
                        
    # Set print options
    np.set_printoptions(suppress=True, precision=4)

    return np.round(V, decimals=4), policy, t
            

def choose_action(Q,s,eps):
    r = np.random.rand()
    if r<eps:
        a =  np.random.randint(4)
    else:
        a = np.argmax(Q[s[0]][s[1]])

    if a==0:
        return a,"AL"
    elif a==1:
        return a,"AU"
    elif a==2:
        return a,"AR"
    else:
        return a,"AD"

def DynaQ(n,eps,alpha,gamma,T):
    Q = np.zeros((5,5,4))
    Model = np.zeros((5,5,4,3)) #r,s
    Model_visited = np.zeros((5,5,4)) # vistied 1 else 0

    mse_list = []
    while True:
        s = (np.random.randint(5),np.random.randint(5))
        if grid.grid[s[0]][s[1]] != "o" and grid.grid[s[0]][s[1]] != "g" :
            break
            
    for t in range(T):    
        a_int,a_str = choose_action(Q,s,eps)
        s1 = grid.nextstate(s,a_str)
        s1_backup = s1
        r = grid.reward(s1)
    
        Q[s[0]][s[1]][a_int] = Q[s[0]][s[1]][a_int] + alpha*(r+gamma*np.max(Q[s1[0]][s1[1]])-Q[s[0]][s[1]][a_int])
        
        Model[s[0]][s[1]][a_int][0] = r
        Model[s[0]][s[1]][a_int][1] = s1[0]
        Model[s[0]][s[1]][a_int][2] = s1[1]
        Model_visited[s[0]][s[1]][a_int] = 1
    
        for i in range(n):
            s_choice = np.where(np.sum(Model_visited,axis = 2)>0)
            s0_choice = s_choice[0]
            s1_choice =  s_choice[1]
            # print(s_choice)
            # print(s0_choice)
            # print(s1_choice)
            j = np.random.randint(len(s0_choice))
            s = (s0_choice[j],s1_choice[j])
            a_choice = np.where(Model_visited[s[0]][s[1]]>0)[0]   
            # print(a_choice)
            a = np.random.choice(a_choice)
            r = Model[s[0]][s[1]][a][0]
            s1 = (int(Model[s[0]][s[1]][a][1]),int(Model[s[0]][s[1]][a][2]))
            Q[s[0]][s[1]][a] = Q[s[0]][s[1]][a] + alpha*(r+gamma*np.max(Q[s1[0]][s1[1]])-Q[s[0]][s[1]][a])

        s = s1_backup
        if grid.terminal(s):
            while True:
                s = (np.random.randint(5),np.random.randint(5))
                if grid.grid[s[0]][s[1]] != "o" and grid.grid[s[0]][s[1]] != "g" :
                    break
        
        # MSE
        estimated_v = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                v = -np.inf
                for a in range(4):
                    v= max(v,Q[i][j][a])
                estimated_v[i][j] = v
        
        mse_list.append(1/25*np.linalg.norm(optimal_v-estimated_v)**2)

    
    # policy
    policy = np.zeros((5,5),"str")
    for i in range(5):
        for j in range(5):
                if grid.terminal((i,j)):
                    policy[i][j] = "G"
                elif grid.grid[i][j]=="o":
                    policy[i][j] = " "
                else:
                    a = np.argmax(Q[i][j])
                    if a == 0:
                        policy[i][j] = "←"
                    elif a == 1:
                        policy[i][j] = "↑"
                    elif a == 2:
                        policy[i][j] = "→"
                    elif a == 3:
                        policy[i][j] = "↓"
    
    return Q,policy,mse_list


grid = Grid()
    
# value iteration
optimal_v,policy,t = value_iteration(gamma = grid.gamma)
print("optimal value function \n{} ".format(optimal_v))
print("optimal policy \n{}".format(policy))
print("Number of iterations {}".format(t))


# Dyna-Q
trails = 20
avg_mse = np.zeros((trails,10000))
for i in range(trails):
    Q,policy,mse_list = DynaQ(n=5,eps=0.6,alpha=0.3,gamma=0.9,T=10000)
    avg_mse[i] = mse_list

    # print("Estimated value function \n{} ".format(Q))
    print("optimal policy \n{}".format(policy))

plt.plot(np.mean(avg_mse,0))
plt.xlabel("Episodes")
plt.ylabel("Average MSE")
plt.show()



