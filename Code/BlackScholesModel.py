import torch
import os
import numpy as np
from datetime import datetime

import DeepSolver
import Plots
import DataGenerator

import importlib
importlib.reload(DeepSolver)
importlib.reload(Plots)


#######################################
#Here you can set desired financial and machine learning parameters in dictionaries
#######################################
# Financial model parameters

financial_parameters = {
    'T': 1.0,                    # Maturity time
    'drift': [0.3],            # Drift
    'volatility': [0.2],  # Volatility
    's0': 1.0,                 # Spot price
    'r': 0,                    # Interest rate
    'K': 0.5,                  # Strike price
}

#######################################
# Machine learning model parameters

ml_parameters = {
    'M': 256,                    # Batch size
    'hidden_dim': 512,           # Number of neurons at each LSTM layer
    'R': 80,                    # Number of time-steps
    'epochs': 6000,                 # Number of epochs (training iterations)
    'learning_rate': 0.0005,     # Learning rate
    'eval_size': 10000            # Size of the evaluation set
}

#######################################

#Once parameters are set run this file


#######################################
#Creating paths and folders
#######################################

market_name = "BlackScholes"


current_datetime = datetime.now()

formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M")

#Here you can set the name of the folder manualy. Used for loading models
formatted_datetime = "2024_01_03_19_53"

market_path = market_name+"/"+ formatted_datetime +"/"

if not os.path.exists(market_path):
    os.makedirs(market_path)

graph_path = market_path + "Graphs/"
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

model_path = market_path  + "Model/"
if not os.path.exists(model_path):
    os.makedirs(model_path)


########################################
#Black-Scholes class
#######################################
class BlackScholes():
    def __init__(self,market_name,T,drift,volatility,s0,r,K,R):
        self.name = market_name
        self.T = T
        self.drift = torch.tensor(drift).unsqueeze(-1).float()
        self.volatility = torch.tensor(volatility).unsqueeze(-1).float()
        self.dim_BM = len(volatility)
        self.s0 = s0
        self.r = r
        self.K = K
        self.R = R
        self.dt = T/R
        self.initial = self.black_scholes_price()


    #Computes new instance of wealth according to the update rule prescribed by Euler-Maruyama scheme
    def new_wealth_instance(self,x, control, dB, i):
        dx = x * ((1-control) * self.r + control * self.drift) * self.dt +\
            x * control * torch.matmul(dB[i,:,:],self.volatility)
        return x + dx

    # Computes new instance of wealth according to the update rule prescribed by Euler-Maruyama scheme
    def new_stock_instance(self,s,dB,i):
        ds = s * self.drift * self.dt + \
             s * torch.matmul(dB[i, :, :], self.volatility)
        return s + ds

    #Computes the option claims given terminal stock price and dimensions of assets. This is for the case of
    # European Call option
    def claim(self,s, dim1, dim2):
        K_tensor = torch.ones(dim1, dim2) * self.K
        return torch.max(torch.zeros(dim1, dim2), s - K_tensor)

    #Computes the European call option price
    def black_scholes_price(self):
        normal = torch.distributions.normal.Normal(0,1)
        joint_volatility = torch.linalg.norm(self.volatility)
        d1 = (1 / (joint_volatility * np.sqrt(self.T))) * (np.log(self.s0 / self.K) + self.T * (self.r + 0.5 * joint_volatility ** 2))
        d2 = d1 - joint_volatility * np.sqrt(self.T)
        call = normal.cdf(d1) * self.s0 - normal.cdf(d2) * self.K * np.exp(-self.r * self.T)
        return float(call)

    #Computes the theoretical hedging portfolio ratio for European Call option
    def black_scholes_portfolio(self, stock, wealth):
        t_tensor = torch.linspace(0,self.T,self.R).unsqueeze(-1)
        t_tensor = t_tensor.unsqueeze(-1).repeat_interleave(stock.shape[1],1) #creates a time tensor with same shape
        #as stock
        joint_volatility = torch.linalg.norm(self.volatility)
        d1 = (torch.log(stock / self.K) + (self.r + 0.5 * joint_volatility ** 2) * (
                    torch.ones(stock.shape) * self.T - t_tensor)) / (
                     joint_volatility * torch.sqrt(self.T - t_tensor))
        normal = torch.distributions.normal.Normal(0, 1)
        portfolio = normal.cdf(d1)
        return portfolio*stock/wealth

    #gives a realization of black scholes market using theoretical price and portfolio
    def black_scholes_realization(self,M):
        stock = torch.ones(self.R, M)*self.s0
        wealth = torch.ones(self.R,M)*self.initial
        portfolio = torch.ones(self.R,M)
        dB = DataGenerator.gen_dB(self.R, M, self.dim_BM, self.T)
        stock_log = torch.log(stock)
        wealth_log = torch.log(wealth)
        for i in range(self.R):
            por = self.black_scholes_portfolio(stock,wealth)[i,:]
            portfolio[i,:] = por
            if i < self.R-1:
                x_log = self.new_log_wealth_instance(wealth_log[i,:], por, dB, i)
                s_log = self.new_log_stock_instance(stock_log[i,:], dB, i)
                wealth[i+1,:] = torch.exp(x_log)
                stock[i + 1, :] = torch.exp(s_log)
                stock_log[i+1,:] = s_log
                wealth_log[i+1,:] = x_log

        return wealth, stock, portfolio


#################################################
#Training
#################################################

#Initialize financial model
fin_model = BlackScholes("BlackScholes",financial_parameters["T"],financial_parameters["drift"],
                   financial_parameters["volatility"],financial_parameters["s0"],financial_parameters["r"],
                   financial_parameters["K"],ml_parameters["R"])

#Initialize deep neural network
net = DeepSolver.DeepNet(ml_parameters["R"],ml_parameters["hidden_dim"],ml_parameters["M"],fin_model)

#Initialize the solver and train the model
solver = DeepSolver.HedgeSolver(net,ml_parameters["learning_rate"],ml_parameters["epochs"])
solver.train()



#Save model weights, losses, and initial values
torch.save(solver.best_state_dict, model_path + "weights")
np.save(model_path+"losses", solver.losses)
np.save(model_path + "initials", solver.initials)


###################################################
#Evaluation
###################################################
with torch.no_grad():
    #Initialize evaluation net
    net_eval = DeepSolver.DeepNet(ml_parameters["R"], ml_parameters["hidden_dim"], ml_parameters["eval_size"], fin_model)

    #Load model weights and set in evaluation mode
    net_eval.load_state_dict(torch.load(model_path + "weights"))
    net_eval.eval()

    evaluator = DeepSolver.HedgeEvaluator(net_eval, fin_model)
    evaluator.eval()



# Make a text file with financial an ML information, training time and evaluated loss

with open(market_path + 'Info.txt', 'w') as file:
    # Write financial_data to the file
    file.write("Financial parameters:\n")
    for key, value in financial_parameters.items():
        file.write(f"{key}: {value}\n")

    # Add a separator between the two dictionaries
    file.write("\n---\n\n")

    # Write lstm_data to the file
    file.write("Machine learning parameters:\n")
    for key, value in ml_parameters.items():
        file.write(f"{key}: {value}\n")

    file.write("\n---\n\n")
    file.write("Loss: " + str(evaluator.loss))

    file.write("\n---\n\n")
    file.write("Initial: " + str(float(evaluator.wealth[0,0,0])))

    file.write("\n---\n\n")
    file.write("Theoretical Initial: " + str(fin_model.initial))

    file.write("\n---\n\n")
    file.write("Training time: " + str(solver.time) + " min")

    file.write("\n---\n\n")
    file.write("L^2 error: " + str(float(evaluator.l2)))



#######################################################################################################
#Make and save plots

ploter = Plots.Plots(fin_model, solver, evaluator, graph_path)

save = True #Whether you want to save the graphs
#save = False
start = 30

ploter.plot_losses(start=start, save=save)
ploter.plot_initials(start=start, save=save)

i = np.random.randint(ml_parameters["eval_size"])
ploter.plot_market(i, save=save)

