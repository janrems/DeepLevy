import matplotlib.pyplot as plt
import torch

class Plots():
    def __init__(self, fin_model, solver, evaluator, path):
        self.fin_model = fin_model
        self.losses = solver.losses
        self.initials = solver.initials
        self.evaluator = evaluator
        self.t = torch.linspace(0,fin_model.T, fin_model.R)
        self.path = path

    #Plots the loss from "start" onward
    def plot_losses(self, save=False, start=0):
        plt.plot(torch.linspace(start,len(self.losses), len(self.losses[start:])), self.losses[start:])
        if save:
            plt.savefig(self.path + "Loss.jpg", dpi=300)
        plt.show()

    # Plots the initials from "start" onward
    def plot_initials(self, save=False, start=0):
        plt.plot(torch.linspace(start, len(self.initials), len(self.initials[start:])), self.initials[start:])
        plt.axhline(y=self.fin_model.initial, color='r', linestyle='--', label="BS Option price")
        if save:
            plt.savefig(self.path + "Initials.jpg", dpi=300)
        plt.show()

    #Plot a randomly chosen market ralization
    def plot_market(self, i, save=False):

        opt = max(self.evaluator.stock[-1, i,0].detach() - self.fin_model.K, 0)
        plt.plot(self.t, self.evaluator.control[:, i,0].detach(), "black", label="Replicating portfolio")
        if self.fin_model.name == "BlackScholes":
            BS_portfolio = self.fin_model.black_scholes_portfolio(self.evaluator.stock, self.evaluator.wealth)
            plt.plot(self.t, BS_portfolio[:,i,0].detach(), color="black", linestyle="--", label="BS replicating portfolio")
        plt.plot(self.t, self.evaluator.wealth[:, i,0].detach(), "blue", label="Wealth process")
        plt.plot(self.t, self.evaluator.stock[:, i,0].detach(), label="Stock process")
        plt.axhline(y=opt, color='r', linestyle='-', label="Option payoff")
        plt.legend(loc="upper right")
        if save:
            plt.savefig(self.path + "Market.jpg", dpi=300)
        plt.show()

