
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


np.random.seed(0)


class Departementale:
    ''' Simulation d'une route à une voie de taille L contenant n voitures
    ayant chacune une vitesse comprise entre 0 et vmax. Chaque voiture ayant
    une probabilité p de freiner à chaque pas de temps, des embouteillages
    peuvent apparaitre.

    Attributs
    ----------
    L : int
        Longueur de la route
    nb_cars : int
        Nombre de voitures sur la route
    vmax : int
        Vitesse maximale des voitures
    tmax : int
        Durée de la simulation
    dt : float
        Pas de temps
    proba_slow : float
        Probabilié qu'une voiture ralentisse à chaque pas de temps
        
    Méthodes
    --------
    linear_plot()
        Plot linéaire de la simulation
    polar_plot()
        Plot en vue polaire de la simulation
    combined_plot()
        Plot linéaire et polaire 
    '''
    
    def __init__(self, L=100, nb_cars=15, vmax=3, tmax=200, dt=0.1, proba_slow=0.25):
        self.L = L
        self.nb_cars = nb_cars
        self.vmax = vmax
        self.tmax = tmax
        self.dt = dt
        self.proba_slow = proba_slow

    def __road(self, cars):
        """positionne les voitures sur la route"""
        road = np.zeros([1, self.L])
        for car in cars:
            road[0][car[0]] = 1
        return road 

    def __create_cars(self):
        """initialise la position et la vitesse des voitures"""
        return np.array([[i, j] for i, j in zip(range(1, self.nb_cars*2, 2), 
                                                np.random.choice([1, 2, 3], 
                                                                 self.nb_cars))])

    def __traffic_density(self, cars):
        """calcul la densité du traffic"""
        r = self.__road(cars)
        traffic = np.zeros([1, self.L])
        for k in range(2, self.L - 2):
            traffic[0][k] = r[0][k-2]/4 + r[0][k-1]/2 + r[0][k] + r[0][k+1]/2 + r[0][k+2]/4
        return traffic

    def __simulation(self, cars):
        """simule le mouvement des voitures sur la route pour un pas de temps"""
        for ix, car in enumerate(cars):
            #-- probabilité d'accélération
            p = np.random.binomial(1, 1 - self.proba_slow)
            #-- ancienne position de la voiture k
            old_position = car[0]
            #-- détermination de la voiture k+1
            if ix == len(cars)-1:
                next_car = cars[0]
            else: 
                next_car = cars[ix+1]
            #-- détermination de la position t+1
            next_position = (car[0] + car[1]) % self.L 
            #-- On empêche les voitures de se doubler
            if ix != len(cars)-1 and (next_position >= next_car[0]) and (next_position - next_car[0] < 3):
                next_position = next_car[0] - 1
            #-- mise à jour de la position et de la vitesse
            car[0] = next_position
            if next_position < old_position:
                car[1] = car[1]
            else:
                car[1] = (min((car[0] - old_position) + 1, self.vmax) * p +  # accélération 
                          max((car[0] - old_position) - 1, 0) * (1-p))       # décélération
            #-- on empêche les voitures de dépasser L
            if next_position > self.L:
                car[0] = 0
            self.t += self.dt

    def linear_plot(self, fps=24, scrolling_display=False):
        """ Plot linéaire de la simulation  
        
        Arguments
        ---------
        fps : int
            images par seconde
        scrolling_display : bool
            Affichage défilant ou statique de la simulation
        """
        cars = self.__create_cars()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
        cmap_norm = Normalize(vmin=0, vmax=3)
        self.t = 0
        while self.t < self.tmax:
            self.__simulation(cars)
            ax1.clear()
            ax2.clear()
            ax1.imshow(self.__road(cars), cmap='magma')
            ax2.imshow(self.__traffic_density(cars), cmap='YlOrRd', norm=cmap_norm)
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax1.set_title('Simulation')
            ax2.set_title('Traffic density')
            if scrolling_display == True:
                ax1.set_xlim([np.min(cars[:, 0]) - 2, np.max(cars[:, 0]) + 2])
                ax2.set_xlim([np.min(cars[:, 0]) - 2, np.max(cars[:, 0]) + 2])
            plt.pause(1/fps)
        fig.show()

    def polar_plot(self, fps=24):
        """ Plot en vue polaire de la simulation  
        
        Arguments
        ---------
        fps : int
            images par seconde 
        """
        cars = self.__create_cars()
        theta = 2 * np.pi * np.linspace(0, 1, self.L)
        cmap_norm = Normalize(vmin=-1, vmax=2)
        self.fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(projection='polar')
        self.t = 0
        while self.t < self.tmax:
            self.__simulation(cars)
            ax1.clear()
            ax1.scatter(theta, np.ones([self.L])*1.1, c=self.__road(cars), 
                        cmap='Greys', norm=cmap_norm, s=15, zorder=2)
            ax1.plot(theta, np.ones([self.L])*1.1, c='#c6c6c6', lw=10, zorder=1)
            ax1.scatter(1, 1, color='w')        
            ax1.set_yticks([])
            ax1.set_xticks([])
            ax1.spines[:].set_visible(False)
            ax1.set_title('Simulation')
            plt.pause(1/fps)
        plt.show()

        
    def combined_plot(self, fps=24):
        """ Plot en vue polaire et linéaire de la simulation  
        
        Arguments
        ---------
        fps : int
            images par seconde #c6c6c6
        """
        cars = self.__create_cars()
        theta = 2 * np.pi * np.linspace(0, 1, self.L)
        cmap_norm = Normalize(vmin=0, vmax=3)
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(6, 1, figure=fig)
        ax1 = fig.add_subplot(gs[:4, 0], projection='polar')
        ax2 = fig.add_subplot(gs[4, 0])
        ax3 = fig.add_subplot(gs[5, 0])
        self.t = 0
        while self.t < self.tmax:
            self.__simulation(cars)
            ax1.clear()
            ax2.clear()
            ax3.clear()
            #-- polar plot
            ax1.scatter(theta, np.ones([self.L])*1.1, c=self.__road(cars), 
                        cmap='magma', s=40, zorder=2)
            ax1.plot(theta, np.ones([self.L])*1.1, c='#000004', lw=8, zorder=1)
            ax1.scatter(1, 1, color='w')        
            ax1.set_yticks([])
            ax1.set_xticks([])
            ax1.spines[:].set_visible(False)
            ax1.set_title('Simulation')
            #-- linear plot
            ax2.imshow(self.__road(cars), cmap='magma')
            ax3.imshow(self.__traffic_density(cars), cmap='YlOrRd', norm=cmap_norm)
            ax2.set_yticks([])
            ax3.set_yticks([])
            ax2.set_title('Simulation')
            ax3.set_title('Traffic density')  
            plt.pause(1/fps)
        plt.show()


if __name__ == '__main__':

   d91 = Departementale(nb_cars=35, L=100, tmax=500, proba_slow=.25, vmax=3)
   d77 = Departementale(nb_cars=60, L=300, tmax=4000, proba_slow=.3, vmax=4)
   
   #d91.linear_plot()
   #d77.polar_plot()
   #d91.combined_plot()
